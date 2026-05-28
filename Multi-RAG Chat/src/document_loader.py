"""资料加载模块。

把不同来源的资料统一转换成 LangChain Document：
- PDF/TXT/DOCX 直接解析为文本；
- 图片先通过视觉语言模型生成中文描述，再作为文本进入知识库。
"""

import base64
import json
import mimetypes
from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from tqdm import tqdm

from src.qwen_api import qwen_chat

# 如果想切回本地 Ollama，可以取消下面这一行和 describe_image 中的 Ollama 代码注释。
# from langchain_ollama import ChatOllama


SUPPORTED_TEXT_SUFFIXES = {".pdf", ".txt", ".docx"}
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _image_cache_path(data_path: Path) -> Path:
    """图片描述缓存文件路径。"""
    return data_path / ".image_descriptions.json"


def _load_image_cache(data_path: Path) -> dict:
    """读取图片描述缓存，避免每次重建都重复调用视觉模型。"""
    cache_path = _image_cache_path(data_path)
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_image_cache(data_path: Path, cache: dict):
    """保存图片描述缓存。"""
    cache_path = _image_cache_path(data_path)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _image_cache_key(image_path: Path) -> str:
    """用路径、文件大小和修改时间组成缓存 key，图片变化后会自动重新描述。"""
    stat = image_path.stat()
    return f"{image_path.resolve()}::{stat.st_size}::{stat.st_mtime_ns}"


def _image_to_data_url(image_path: Path) -> str:
    """把本地图片转成 base64 data URL，供 Qwen-VL API 读取。"""
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def describe_image(image_path: Path, vision_model: str | None = None) -> str:
    """使用 Qwen 多模态 API，把图片转换成可检索的文本描述。"""
    try:
        # 图片无法直接进入文本向量库，因此先用多模态模型生成描述。
        # 生成的描述越详细，后续用户针对图片提问时越容易被检索到。
        prompt = (
            "请用中文详细描述这张图片，包括主要内容、图表数据、文字信息、"
            "整体含义等。描述要清晰、专业，便于后续知识库检索。"
        )

        return qwen_chat(
            [
                {
                    "role": "user",
                    "content": [
                        # Qwen-VL 的 OpenAI 兼容接口需要文本 prompt 和 image_url。
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": _image_to_data_url(image_path)}},
                    ],
                }
            ],
            temperature=0.3,
            model=vision_model,
        )

        # 原本的本地 Ollama 写法保留在这里，如需切回本地模型，注释上面的 qwen_chat，取消下面代码注释。
        # llm = ChatOllama(model="qwen2.5vl:7b", temperature=0.3, num_gpu=999)
        # response = llm.invoke(
        #     [
        #         {
        #             "role": "user",
        #             "content": [
        #                 {"type": "text", "text": prompt},
        #                 {"type": "image_url", "image_url": str(image_path)},
        #             ],
        #         }
        #     ]
        # )
        # return response.content.strip()
    except Exception as e:
        # 图片处理失败时返回一段占位文本，避免单张图片错误中断整个知识库构建。
        print(f"图片描述生成失败 {image_path.name}: {e}")
        return f"图片描述生成失败：{image_path.name}"


def _add_metadata(docs: list[Document], file: Path, file_type: str) -> list[Document]:
    """给每个文档片段补充来源信息，方便后续展示引用来源。"""
    for doc in docs:
        doc.metadata.update(
            {
                "source": str(file),
                "file_name": file.name,
                "type": file_type,
            }
        )
    return docs


def load_documents(data_dir: str = "./data", process_images: bool = True, use_image_cache: bool = True):
    """加载 data 目录中的资料，并返回 Document 列表。"""
    documents = []
    data_path = Path(data_dir)

    if not data_path.exists():
        # 首次运行项目时 data/ 可能还不存在，自动创建能降低使用门槛。
        data_path.mkdir(exist_ok=True)
        print(f"已创建数据目录：{data_dir}")
        return documents

    # 先递归收集全部文件，再按后缀分类处理。
    # 这样用户可以在 data/ 下建立子目录管理资料。
    all_files = [file for file in data_path.glob("**/*") if file.is_file()]
    pdf_files = [file for file in all_files if file.suffix.lower() == ".pdf"]
    txt_files = [file for file in all_files if file.suffix.lower() == ".txt"]
    docx_files = [file for file in all_files if file.suffix.lower() == ".docx"]
    image_files = [file for file in all_files if file.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES]

    print(
        "找到文件 -> "
        f"PDF:{len(pdf_files)} | TXT:{len(txt_files)} | "
        f"DOCX:{len(docx_files)} | 图片:{len(image_files)}"
    )

    for file in tqdm(pdf_files, desc="加载 PDF"):
        try:
            # PyPDFLoader 会按页加载 PDF，页码等信息会保留在 metadata 中。
            loader = PyPDFLoader(str(file))
            documents.extend(_add_metadata(loader.load(), file, "pdf"))
        except Exception as e:
            print(f"PDF 加载失败 {file.name}: {e}")

    for file in tqdm(txt_files, desc="加载 TXT"):
        try:
            # 课程资料通常是 UTF-8 文本；如果遇到 GBK 文件，可以在这里扩展自动探测。
            loader = TextLoader(str(file), encoding="utf-8")
            documents.extend(_add_metadata(loader.load(), file, "txt"))
        except Exception as e:
            print(f"TXT 加载失败 {file.name}: {e}")

    for file in tqdm(docx_files, desc="加载 DOCX"):
        try:
            # Word 文档通常包含课程报告或笔记，先提取纯文本再进入 RAG。
            loader = Docx2txtLoader(str(file))
            documents.extend(_add_metadata(loader.load(), file, "docx"))
        except Exception as e:
            print(f"DOCX 加载失败 {file.name}: {e}")

    if process_images:
        image_cache = _load_image_cache(data_path) if use_image_cache else {}
        cache_changed = False

        for file in tqdm(image_files, desc="处理图片"):
            try:
                cache_key = _image_cache_key(file)
                if use_image_cache and cache_key in image_cache:
                    desc = image_cache[cache_key]
                else:
                    desc = describe_image(file)
                    if use_image_cache:
                        image_cache[cache_key] = desc
                        cache_changed = True

                # 把图片描述包装成 Document，使图片也能参与同一套向量检索流程。
                # metadata 中保留 type=image，前端可以据此显示图片来源。
                content = f"【图片文件】{file.name}\n\n{desc}"
                documents.append(
                    Document(
                        page_content=content,
                        metadata={"source": str(file), "file_name": file.name, "type": "image"},
                    )
                )
            except Exception as e:
                print(f"图片处理失败 {file.name}: {e}")

        if use_image_cache and cache_changed:
            _save_image_cache(data_path, image_cache)
    elif image_files:
        print(f"已跳过 {len(image_files)} 张图片；如需图片问答，请启用图片内容处理。")

    print(f"总共加载 {len(documents)} 个文档或图片描述")
    return documents
