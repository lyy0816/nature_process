"""向量库模块。

负责把 Document 切分成文本块，使用 embedding 模型编码成向量，
并保存到本地 Chroma 数据库中。
"""

import os
import shutil
import time
import gc

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def remove_existing_vectorstore(persist_dir: str, retries: int = 5, delay: float = 0.5):
    """删除旧的向量库目录。

    Windows 上 Chroma 文件锁释放较慢，因此加入重试机制：
    每次失败后 sleep 等待并 gc.collect()，超过重试次数则抛出 RuntimeError。
    """
    if not os.path.exists(persist_dir):
        return

    for attempt in range(1, retries + 1):
        try:
            gc.collect()
            shutil.rmtree(persist_dir)
            return
        except PermissionError as e:
            if attempt == retries:
                raise RuntimeError(
                    "旧知识库文件仍被占用。请关闭正在运行的 Streamlit 页面或终端，"
                    "重新启动应用后再重建知识库。"
                ) from e
            time.sleep(delay)


def create_embeddings(device: str = "cuda"):
    """创建 BAAI/bge-m3 Embedding 模型实例。

    Args:
        device: 推理设备，"cuda" 使用 GPU，"cpu" 使用 CPU。

    Returns:
        HuggingFaceEmbeddings 实例，已配置归一化，适用于 Chroma 向量检索。

    bge-m3 支持中英文语义表示，向量归一化后可使用内积或余弦相似度进行比较。
    """
    # normalize_embeddings=True 会把向量归一化，便于使用相似度进行比较。
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_vectorstore(
    documents,
    persist_dir: str = "./chroma_db",
    chunk_size: int = 700,
    chunk_overlap: int = 100,
    device: str = "cuda",
):
    """将 Document 列表切分、向量化并持久化到本地 Chroma 数据库。

    Args:
        documents: LangChain Document 列表。
        persist_dir: Chroma 持久化目录，默认 chroma_db/。
        chunk_size: 文本切分块大小（字符数）。
        chunk_overlap: 相邻块之间的重叠字符数，用于保持跨块上下文连贯。
        device: Embedding 推理设备。

    Returns:
        Chroma 向量库实例，或 None（无文档时）。
    """
    if not documents:
        print("没有可处理的文档。")
        return None

    print("正在切分文本...")
    # overlap 可以减少跨段落信息被硬切断的问题，有利于检索完整上下文。
    # separators 按从强到弱的边界切分：段落、换行、中文标点、空格、字符。
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    splits = text_splitter.split_documents(documents)

    print("正在向量化...")
    # bge-m3 支持中英文语义表示，适合课程资料和中文问答场景。
    # 这里的 embedding 决定了“问题”和“文档片段”能否在语义空间中靠近。
    embeddings = create_embeddings(device=device)

    if os.path.exists(persist_dir):
        # 重建知识库时删除旧向量库，避免旧资料和新资料混在一起。
        remove_existing_vectorstore(persist_dir)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="multi_rag",
    )
    # persist_directory 指定后，Chroma 会把向量数据保存到本地文件夹。
    print(f"向量库构建完成，共 {len(splits)} 个文本块。")
    return vectorstore


def load_vectorstore(persist_dir: str = "./chroma_db", device: str = "cuda"):
    """加载已持久化的 Chroma 向量数据库。

    必须使用与构建时相同的 embedding 模型，否则查询向量空间不一致。

    Args:
        persist_dir: Chroma 持久化目录。
        device: Embedding 推理设备。

    Returns:
        Chroma 向量库实例，或 None（目录不存在时）。
    """
    if not os.path.exists(persist_dir):
        print("向量库不存在，请先构建。")
        return None

    # 加载时必须使用与构建时同类的 embedding 模型，否则查询向量空间不一致。
    # 用户提问时，问题也会被同一个模型编码成向量，再去 Chroma 中找近邻。
    embeddings = create_embeddings(device=device)
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="multi_rag",
    )
    print(f"向量库加载成功，共 {vectorstore._collection.count()} 个文本块。")
    return vectorstore
