"""Streamlit 前端入口。

这个文件负责把本地 RAG 能力包装成可交互页面，包括资料上传、
知识库构建、聊天问答、来源引用和对话导出。
"""

import json
import gc
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

import streamlit as st

from src.document_loader import SUPPORTED_IMAGE_SUFFIXES, SUPPORTED_TEXT_SUFFIXES, load_documents
from src.qwen_api import qwen_chat
from src.rag_chain import create_rag_chain
from src.vectorstore import create_vectorstore, load_vectorstore


DATA_DIR = Path("./data")
VECTOR_DIR = Path("./chroma_db")
# 支持的文件类型由文本资料类型和图片资料类型合并而来。
# 上传文件时会用这个集合做过滤，避免无关文件进入知识库。
SUPPORTED_SUFFIXES = SUPPORTED_TEXT_SUFFIXES | SUPPORTED_IMAGE_SUFFIXES


# 页面基础配置：标题、浏览器标签图标和宽屏布局。
st.set_page_config(page_title="Multi-RAG Chat", page_icon="📚", layout="wide")


# Streamlit 每次交互都会重新运行脚本，因此对话历史需要保存在 session_state 中。
if "messages" not in st.session_state:
    st.session_state.messages = []
if "kb_summary" not in st.session_state:
    st.session_state.kb_summary = ""
if "kb_questions" not in st.session_state:
    st.session_state.kb_questions = ""


def list_data_files() -> list[Path]:
    """返回 data 目录中已经上传或手动放入的资料文件。"""
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(exist_ok=True)
    return sorted([file for file in DATA_DIR.glob("**/*") if file.is_file()])


def save_uploaded_files(uploaded_files) -> list[str]:
    """保存页面上传的文件，并过滤掉不支持的格式。"""
    DATA_DIR.mkdir(exist_ok=True)
    saved = []
    for uploaded_file in uploaded_files:
        # 只保留文件名，避免上传文件名中带路径造成目录穿越风险。
        file_name = Path(uploaded_file.name).name
        suffix = Path(file_name).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            continue
        target = DATA_DIR / file_name
        # Streamlit 上传文件是内存对象，这里写入 data/，后续统一从 data/ 构建知识库。
        target.write_bytes(uploaded_file.getbuffer())
        saved.append(file_name)
    return saved


def rebuild_knowledge_base(
    chunk_size: int,
    chunk_overlap: int,
    device: str,
    process_images: bool,
):
    """重新解析资料并构建 Chroma 向量库。"""
    # Windows 对正在使用的 Chroma 文件会加锁；重建前先释放缓存里的旧对象。
    st.cache_resource.clear()
    gc.collect()

    # 先把 PDF/TXT/DOCX/图片统一加载为 LangChain Document。
    # 这样后面的向量化流程不需要关心原始文件类型。
    docs = load_documents(str(DATA_DIR), process_images=process_images, use_image_cache=True)
    if not docs:
        return None, 0
    # 再进行文本切分、embedding 和向量库持久化。
    # chunk_size/chunk_overlap 由页面滑块传入，便于做课程实验对比。
    vectorstore = create_vectorstore(
        docs,
        persist_dir=str(VECTOR_DIR),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        device=device,
    )
    # 重建向量库后必须清理缓存，否则页面可能继续使用旧的向量库对象。
    st.cache_resource.clear()
    gc.collect()
    count = vectorstore._collection.count() if vectorstore else 0
    return vectorstore, count


def get_vector_count_from_sqlite() -> int:
    """从 Chroma 的 sqlite 文件读取文本块数量，避免为了显示状态而锁住 Chroma。"""
    sqlite_path = VECTOR_DIR / "chroma.sqlite3"
    if not sqlite_path.exists():
        return 0

    try:
        # 使用只读连接查询 embeddings 表，不创建 Chroma 客户端，降低 Windows 文件锁概率。
        db_uri = sqlite_path.resolve().as_uri() + "?mode=ro"
        with sqlite3.connect(db_uri, uri=True) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
            return int(cursor.fetchone()[0])
    except Exception:
        return 0


def format_distance(score: float) -> str:
    """格式化 Chroma 返回的距离分数。

    Chroma/LangChain 的 similarity_search_with_score 返回的不是“准确率”，
    而是向量距离。距离越小，表示问题向量和文档向量越接近。
    """
    return f"{score:.4f}"


def build_source_rows(results):
    """把检索结果整理成可展示、可导出的表格行。"""
    rows = []
    for i, (doc, score) in enumerate(results, 1):
        # 去掉多余换行，避免表格摘要被撑得过高。
        content = " ".join(doc.page_content.split())
        rows.append(
            {
                "排名": i,
                "文件名": doc.metadata.get("file_name", "未知文件"),
                "类型": doc.metadata.get("type", "document"),
                "检索距离": format_distance(score),
                "内容摘要": content[:180] + ("..." if len(content) > 180 else ""),
            }
        )
    return rows


def render_sources(results):
    """展示 RAG 检索到的参考来源和距离分数。"""
    if not results:
        st.info("没有检索到相关来源。")
        return

    rows = build_source_rows(results)
    # 先用简洁列表展示最重要的来源，方便用户快速判断答案依据。
    st.markdown("**参考来源**")
    for row in rows[:5]:
        st.markdown(
            f"{row['排名']}. `{row['文件名']}` "
            f"（{row['类型']}，检索距离 {row['检索距离']}，越小越相关）"
        )

    st.markdown("**检索结果表格**")
    st.caption("注意：这里显示的是向量检索距离，不是准确率；距离越小，代表排序越靠前。")
    st.dataframe(rows, use_container_width=True, hide_index=True)

    # 完整片段默认折叠，避免主聊天界面被大段原文淹没。
    with st.expander("查看完整检索片段", expanded=False):
        for i, (doc, score) in enumerate(results, 1):
            file_name = doc.metadata.get("file_name", "未知文件")
            file_type = doc.metadata.get("type", "document")
            st.markdown(f"**来源 {i}｜{file_type}｜`{file_name}`**")
            st.caption(f"检索距离：{format_distance(score)}，越小越相关")
            st.write(doc.page_content[:1000] + ("..." if len(doc.page_content) > 1000 else ""))
            st.divider()


def exportable_messages():
    """把对话历史转换成 JSON 可序列化的数据。"""
    messages = []
    for msg in st.session_state.messages:
        item = {"role": msg.get("role"), "content": msg.get("content", "")}
        if msg.get("sources"):
            # LangChain Document 对象不能直接 json.dumps，所以转成普通字典。
            item["sources"] = build_source_rows(msg["sources"])
        messages.append(item)
    return messages


def collect_knowledge_context(vectorstore, limit: int = 12) -> str:
    """从向量库中抽取一批文本片段，用于生成摘要和推荐问题。"""
    try:
        # 这里不是根据某个用户问题检索，而是直接从 Chroma 中取一批已有片段。
        # 目的是快速了解“当前知识库大概包含什么内容”。
        raw = vectorstore._collection.get(limit=limit, include=["documents", "metadatas"])
    except Exception:
        return ""

    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []
    snippets = []
    for i, text in enumerate(documents, 1):
        # metadata 中的 file_name/type 可以帮助模型判断片段来自 PDF、文本还是图片描述。
        metadata = metadatas[i - 1] if i - 1 < len(metadatas) else {}
        file_name = metadata.get("file_name", "未知文件")
        file_type = metadata.get("type", "document")
        clean_text = " ".join((text or "").split())
        # 每个片段截断到 800 字，避免 prompt 太长导致本地模型响应变慢。
        snippets.append(f"【片段 {i}｜{file_type}｜{file_name}】\n{clean_text[:800]}")
    return "\n\n".join(snippets)


def generate_knowledge_summary(context: str) -> str:
    """根据知识库片段生成课程展示用的资料摘要。"""
    if not context.strip():
        return "知识库中没有可用于生成摘要的文本片段。"

    # 这里把“摘要、知识点、关键词”明确写进 prompt，
    # 是为了让输出更像 NLP 课程作业中的知识库概览，而不是普通闲聊回答。
    prompt = f"""请根据下面的知识库片段，生成一个简洁的中文知识库概览。

要求：
1. 总结主要主题。
2. 提炼 3-6 个核心知识点。
3. 给出 5-8 个关键词。
4. 如果包含图片内容，也请说明图片大致涉及什么。
5. 不要编造片段中没有的信息。

知识库片段：
{context}
"""
    return qwen_chat([{"role": "user", "content": prompt}], temperature=0.2)


def generate_recommended_questions(context: str) -> str:
    """根据知识库片段生成适合测试系统的推荐问题。"""
    if not context.strip():
        return "知识库中没有可用于生成推荐问题的文本片段。"

    # 推荐问题可以帮助用户快速测试系统，也方便答辩时展示“系统能围绕资料提问”。
    # 要求问题必须能从知识库片段中找到依据，避免生成脱离资料的问题。
    prompt = f"""请根据下面的知识库片段，生成 8 个适合用户向 RAG 系统提问的问题。

要求：
1. 问题要覆盖概念解释、流程总结、细节查询、对比分析和图片理解。
2. 每个问题单独一行，用编号列表输出。
3. 问题必须能从知识库片段中找到依据，不要生成无关问题。

知识库片段：
{context}
"""
    return qwen_chat([{"role": "user", "content": prompt}], temperature=0.4)


@st.cache_resource(show_spinner="正在加载知识库...")
def get_rag_chain_and_vs(top_k: int, temperature: float, device: str):
    """加载向量库并创建 RAG 链；使用缓存避免每次刷新都重新加载模型。"""
    vectorstore = load_vectorstore(str(VECTOR_DIR), device=device)
    if vectorstore is None:
        return None, None
    chain = create_rag_chain(vectorstore, top_k=top_k, temperature=temperature)
    return chain, vectorstore


st.title("📚 Multi-RAG Chat")
st.caption("基于 RAG 的多模态本地知识库问答系统")

with st.sidebar:
    # 侧边栏集中放置资料管理、知识库构建和问答参数，主区域只保留聊天体验。
    st.header("📁 资料管理")
    # 用户可以通过页面上传资料，也可以直接把文件放进 data/ 文件夹。
    uploaded_files = st.file_uploader(
        "上传资料",
        type=["pdf", "txt", "docx", "jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        saved_files = save_uploaded_files(uploaded_files)
        if saved_files:
            st.success(f"已保存 {len(saved_files)} 个文件。")
        else:
            st.warning("没有可保存的受支持文件。")

    files = list_data_files()
    if files:
        st.write("**当前资料**")
        # 只展示前 10 个文件，避免资料较多时侧边栏过长。
        for file in files[:10]:
            st.caption(f"- {file.name}")
        if len(files) > 10:
            st.caption(f"... 共 {len(files)} 个文件")
    else:
        st.info("请先上传或放入资料文件。")

    st.divider()
    st.header("🧠 知识库")
    # 如果没有 NVIDIA GPU 或 CUDA 环境，可以切换到 cpu，速度会慢但更稳。
    device = st.selectbox("Embedding 设备", ["cuda", "cpu"], index=0)
    # chunk_size 越大，单个片段信息越完整，但检索粒度会变粗。
    chunk_size = st.slider("Chunk size", 300, 1500, 700, step=100)
    # chunk_overlap 用于保留相邻片段之间的上下文衔接。
    chunk_overlap = st.slider("Chunk overlap", 0, 300, 100, step=50)
    process_images = st.checkbox(
        "处理图片内容（较慢）",
        value=False,
        help="关闭时只构建 PDF/TXT/DOCX，可明显加快建库；开启后会调用视觉模型描述图片，并缓存结果。",
    )

    rebuild_clicked = st.button("重建知识库", type="primary", use_container_width=True)
    if rebuild_clicked:
        # 图片会调用本地视觉语言模型生成描述，耗时通常比纯文本更长。
        spinner_text = "正在解析资料并构建向量库..."
        if process_images:
            spinner_text = "正在解析资料并构建向量库，图片会调用视觉模型，首次处理可能较慢..."
        with st.spinner(spinner_text):
            try:
                _, count = rebuild_knowledge_base(chunk_size, chunk_overlap, device, process_images)
                if count:
                    st.success(f"知识库构建完成，共 {count} 个文本块。")
                else:
                    st.warning("没有构建出文本块，请检查 data 目录中的资料。")
            except Exception as e:
                st.error(f"知识库构建失败：{e}")

    if st.button("清空知识库", use_container_width=True):
        if VECTOR_DIR.exists():
            # 这里只删除向量库，不删除 data/ 中的原始资料。
            shutil.rmtree(VECTOR_DIR)
            st.cache_resource.clear()
            st.success("知识库已清空。")
            st.rerun()
        st.info("当前没有可清空的知识库。")

    vector_count = 0
    if VECTOR_DIR.exists():
        # 这里只读 sqlite 统计数量，不创建 Chroma 实例，避免重建时旧文件被占用。
        vector_count = get_vector_count_from_sqlite()
    if vector_count:
        st.success(f"已索引 {vector_count} 个文本块")
    else:
        st.warning("知识库尚未构建")

    image_files = [file for file in files if file.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES]
    if image_files:
        st.divider()
        st.header("🖼️ 图片预览")
        # 图片预览只展示前 3 张，让用户确认图片资料已被系统识别。
        for image in image_files[:3]:
            try:
                st.image(str(image), caption=image.name, width="stretch")
            except Exception:
                st.caption(f"{image.name} 预览失败")

    st.divider()
    st.header("⚙️ 问答参数")
    # top_k 控制每次提问时取回多少个相关片段，数值越大上下文越多。
    top_k = st.slider("检索数量 top_k", 3, 12, 6)
    # temperature 越高回答越发散，越低越稳定，课程展示建议保持中低值。
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)

    st.divider()
    st.header("💬 对话管理")
    if st.button("清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.session_state.get("messages"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 导出的 JSON 可作为实验记录，方便比较不同参数下的问答效果。
        export_data = {
            "export_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "messages": exportable_messages(),
        }
        st.download_button(
            label="导出对话 JSON",
            data=json.dumps(export_data, ensure_ascii=False, indent=2),
            file_name=f"Multi_RAG_Chat_{timestamp}.json",
            mime="application/json",
            use_container_width=True,
        )


with st.expander("📌 知识库概览与推荐问题", expanded=True):
    st.caption("基于当前向量库抽取文本片段，生成摘要和可测试问题。适合课程展示和答辩演示。")
    col_summary, col_questions = st.columns(2)

    with col_summary:
        # 这个按钮不会修改知识库，只读取已有向量库片段并让大模型做摘要。
        if st.button("生成知识库摘要", use_container_width=True):
            with st.spinner("正在生成知识库摘要..."):
                try:
                    # 只有知识库已经构建后，才能从 Chroma 中抽取片段生成摘要。
                    summary_vs = load_vectorstore(str(VECTOR_DIR), device=device)
                    if summary_vs is None:
                        st.warning("请先构建知识库。")
                    else:
                        context = collect_knowledge_context(summary_vs)
                        st.session_state.kb_summary = generate_knowledge_summary(context)
                except Exception as e:
                    st.error(f"摘要生成失败：{e}")

    with col_questions:
        # 推荐问题用于降低演示门槛：用户不知道问什么时，可以直接参考这里。
        if st.button("生成推荐问题", use_container_width=True):
            with st.spinner("正在生成推荐问题..."):
                try:
                    question_vs = load_vectorstore(str(VECTOR_DIR), device=device)
                    if question_vs is None:
                        st.warning("请先构建知识库。")
                    else:
                        context = collect_knowledge_context(question_vs)
                        st.session_state.kb_questions = generate_recommended_questions(context)
                except Exception as e:
                    st.error(f"推荐问题生成失败：{e}")

    if st.session_state.kb_summary:
        # 结果放在 session_state 中，避免页面刷新后立刻消失。
        st.markdown("**知识库摘要**")
        st.markdown(st.session_state.kb_summary)

    if st.session_state.kb_questions:
        st.markdown("**推荐问题**")
        st.markdown(st.session_state.kb_questions)


for msg in st.session_state.messages:
    # 历史消息重放，让 Streamlit 重新运行脚本后仍能看到完整对话。
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

if prompt := st.chat_input("输入你的问题..."):
    # 用户消息先加入历史，再立即渲染到页面上。
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    full_response = ""
    source_results = []
    with st.chat_message("assistant"):
        # message_placeholder 用于流式刷新模型输出。
        message_placeholder = st.empty()
        # sources_placeholder 用于在回答下方展示检索证据。
        sources_placeholder = st.container()

        with st.spinner("正在检索知识库并生成回答..."):
            try:
                chain, vectorstore = get_rag_chain_and_vs(top_k, temperature, device)
                if chain is None or vectorstore is None:
                    st.error("请先上传资料并重建知识库。")
                    st.stop()

                # 先单独检索一次，用于后续展示“回答依据”；RAG 链内部也会检索一次用于生成。
                source_results = vectorstore.similarity_search_with_score(prompt, k=top_k)
                for chunk in chain.stream(prompt):
                    # 不同 LangChain 版本返回的 chunk 类型可能不同，所以做兼容处理。
                    full_response += chunk.content if hasattr(chunk, "content") else str(chunk)
                    message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)
                with sources_placeholder:
                    render_sources(source_results)
            except Exception as e:
                full_response = f"发生错误：{e}"
                st.error(full_response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response,
            # 保存来源结果，页面重跑后也能恢复检索表格。
            "sources": source_results,
        }
    )

st.caption("Multi-RAG Chat | 文档上传 + 一键建库 + 来源引用 + 检索可视化")
