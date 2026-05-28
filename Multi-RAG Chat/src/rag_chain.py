"""RAG 链路模块。

把“用户问题 -> 向量检索 -> Prompt 组装 -> Qwen API 回答”
串成一个带 invoke/stream 方法的简单问答链。
"""

from src.qwen_api import qwen_chat, qwen_chat_stream

# 如果想切回 LangChain + 本地 Ollama，可以恢复旧版写法：
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_ollama import ChatOllama


class QwenRagChain:
    """轻量 RAG 问答链，基于 Qwen API 的检索增强生成。

    工作流程：用户问题 -> 向量相似度检索 -> 组装 Prompt（含上下文）-> Qwen 生成回答。

    Attributes:
        vectorstore: Chroma 向量库实例，用于检索相关文档片段。
        top_k: 每次检索取回的文档片段数量。
        temperature: 模型生成温度参数。
    """

    def __init__(self, vectorstore, top_k=6, temperature=0.7):
        """初始化 RAG 链。

        Args:
            vectorstore: Chroma 向量库实例。
            top_k: 检索返回的文档片段数，越大上下文越丰富但 Prompt 越长。
            temperature: 生成温度，越高越有创造性，越低越确定。
        """
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.temperature = temperature

    def _format_docs(self, docs) -> str:
        """把检索到的 Document 列表格式化为模型可读的带来源标记的上下文字符串。"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            file_name = doc.metadata.get("file_name", "未知文件")
            file_type = doc.metadata.get("type", "document")
            formatted.append(f"【来源 {i}｜{file_type}｜{file_name}】\n{doc.page_content}")
        return "\n\n".join(formatted) if formatted else "（无相关检索结果）"

    def _build_messages(self, question: str):
        """检索相关片段并组装 Qwen Chat Completions 消息。

        先调用向量库检索 top_k 个相关文档，再拼接系统指令、上下文和用户问题。
        """
        docs = self.vectorstore.similarity_search(question, k=self.top_k)
        context = self._format_docs(docs)
        prompt = f"""你是一个专业的多模态智能文档问答助手。

【回答规则】
1. 如果用户问题与知识库中的文档、PDF、图片或图表有关，必须优先依据下方上下文回答。
2. 如果上下文中没有足够信息，请明确说明“知识库中没有找到充分依据”，不要编造细节。
3. 回答要使用清晰、专业的中文。
4. 如果问题是通用知识、闲聊或编程问题，可以正常回答，但要说明是否使用了知识库信息。

【知识库上下文】
{context}

【用户问题】
{question}

请给出回答："""
        return [{"role": "user", "content": prompt}]

    def invoke(self, question: str) -> str:
        """执行完整 RAG 流程并一次性返回完整回答。"""
        return qwen_chat(self._build_messages(question), temperature=self.temperature)

    def stream(self, question: str):
        """执行完整 RAG 流程并以生成器方式流式返回回答片段。"""
        yield from qwen_chat_stream(self._build_messages(question), temperature=self.temperature)


def create_rag_chain(vectorstore, top_k=6, temperature=0.7):
    """工厂函数：创建基于 Qwen 多模态 API 的 RAG 问答链。

    Args:
        vectorstore: Chroma 向量库实例。
        top_k: 检索返回的文档片段数。
        temperature: 生成温度参数。

    Returns:
        QwenRagChain 实例，支持 invoke(question) 和 stream(question) 两种调用方式。
    """
    return QwenRagChain(vectorstore, top_k=top_k, temperature=temperature)
