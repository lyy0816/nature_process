"""Multi-RAG Chat 核心模块。

本包包含资料加载、向量库管理、RAG 问答链和 Qwen API 调用等子模块，
对外统一通过 __all__ 暴露常用接口。
"""

from .document_loader import load_documents
from .vectorstore import create_vectorstore, load_vectorstore
from .rag_chain import create_rag_chain

__all__ = ["load_documents", "create_vectorstore", "load_vectorstore", "create_rag_chain"]