"""命令行 RAG 测试脚本。

这个脚本用于快速验证知识库是否可用，不依赖 Streamlit 页面。
适合调试检索质量、模型连接和 RAG 回答效果。
"""

from src.rag_chain import create_rag_chain
from src.vectorstore import load_vectorstore


if __name__ == "__main__":
    # 这个脚本用于最小化测试：不启动网页，只在终端里输入问题。
    print("=" * 70)
    print("🚀 Multi-RAG Chat 命令行测试")
    print("=" * 70)

    # 先加载已经由 build_kb.py 或页面按钮构建好的本地向量库。
    # 如果 chroma_db/ 不存在，说明还没有完成知识库构建。
    vectorstore = load_vectorstore("./chroma_db")
    if not vectorstore:
        print("知识库不存在，请先运行 python build_kb.py 或在页面中重建知识库。")
        exit()

    # 创建 RAG 链：检索器负责找资料，大模型负责基于资料生成回答。
    print("正在创建 RAG 链...")
    rag_chain = create_rag_chain(vectorstore, top_k=6, temperature=0.7)

    print("\n系统就绪！输入 'exit'、'quit' 或 '退出' 结束测试。\n")

    while True:
        question = input("你: ")
        # 提供多种退出词，方便中文和英文输入习惯。
        if question.lower() in ["exit", "quit", "退出"]:
            print("再见！")
            break
        if not question.strip():
            # 空输入不调用模型，避免浪费本地推理资源。
            continue

        print("思考中...")
        try:
            # invoke 会执行完整 RAG 流程：检索相关片段 -> 组装 Prompt -> 调用模型。
            response = rag_chain.invoke(question)
            print(f"\n助手: {response}\n")
        except Exception as e:
            print(f"错误: {e}")
