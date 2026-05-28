"""命令行知识库构建脚本。

适合在不启动 Streamlit 页面的情况下，手动把 data/ 目录中的资料
解析、切分、向量化，并写入 chroma_db/。
"""

import argparse

from src.document_loader import load_documents
from src.vectorstore import create_vectorstore


def parse_args():
    """解析命令行参数。

    Returns:
        argparse.Namespace，包含:
        - skip_images: 是否跳过图片内容处理。
        - device: Embedding 设备（cuda/cpu）。
    """
    parser = argparse.ArgumentParser(description="构建 Multi-RAG Chat 本地知识库")
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="跳过图片内容描述，只处理 PDF/TXT/DOCX，可显著加快建库。",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Embedding 设备，默认使用 cuda；没有 GPU 时使用 cpu。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 这个脚本是“离线建库”入口：先准备知识库，再启动问答。
    # 如果使用 Streamlit 页面，也可以通过左侧“重建知识库”完成同样操作。
    print("=" * 75)
    print("🚀 Multi-RAG Chat 多模态知识库构建工具")
    print("使用 BAAI/bge-m3 构建向量库")
    if args.skip_images:
        print("当前为快速模式：跳过图片内容处理")
    else:
        print("当前会处理图片：首次会调用 Qwen2.5-VL:7b，速度较慢，结果会缓存")
    print("=" * 75)

    # 第一步：读取 data 目录中的资料。
    # 图片处理可通过 --skip-images 关闭，适合只想快速测试文本资料的情况。
    docs = load_documents("./data", process_images=not args.skip_images, use_image_cache=True)

    # 第二步：将文档切分成 chunk，并写入 Chroma 向量数据库。
    # 构建结果会保存到 chroma_db/，下次问答时可以直接加载。
    if docs:
        vectorstore = create_vectorstore(docs, device=args.device)
        if vectorstore:
            print("\n🎉 知识库构建完成，可以运行 app.py 或 test_rag.py 进行问答。")
    else:
        print("未找到可处理的文档，请检查 data 文件夹。")
