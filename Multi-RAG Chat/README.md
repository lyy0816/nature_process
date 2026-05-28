# Multi-RAG Chat

## 项目简介

Multi-RAG Chat 是一个面向自然语言处理课程期末大作业的本地知识库问答系统。项目基于检索增强生成（Retrieval-Augmented Generation, RAG）思想，将用户本地的 PDF、TXT、DOCX 文档以及图片内容转换为可检索的知识库，并通过大语言模型完成问答。

与普通聊天机器人不同，本项目的回答不是只依赖模型自身参数知识，而是先从本地知识库中检索与问题最相关的内容，再将检索结果作为上下文交给大语言模型生成答案，从而降低幻觉、增强回答的可追溯性和领域适应能力。

## 项目特点

- 支持多种数据来源：PDF、TXT、DOCX、JPG、PNG、JPEG。
- 支持图片内容理解：使用多模态视觉模型对图片生成文本描述，再进入 RAG 知识库，首次处理后缓存结果避免重复调用。
- 使用语义向量检索：通过 `BAAI/bge-m3` 将文本片段编码为语义向量。
- 使用 Chroma 构建本地向量数据库。
- 默认使用阿里云百炼 DashScope 的 Qwen 多模态 API（`qwen3-vl-plus`），同时保留了本地 Ollama 的切换能力。
- 使用 Streamlit 构建可交互的聊天界面。
- 支持在页面中上传资料，并一键重建本地知识库。
- 支持调节 chunk size、chunk overlap、top_k、temperature 等参数。
- 支持自动生成知识库摘要和推荐问题，便于课程展示和快速测试。
- 支持聊天历史记录展示和 JSON 导出。
- 支持展示回答引用来源、检索距离和检索结果表格，便于观察 RAG 的检索过程。

## 技术路线

系统整体流程如下：

```text
本地资料
  ├── PDF / TXT / DOCX
  └── 图片文件
        ↓
文档解析与图片描述生成（Qwen VL API / Ollama 本地模型）
        ↓
文本切分 Chunking
        ↓
Embedding 语义向量化（BAAI/bge-m3）
        ↓
Chroma 向量数据库
        ↓
用户问题向量检索
        ↓
构造 Prompt 上下文
        ↓
Qwen API 生成回答（或切回 Ollama 本地模型）
        ↓
Streamlit 页面展示
```

## 核心 NLP 技术

### 1. 文档解析

项目使用 LangChain 提供的文档加载器读取不同类型的数据文件：

- `PyPDFLoader`：读取 PDF 文档。
- `TextLoader`：读取 TXT 文本文档。
- `Docx2txtLoader`：读取 Word 文档。

图片文件会先通过多模态模型（Qwen VL 或 Ollama `qwen2.5vl:7b`）生成自然语言描述，结果缓存在 `data/.image_descriptions.json`，再作为文本进入后续处理流程。

### 2. 文本切分

长文档会被切分为较小的文本片段，便于向量检索和上下文拼接。项目使用 `RecursiveCharacterTextSplitter`，按段落、换行、中文标点、空格、字符的优先级逐级切分，并设置 overlap 以尽量保留上下文连续性。

### 3. 语义向量表示

项目使用 `BAAI/bge-m3` 作为 embedding 模型，将文本片段转换为高维语义向量。bge-m3 支持中英文语义表示，向量经过归一化处理。相比关键词匹配，语义向量检索可以更好地处理同义表达、上下文相似和自然语言问题。

### 4. 向量检索

项目使用 Chroma 作为本地向量数据库，向量数据持久化到 `chroma_db/` 目录。用户提问后，系统会从向量库中检索最相关的若干个文本片段，并将这些片段作为大语言模型的上下文。检索结果中包含向量距离分数（距离越小越相关）。

### 5. 检索增强生成

项目构建了一个轻量 RAG 链（`QwenRagChain`）：

1. 用户输入问题。
2. 检索器从知识库中找到相关文档片段。
3. 系统将检索结果和用户问题组合为 Prompt。
4. 大语言模型根据上下文生成回答，支持流式和非流式两种调用方式。

## 项目结构

```text
Multi-RAG Chat/
├── .env                     # 环境变量（API Key、模型名等）
├── app.py                   # Streamlit 聊天界面入口
├── build_kb.py              # 命令行知识库构建脚本
├── test_rag.py              # 命令行测试脚本
├── requirements.txt         # 项目依赖
├── data/                    # 存放待构建知识库的文档和图片
├── chroma_db/               # Chroma 向量数据库持久化目录
└── src/
    ├── __init__.py          # 包初始化，暴露核心接口
    ├── document_loader.py   # 文档与图片加载处理
    ├── qwen_api.py          # Qwen / DashScope API 调用封装
    ├── vectorstore.py       # 向量库创建与加载
    └── rag_chain.py         # RAG 问答链
```

## 环境依赖

建议使用 Python 3.10 或更高版本。

安装 Python 依赖：

```bash
pip install -r requirements.txt
```

### 配置 API Key

本项目默认使用阿里云百炼 DashScope 的 Qwen 多模态 API。在项目根目录的 `.env` 文件中配置：

```env
# 阿里云百炼 API Key
DASHSCOPE_API_KEY=你的API_KEY
# Qwen 兼容 OpenAI 接口的地址
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
# 默认使用的 Qwen 模型
QWEN_VL_MODEL=qwen3-vl-plus
```

也可以直接设置系统环境变量：

```bash
# Windows CMD
set DASHSCOPE_API_KEY=你的API_KEY

# PowerShell
$env:DASHSCOPE_API_KEY="你的API_KEY"
```

### 切回本地 Ollama（可选）

代码中 `src/qwen_api.py` 和 `src/rag_chain.py` 保留了原来的 Ollama 本地调用写法，如需切回本地模型，按文件中的注释说明恢复即可。使用本地模式需要提前安装 Ollama 并拉取模型：

```bash
ollama pull qwen2.5:7b       # 文本问答
ollama pull qwen2.5vl:7b     # 图片描述
```

## 使用方法

### 1. 准备数据

可以在 Streamlit 页面左侧直接上传资料，也可以手动将需要构建知识库的文件放入 `data/` 目录，例如：

```text
data/
├── lecture.pdf
├── notes.txt
├── report.docx
└── image.png
```

### 2. 构建知识库

方式一：在 Streamlit 页面左侧点击"重建知识库"。

方式二：在命令行运行：

```bash
python build_kb.py
```

可选参数：

- `--skip-images`：跳过图片处理，只构建 PDF/TXT/DOCX，加快建库速度。
- `--device cpu`：使用 CPU 做 embedding（默认为 cuda）。

程序会自动读取 `data/` 目录中的文件，完成文档解析、图片描述生成、文本切分、向量化，并将结果保存到 `chroma_db/` 目录。

### 3. 启动聊天系统

运行：

```bash
streamlit run app.py
```

启动后浏览器会打开 Streamlit 页面。左侧侧边栏提供资料上传、知识库管理、问答参数调节、对话导出等功能；主区域为聊天界面，用户输入问题后系统会基于知识库生成回答并展示检索来源。

### 4. 命令行测试

也可以使用命令行方式快速验证 RAG 系统是否正常工作：

```bash
python test_rag.py
```

## 问答参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| Chunk size | 700 | 文本切分块大小，越大单块信息越完整但检索粒度越粗 |
| Chunk overlap | 100 | 相邻块重叠字符数，用于保持跨块上下文连贯 |
| top_k | 6 | 每次检索返回的文档片段数 |
| Temperature | 0.7 | 生成随机性，0 为确定性输出，1 为最发散 |
| Embedding 设备 | cuda | CPU/GPU 切换，无 NVIDIA GPU 时选 cpu |

## 示例问题

可以根据放入 `data/` 目录的材料提问，例如：

- 这份文档主要讲了什么？
- 请总结图片中的主要内容。
- 根据资料，某个概念的定义是什么？
- 文档中提到了哪些关键方法？
- 请根据知识库回答这个问题，并尽量引用相关内容。

## 课程作业说明

本项目可以作为自然语言处理课程期末大作业，重点体现以下内容：

- 文本预处理与文档解析。
- 文本切分与上下文建模。
- 语义表示学习与 embedding 模型应用。
- 基于向量数据库的语义检索。
- 检索增强生成问答系统设计。
- 多模态信息向文本知识库的转换。
- 大语言模型 API 调用与交互系统实现。

## 当前局限

- 图片内容依赖多模态模型描述，描述质量会影响最终问答效果。
- 当前主要使用向量检索，尚未加入 BM25、reranker 等增强检索方法。
- 缺少系统化的自动评测指标，回答质量主要通过人工观察。

## 可改进方向

- 增加 reranker，提高检索结果排序质量。
- 增加 BM25 与向量检索结合的混合检索。
- 增加知识库管理页面，支持删除指定文档、增量更新而非全量重建。
- 设计评测集，对不同 `top_k`、chunk size 和 embedding 模型进行对比实验。
- 支持更多 embedding 模型选择。

## 总结

Multi-RAG Chat 将文档解析、语义向量检索、检索增强生成和大语言模型问答结合起来，形成了一个完整的自然语言处理应用系统。它既能展示 NLP 中的文本表示和语义检索技术，也能体现大语言模型在实际知识库问答场景中的应用价值。
