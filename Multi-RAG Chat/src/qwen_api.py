"""Qwen / DashScope API 调用工具。

阿里云百炼的 Qwen 模型提供 OpenAI 兼容接口，因此可以直接使用
openai Python 客户端调用。默认使用 Qwen 多模态模型 qwen3-vl-plus。
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_MODEL = "qwen3-vl-plus"


def get_qwen_client() -> OpenAI:
    """创建 Qwen API 客户端。

    从环境变量 DASHSCOPE_API_KEY 读取密钥，base_url 使用阿里云百炼的
    OpenAI 兼容地址。未配置密钥时抛出 RuntimeError。
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("未检测到 DASHSCOPE_API_KEY，请先配置阿里云百炼 API Key。")

    return OpenAI(
        api_key=api_key,
        base_url=os.getenv("DASHSCOPE_BASE_URL", DEFAULT_QWEN_BASE_URL),
    )


def get_qwen_model() -> str:
    """读取当前使用的 Qwen 模型名。

    优先从环境变量 QWEN_VL_MODEL 读取，未设置时回退到 qwen3-vl-plus。
    可通过 .env 文件或系统环境变量配置。
    """
    return os.getenv("QWEN_VL_MODEL", DEFAULT_QWEN_MODEL)


def qwen_chat(messages, temperature: float = 0.7, model: str | None = None) -> str:
    """非流式调用 Qwen Chat Completions，返回完整回答文本。

    Args:
        messages: OpenAI 格式的消息列表，支持多模态 content（文本 + image_url）。
        temperature: 生成随机性，0 为确定性输出，1 为最发散。
        model: 模型名，默认使用环境变量 QWEN_VL_MODEL 或 qwen3-vl-plus。
    """
    client = get_qwen_client()
    response = client.chat.completions.create(
        model=model or get_qwen_model(),
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def qwen_chat_stream(messages, temperature: float = 0.7, model: str | None = None):
    """流式调用 Qwen Chat Completions，逐段 yield 文本。

    适用于 Streamlit 等需要逐步展示回答内容的场景。
    参数含义与 qwen_chat 相同。
    """
    client = get_qwen_client()
    response = client.chat.completions.create(
        model=model or get_qwen_model(),
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    for chunk in response:
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if content:
            yield content
