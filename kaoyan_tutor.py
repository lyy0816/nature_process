import requests
import json
import os

# 配置部分
OLLAMA_API_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:7b"  # 你使用的模型
PROMPT_FILE = "prompt.txt"

def load_system_prompt():
    """读取本地提示词文件"""
    if not os.path.exists(PROMPT_FILE):
        print(f"错误：未找到 {PROMPT_FILE} 文件，请确保提示词已保存。")
        return None
    
    with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
        return f.read()

def chat_with_tutor(user_input, history):
    """
    发送请求到 Ollama
    :param user_input: 用户当前输入
    :param history: 对话历史列表
    :return: 模型回复内容
    """
    # 构建消息列表
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}, # 注入你的考研辅导人设
        *history,                                     # 历史对话
        {"role": "user", "content": user_input}       # 当前问题
    ]

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,  # 为了简化代码，先关闭流式输出，如需打字机效果可改为 True
        "options": {
            "temperature": 0.7,  # 控制创造性，0.7 适合教学
            "top_p": 0.9
        }
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        return result['message']['content']
    except requests.exceptions.ConnectionError:
        return " 连接失败：请确保 Ollama 服务已启动 (运行 'ollama serve')。"
    except Exception as e:
        return f" 发生错误：{str(e)}"

def main():
    global SYSTEM_PROMPT
    print(" 正在加载考研英语 AI 辅导助手...")
    
    SYSTEM_PROMPT = load_system_prompt()
    if not SYSTEM_PROMPT:
        return

    print(" 加载成功！我是你的考研英语专属辅导员。")
    print(" 提示：输入 'quit' 或 'exit' 退出程序。")
    print("-" * 50)

    # 初始化对话历史
    conversation_history = []

    while True:
        try:
            user_input = input("\n 考生提问：").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print(" 祝你考研顺利，金榜题名！再见。")
                break
            
            if not user_input:
                continue

            print("\n 辅导员正在思考", end="", flush=True)
            
            # 获取回复
            response_text = chat_with_tutor(user_input, conversation_history)
            
            # 清除加载提示并打印结果
            print("\r" + " " * 40 + "\r") # 清除上一行的加载文字
            print(response_text)
            print("-" * 50)

            # 更新历史记录 (保留最近 10 轮对话以节省显存，qwen2.5:7b 上下文有限)
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": response_text})
            
            # 简单的上下文窗口管理，防止显存溢出
            if len(conversation_history) > 20: 
                conversation_history = conversation_history[-20:]

        except KeyboardInterrupt:
            print("\n\n 检测到中断，已退出。祝备考顺利！")
            break

if __name__ == "__main__":
    main()