# call_ngram.py
from ngram_model import NGramModel

# ==================== 加载模型 ====================
model = NGramModel.load("ngram_model.pkl")

# ==================== 各种调用方式 ====================

print("=== N-Gram 模型调用演示 ===\n")

# 1. 预测下一个字（字符级）
print("1. 预测下一个字:")
print(model.predict_next("苹果", mode="char", top_k=5))

# 2. 预测下一个词（词级）
print("\n2. 预测下一个词:")
print(model.predict_next("机器学习", mode="word", top_k=5))

# 3. 自动续写（字符级）
print("\n3. 自动续写（字符）:")
print(model.generate("今天天气真好", length=20, mode="char"))

# 4. 自动续写（词级）
print("\n4. 自动续写（词级）:")
print(model.generate("自然语言处理", length=10, mode="word"))

# 5. 猜字游戏式调用
def guess_next():
    prefix = input("\n请输入前缀（猜下一个字）: ")
    result = model.predict_next(prefix, mode="char", top_k=3)
    print("模型建议:", result)

guess_next()  # 取消注释以启用交互式猜字游戏