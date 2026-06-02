import jieba
from collections import defaultdict, Counter
import random
import pickle
import os
from typing import List, Tuple, Optional


class NGramModel:
    def __init__(self, n: int = 3):
        self.n = n
        self.ngram_counts = defaultdict(Counter)   # (context_tuple) -> {next: count}
        self.vocab = set()
        self.is_trained = False

    def train(self, texts: List[str], use_word: bool = True, use_char: bool = True):
        """
        训练 N-Gram 模型
        texts: 训练文本列表
        use_word: 是否训练词级 N-Gram
        use_char: 是否训练字符级 N-Gram
        """
        print(f"开始训练 {self.n}-gram 模型...")

        for text in texts:
            text = text.strip()
            if not text:
                continue

            # 字符级 N-Gram（适合猜字、补全）
            if use_char:
                chars = list(text)
                self.vocab.update(chars)
                for i in range(len(chars) - self.n + 1):
                    context = tuple(chars[i:i + self.n - 1])
                    next_item = chars[i + self.n - 1]
                    self.ngram_counts[context][next_item] += 1

            # 词级 N-Gram（适合猜词、续写）
            if use_word:
                words = list(jieba.cut(text, cut_all=False))
                self.vocab.update(words)
                for i in range(len(words) - self.n + 1):
                    context = tuple(words[i:i + self.n - 1])
                    next_item = words[i + self.n - 1]
                    self.ngram_counts[context][next_item] += 1

        self.is_trained = True
        print(f"训练完成！词汇量: {len(self.vocab)}，N-gram 条目: {len(self.ngram_counts)}")

    def predict_next(self, context: str, mode: str = "char", top_k: int = 5) -> List[Tuple[str, float]]:
        """
        预测下一个字符或词
        mode: 'char' 或 'word'
        返回: [(候选, 概率), ...] top_k 个
        """
        if not self.is_trained:
            raise Exception("模型尚未训练，请先调用 train()")

        if mode == "char":
            tokens = list(context)
        else:  # word
            tokens = list(jieba.cut(context))

        # 取最后 n-1 个作为 context
        context_tokens = tokens[-(self.n - 1):]
        context_tuple = tuple(context_tokens)

        if context_tuple in self.ngram_counts:
            counter = self.ngram_counts[context_tuple]
            total = sum(counter.values())
            candidates = [(item, count / total) for item, count in counter.most_common(top_k)]
            return candidates
        else:
            # 回退策略：随机返回高频词
            fallback = [(item, 0.1) for item in random.sample(list(self.vocab), min(top_k, len(self.vocab)))]
            return fallback

    def generate(self, prefix: str, length: int = 20, mode: str = "char") -> str:
        """根据前缀自动续写"""
        result = prefix
        for _ in range(length):
            candidates = self.predict_next(result, mode=mode, top_k=3)
            if candidates:
                next_item = random.choices([c[0] for c in candidates], 
                                         weights=[c[1] for c in candidates])[0]
                result += next_item if mode == "char" else " " + next_item
        return result.strip()

    def save(self, filepath: str):
        """保存模型"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"模型已保存: {filepath}")

    @staticmethod
    def load(filepath: str) -> 'NGramModel':
        """加载模型"""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"模型已加载: {filepath}")
        return model


if __name__ == "__main__":
    # ------------------- 1. 准备数据 -------------------
    dataset = [
        "今天天气真好适合出去玩",
        "苹果香蕉橙子芒果都是水果",
        "机器学习和深度学习是人工智能的重要分支",
        "我喜欢吃芒果和桃子",
        "猫和狗是人类常见宠物",
        "长江黄河是中国的母亲河",
        "蓝色天空白云飘飘",
        "红色代表热情紫色代表神秘",
        "电脑手机键盘屏幕都是电子产品",
        "森林高山海洋都有独特的风景",
        "自然语言处理是人工智能的一个重要方向",
    ]

    # ------------------- 2. 训练模型 -------------------
    model = NGramModel(n=3)
    model.train(dataset, use_word=True, use_char=True)

    # ------------------- 3. 保存模型 -------------------
    model.save("ngram_model.pkl")
