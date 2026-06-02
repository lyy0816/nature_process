import os
import pickle
import numpy as np
from collections import Counter
import jieba   # 新增

RESULT_DIR = os.path.join(os.path.dirname(__file__), "result")
os.makedirs(RESULT_DIR, exist_ok=True)

def load_npy(path):
    data = np.load(path, allow_pickle=True)
    print(f"加载 {os.path.basename(path)}: {data.shape}")
    return data

def build_vocab(X_train, min_freq=2):
    """构建词表"""
    word_freq = Counter()
    for doc in X_train:
        if isinstance(doc, str):
            # 如果是原始字符串，先用 jieba 分词
            words = jieba.lcut(doc)
        elif isinstance(doc, (list, np.ndarray)):
            # 如果已经是分词后的列表
            words = [str(t) for t in doc]
        else:
            words = [str(doc)]
        
        word_freq.update(words)
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in word_freq.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    
    with open(os.path.join(RESULT_DIR, 'vocab.pkl'), 'wb') as f:
        pickle.dump(vocab, f)
    
    print(f"词表构建完成，大小: {len(vocab)}")
    return vocab

def load_vocab():
    with open(os.path.join(RESULT_DIR, 'vocab.pkl'), 'rb') as f:
        return pickle.load(f)