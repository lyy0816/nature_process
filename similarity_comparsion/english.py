import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ====================== 创建结果文件夹 ======================
result_dir = "result_local"
os.makedirs(result_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ====================== 数据集 ======================
# 单词分类：水果、动物、科技、颜色、自然（与中文版对照）
words = [
    # 水果
    "apple", "banana", "orange", "grape", "mango", "peach",
    # 动物
    "dog", "cat", "lion", "tiger", "horse", "bird",
    # 科技
    "computer", "phone", "keyboard", "screen", "camera",
    # 颜色
    "red", "blue", "green", "yellow", "purple",
    # 自然
    "river", "mountain", "forest", "ocean", "cloud"
]

# 句子分类：食物、科技、天气、动物、颜色、自然（与中文版对照）
sentences = [
    # 食物相关
    "I love eating apple",
    "I love eating banana",
    "Strawberries are sweet",
    # 科技相关
    "The computer is very fast",
    "The laptop is portable",
    "The phone screen is clear",
    "The keyboard feels great",
    # 天气相关
    "Today is a sunny day",
    "The weather is very hot",
    "The sky is cloudy today",
    # 动物相关
    "The dog is very friendly",
    "The cat likes to sleep",
    "The bird sings beautifully",
    # 颜色相关
    "The sky is blue today",
    "The leaves turn green",
    # 自然相关
    "The river flows quietly",
    "The mountain is very high"
]

# ====================== 【已注释】原始 fastText 预训练模型代码 ======================
# 以下为原来的预训练词向量方法，现已替换为局部文本表示方法（One-hot / BoW / TF-IDF / n-gram）
#
# from gensim.models.keyedvectors import KeyedVectors
# MODEL_PATH = "wiki-news-300d-1M-subword.vec"
# print("正在加载英文词向量模型（wiki-news-300d-1M-subword.vec）...")
# model = KeyedVectors.load_word2vec_format(MODEL_PATH, binary=False, unicode_errors='ignore')
# print(f"模型加载成功！维度: {model.vector_size}\n")
#
# def get_word_vector(word, model):
#     return model[word] if word in model else np.zeros(model.vector_size)
#
# def get_sentence_vector(sentence, model):
#     words_in_sentence = sentence.lower().split()
#     vecs = [model[w] for w in words_in_sentence if w in model]
#     if vecs:
#         return np.mean(vecs, axis=0)
#     return np.zeros(model.vector_size)
#
# def cosine_sim(v1, v2):
#     return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
#
# word_vectors = np.array([get_word_vector(w, model) for w in words])
# sentence_vectors = np.array([get_sentence_vector(s, model) for s in sentences])

print("数据集加载完成！")
print(f"  Words count: {len(words)}")
print(f"  Sentences count: {len(sentences)}")

# ====================== 工具函数 ======================
def pca_2d(data):
    """使用 SVD 将高维数据降维到 2 维"""
    mean = np.mean(data, axis=0)
    centered = data - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ Vt.T[:, :2]

def run_comparison(vectorizer, method_name, texts, labels, data_type, f_out, sim_threshold=0.3):
    """
    运行一种文本表示方法的完整相似度比对实验
    :param vectorizer: sklearn 向量器
    :param method_name: 方法名称
    :param texts: 待向量化的文本列表
    :param labels: 原始文本标签
    :param data_type: "单词" 或 "句子"
    :param f_out: 输出文件句柄
    :param sim_threshold: PCA 连线相似度阈值
    """
    # 1. 向量化
    vecs = vectorizer.fit_transform(texts).toarray()

    # 2. 计算余弦相似度矩阵
    sim_matrix = cosine_similarity(vecs)

    # 3. 写入文本结果
    en_type = {"words": "Words", "sentences": "Sentences"}.get(data_type, data_type)
    f_out.write(f"\n{'='*50}\n")
    f_out.write(f"【{method_name} - {en_type}】\n")
    f_out.write(f"{'='*50}\n")
    f_out.write(f"Vector dimension: {vecs.shape[1]}\n\n")

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            cos = sim_matrix[i][j]
            euc = np.linalg.norm(vecs[i] - vecs[j])
            f_out.write(f"   {labels[i]}  vs  {labels[j]}\n")
            f_out.write(f"      Cosine Similarity = {cos:.4f}    Euclidean Distance = {euc:.4f}\n\n")

    # 4. 热力图
    plt.figure(figsize=(12, 10))
    plt.imshow(sim_matrix, cmap='YlOrRd', interpolation='nearest')
    plt.colorbar(label='Cosine Similarity')
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right', fontsize=10)
    plt.yticks(range(len(labels)), labels, fontsize=10)
    plt.title(f'{method_name} - {en_type} Similarity Heatmap', fontsize=14)
    plt.tight_layout()
    heatmap_path = os.path.join(result_dir, f"english_{method_name}_{data_type}_heatmap_{timestamp}.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 5. PCA 二维散点图
    vecs_2d = pca_2d(vecs)
    plt.figure(figsize=(12, 10))
    plt.scatter(vecs_2d[:, 0], vecs_2d[:, 1], s=200, c='steelblue', alpha=0.8, zorder=3)

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            cos = sim_matrix[i][j]
            if cos >= sim_threshold:
                linewidth = 1 + (cos - sim_threshold) / (1 - sim_threshold) * 4
                alpha = 0.4 + (cos - sim_threshold) / (1 - sim_threshold) * 0.5
                plt.plot([vecs_2d[i, 0], vecs_2d[j, 0]],
                         [vecs_2d[i, 1], vecs_2d[j, 1]],
                         'b-', linewidth=linewidth, alpha=alpha, zorder=1)

    for i, label in enumerate(labels):
        plt.annotate(label, xy=(vecs_2d[i, 0], vecs_2d[i, 1]),
                     xytext=(vecs_2d[i, 0] + 0.02, vecs_2d[i, 1] + 0.02),
                     fontsize=10, fontweight='bold')
    plt.title(f'{method_name} - {en_type} PCA 2D (Similarity >= {sim_threshold})', fontsize=14)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True, alpha=0.3)
    pca_path = os.path.join(result_dir, f"english_{method_name}_{data_type}_pca_{timestamp}.png")
    plt.savefig(pca_path, dpi=300, bbox_inches='tight')
    plt.close()

    return vecs, sim_matrix


# ====================== 主实验流程 ======================
result_txt = os.path.join(result_dir, f"english_local_similarity_{timestamp}.txt")

with open(result_txt, "w", encoding="utf-8") as f:
    f.write("=== English Text Local Similarity Comparison Experiment ===\n")
    f.write(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("Methods: One-hot Encoding, BoW, TF-IDF, n-gram\n")
    f.write("Similarity metrics: Cosine Similarity + Euclidean Distance\n\n")

    # ============================================================
    # 方法1: One-hot Encoding（binary BoW）
    # ============================================================
    print("\n" + "=" * 50)
    print("[1/4] One-hot Encoding method...")
    print("=" * 50)
    onehot = CountVectorizer(binary=True)
    run_comparison(onehot, "OneHot", words, words, "words", f)
    run_comparison(onehot, "OneHot", sentences, sentences, "sentences", f)
    print("  [OneHot] Done!")

    # ============================================================
    # 方法2: BoW（词袋模型）
    # ============================================================
    print("\n" + "=" * 50)
    print("[2/4] BoW (Bag of Words) method...")
    print("=" * 50)
    bow = CountVectorizer()
    run_comparison(bow, "BoW", words, words, "words", f)
    run_comparison(bow, "BoW", sentences, sentences, "sentences", f)
    print("  [BoW] Done!")

    # ============================================================
    # 方法3: TF-IDF
    # ============================================================
    print("\n" + "=" * 50)
    print("[3/4] TF-IDF method...")
    print("=" * 50)
    tfidf = TfidfVectorizer()
    run_comparison(tfidf, "TFIDF", words, words, "words", f)
    run_comparison(tfidf, "TFIDF", sentences, sentences, "sentences", f)
    print("  [TFIDF] Done!")

    # ============================================================
    # 方法4: n-gram（字符级 n-gram，使用原始文本）
    # ============================================================
    print("\n" + "=" * 50)
    print("[4/4] n-gram (character-level n-gram) method...")
    print("=" * 50)
    ngram = CountVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    run_comparison(ngram, "Ngram", words, words, "words", f)
    run_comparison(ngram, "Ngram", sentences, sentences, "sentences", f)
    print("  [Ngram] Done!")

print(f"\n{'='*50}")
print(f"Text results saved to: {result_txt}")
print(f"\nAll experiments completed! Results saved in {result_dir} folder")
