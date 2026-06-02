import numpy as np
import matplotlib.pyplot as plt
import jieba
import os
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ====================== 中文字体设置（解决方块问题） ======================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ====================== 创建结果文件夹 ======================
result_dir = "result_local"
os.makedirs(result_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ====================== 数据集 ======================
# 单词分类：水果、动物、科技、颜色、自然
words = [
    # 水果
    "苹果", "香蕉", "橙子", "葡萄", "芒果", "桃子",
    # 动物
    "狗", "猫", "狮子", "老虎", "马", "鸟",
    # 科技
    "电脑", "手机", "键盘", "屏幕", "相机",
    # 颜色
    "红色", "蓝色", "绿色", "黄色", "紫色",
    # 自然
    "河流", "高山", "森林", "海洋", "云朵"
]

# 句子分类：食物、科技、天气、动物、颜色、自然
sentences = [
    # 食物相关
    "我喜欢吃苹果",
    "我喜欢吃香蕉",
    "草莓酸酸甜甜很好吃",
    # 科技相关
    "电脑速度很快",
    "笔记本电脑很轻便",
    "手机屏幕真清晰",
    "键盘打字手感很好",
    # 天气相关
    "今天天气非常晴朗",
    "今天气温很高很热",
    "天空中飘着白云",
    # 动物相关
    "狗狗非常友好",
    "猫咪喜欢睡觉",
    "鸟儿歌声很动听",
    # 颜色相关
    "天空是蓝色的",
    "树叶变成绿色",
    # 自然相关
    "河流静静流淌",
    "高山非常高耸"
]

# ====================== 【已注释】原始 fastText 预训练模型代码 ======================
# 以下为原来的预训练词向量方法，现已替换为局部文本表示方法（One-hot / BoW / TF-IDF / n-gram）
#
# from gensim.models.fasttext import load_facebook_vectors
# MODEL_PATH = "cc.zh.300.bin"
# print("正在加载中文 fastText 模型（cc.zh.300.bin）... 请耐心等待...")
# model = load_facebook_vectors(MODEL_PATH)
# print(f"模型加载成功！向量维度: {model.vector_size}\n")
# jieba.initialize()
#
# def get_word_vector(word, model):
#     return model[word] if word in model else np.zeros(model.vector_size)
#
# def get_sentence_vector(sentence, model):
#     words_in_sentence = jieba.lcut(sentence)
#     vecs = [model[w] for w in words_in_sentence if w.strip() and w in model]
#     if vecs:
#         return np.mean(vecs, axis=0)
#     return np.zeros(model.vector_size)
#
# def cosine_sim(v1, v2):
#     return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
#
# word_vectors = np.array([get_word_vector(w, model) for w in words])
# sent_vectors = np.array([get_sentence_vector(s, model) for s in sentences])

# ====================== 预处理：中文分词 ======================
def tokenize_chinese(text):
    """对中文文本进行 jieba 分词，用空格连接，供 sklearn 向量器使用"""
    return ' '.join(jieba.lcut(text))

# 预分词后的文本（用于 OneHot / BoW / TF-IDF）
words_tokenized = [tokenize_chinese(w) for w in words]
sentences_tokenized = [tokenize_chinese(s) for s in sentences]

print("数据集加载完成！")
print(f"  单词数量: {len(words)}")
print(f"  句子数量: {len(sentences)}")

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
    :param texts: 待向量化的文本列表（已分词或原始文本）
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
    cn_type = {"words": "单词", "sentences": "句子"}.get(data_type, data_type)
    f_out.write(f"\n{'='*50}\n")
    f_out.write(f"【{method_name} - {cn_type}】\n")
    f_out.write(f"{'='*50}\n")
    f_out.write(f"向量维度: {vecs.shape[1]}\n\n")

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            cos = sim_matrix[i][j]
            euc = np.linalg.norm(vecs[i] - vecs[j])
            f_out.write(f"   {labels[i]}  vs  {labels[j]}\n")
            f_out.write(f"      余弦相似度 = {cos:.4f}    欧氏距离 = {euc:.4f}\n\n")

    # 4. 热力图
    plt.figure(figsize=(12, 10))
    plt.imshow(sim_matrix, cmap='YlOrRd', interpolation='nearest')
    plt.colorbar(label='余弦相似度')
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right', fontsize=10)
    plt.yticks(range(len(labels)), labels, fontsize=10)
    plt.title(f'{method_name} - {cn_type} 余弦相似度热力图', fontsize=14)
    plt.tight_layout()
    heatmap_path = os.path.join(result_dir, f"chinese_{method_name}_{data_type}_heatmap_{timestamp}.png")
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
                     xytext=(vecs_2d[i, 0] + 0.03, vecs_2d[i, 1] + 0.03),
                     fontsize=10, fontweight='bold')
    plt.title(f'{method_name} - {cn_type} PCA 2D（相似度≥{sim_threshold}显示连线）', fontsize=14)
    plt.xlabel('主成分1')
    plt.ylabel('主成分2')
    plt.grid(True, alpha=0.3)
    pca_path = os.path.join(result_dir, f"chinese_{method_name}_{data_type}_pca_{timestamp}.png")
    plt.savefig(pca_path, dpi=300, bbox_inches='tight')
    plt.close()

    return vecs, sim_matrix


# ====================== 主实验流程 ======================
result_txt = os.path.join(result_dir, f"chinese_local_similarity_{timestamp}.txt")

with open(result_txt, "w", encoding="utf-8") as f:
    f.write("=== 中文文本局部相似度比对实验 ===\n")
    f.write(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("方法: One-hot Encoding, BoW, TF-IDF, n-gram\n")
    f.write("相似度度量: 余弦相似度 + 欧氏距离\n\n")

    # ============================================================
    # 方法1: One-hot Encoding（binary BoW）
    # ============================================================
    print("\n" + "=" * 50)
    print("[1/4] One-hot Encoding 方法...")
    print("=" * 50)
    onehot = CountVectorizer(binary=True)
    run_comparison(onehot, "OneHot", words_tokenized, words, "words", f)
    run_comparison(onehot, "OneHot", sentences_tokenized, sentences, "sentences", f)
    print("  [OneHot] 完成！")

    # ============================================================
    # 方法2: BoW（词袋模型）
    # ============================================================
    print("\n" + "=" * 50)
    print("[2/4] BoW（词袋模型）方法...")
    print("=" * 50)
    bow = CountVectorizer()
    run_comparison(bow, "BoW", words_tokenized, words, "words", f)
    run_comparison(bow, "BoW", sentences_tokenized, sentences, "sentences", f)
    print("  [BoW] 完成！")

    # ============================================================
    # 方法3: TF-IDF
    # ============================================================
    print("\n" + "=" * 50)
    print("[3/4] TF-IDF 方法...")
    print("=" * 50)
    tfidf = TfidfVectorizer()
    run_comparison(tfidf, "TFIDF", words_tokenized, words, "words", f)
    run_comparison(tfidf, "TFIDF", sentences_tokenized, sentences, "sentences", f)
    print("  [TFIDF] 完成！")

    # ============================================================
    # 方法4: n-gram（字符级 n-gram，使用原始文本）
    # ============================================================
    print("\n" + "=" * 50)
    print("[4/4] n-gram（字符级 n-gram）方法...")
    print("=" * 50)
    ngram = CountVectorizer(analyzer='char', ngram_range=(2, 4))
    run_comparison(ngram, "Ngram", words, words, "words", f)
    run_comparison(ngram, "Ngram", sentences, sentences, "sentences", f)
    print("  [Ngram] 完成！")

print(f"\n{'='*50}")
print(f"文本结果已保存到: {result_txt}")
print(f"\n全部实验完成！所有结果保存在 {result_dir} 文件夹")
