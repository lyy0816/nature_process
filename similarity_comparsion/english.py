import numpy as np
import matplotlib.pyplot as plt
from gensim.models.keyedvectors import KeyedVectors
import os
from datetime import datetime

# ====================== 创建结果文件夹 ======================
result_dir = "result"
os.makedirs(result_dir, exist_ok=True)  # 如果文件夹不存在则创建
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成时间戳，用于区分不同运行的结果

# ====================== 配置 ======================
MODEL_PATH = "wiki-news-300d-1M-subword.vec"  # 预训练英文 fastText 模型路径（300维，百万词向量）

print("正在加载英文词向量模型（wiki-news-300d-1M-subword.vec）...")
# load_word2vec_format: 加载 Google Word2Vec 格式的预训练词向量文件
# binary=False: 文本格式（非二进制）
# unicode_errors='ignore': 忽略无法解码的字符，避免因罕见 Unicode 字符导致加载失败
model = KeyedVectors.load_word2vec_format(MODEL_PATH, binary=False, unicode_errors='ignore')
print(f"模型加载成功！维度: {model.vector_size}\n")

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

# ====================== 文本转向量 ======================
def get_word_vector(word, model):
    """
    获取单词的词向量
    :param word: 英文单词
    :param model: 预训练的词向量模型
    :return: 单词对应的向量，如果单词不在模型中则返回全零向量
    """
    return model[word] if word in model else np.zeros(model.vector_size)

def get_sentence_vector(sentence, model):
    """
    获取句子的向量表示（词袋均值法）
    原理：将句子分词后，取所有单词向量的平均值作为句子向量
    注意：英文无需专门分词库，直接按空格 split 即可（已用 subword 模型处理未登录词）
    :param sentence: 英文句子
    :param model: 预训练的词向量模型
    :return: 句子向量（各词向量的均值）
    """
    words_in_sentence = sentence.lower().split()  # 转小写后按空格分词，减少大小写不匹配问题
    # 过滤掉空字符串，并只保留在模型中的词（subword 模型可处理大部分未登录词）
    vecs = [model[w] for w in words_in_sentence if w in model]
    if vecs:
        return np.mean(vecs, axis=0)  # 对所有词向量取平均值
    return np.zeros(model.vector_size)  # 如果没有有效词向量，返回零向量

# 将所有单词转换为向量（每个单词直接查表获取）
word_vectors = np.array([get_word_vector(w, model) for w in words])

# 将所有句子转换为向量（每个句子由其分词后的词向量均值表示）
sentence_vectors = np.array([get_sentence_vector(s, model) for s in sentences])

def cosine_sim(v1, v2):
    """
    计算两个向量的余弦相似度
    余弦相似度 = (A·B) / (|A|×|B|)，取值范围 [-1, 1]，值越大表示越相似
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)

# ====================== 保存文本结果 ======================
result_txt = os.path.join(result_dir, f"english_similarity_{timestamp}.txt")

with open(result_txt, "w", encoding="utf-8") as f:
    f.write("=== 英文词向量相似度比对实验结果 ===\n")
    f.write(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"模型: {MODEL_PATH}\n")
    f.write(f"向量维度: {model.vector_size}\n\n")

    # ========== 1. 单词部分 ==========
    f.write("=" * 50 + "\n")
    f.write("【第一部分：单词相似度对比】\n")
    f.write("=" * 50 + "\n\n")

    f.write("1.1 单词列表:\n")
    for i, w in enumerate(words):
        f.write(f"   {i+1:2d}. {w}\n")

    f.write("\n1.2 单词向量示例（前10维）:\n")
    for i, w in enumerate(words):
        f.write(f"   {w}: {word_vectors[i][:10].tolist()}\n")

    f.write("\n1.3 单词对相似度结果:\n")
    # 遍历所有不重复的单词对组合
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            cos = cosine_sim(word_vectors[i], word_vectors[j])  # 余弦相似度
            euc = np.linalg.norm(word_vectors[i] - word_vectors[j])  # 欧氏距离
            f.write(f"   {words[i]}  vs  {words[j]}\n")
            f.write(f"      余弦相似度 = {cos:.4f}    欧氏距离 = {euc:.4f}\n\n")

    # ========== 2. 句子部分 ==========
    f.write("\n" + "=" * 50 + "\n")
    f.write("【第二部分：句子相似度对比】\n")
    f.write("=" * 50 + "\n\n")

    f.write("2.1 句子列表:\n")
    for i, s in enumerate(sentences):
        f.write(f"   {i+1:2d}. {s}\n")

    f.write("\n2.2 句子向量示例（前10维）:\n")
    for i, s in enumerate(sentences):
        f.write(f"   文本: {s}\n")
        f.write(f"   前10维: {sentence_vectors[i][:10].tolist()}\n\n")

    # ========== 2.3 句子维度对比（英文版特有）==========
    # 探究不同向量维度对相似度计算的影响
    # 取句子1 ("I love eating apple") 和句子2 ("I love eating banana") 进行对比
    f.write("\n2.3 不同维度相似度对比（句子1 vs 句子2）:\n")
    dims = [50, 100, 200, 300]  # 逐步增加维度，观察相似度变化
    for d in dims:
        vecs_d = sentence_vectors[:, :d]  # 取前 d 维（模拟不同维度下的表示能力）
        sim = np.dot(vecs_d[0], vecs_d[1]) / (np.linalg.norm(vecs_d[0]) * np.linalg.norm(vecs_d[1]) + 1e-9)
        f.write(f"   {d:3d}维 → 余弦相似度 = {sim:.4f}\n")

    f.write("\n2.4 句子对相似度结果:\n")
    # 遍历所有不重复的句子对组合
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            cos = cosine_sim(sentence_vectors[i], sentence_vectors[j])  # 余弦相似度
            euc = np.linalg.norm(sentence_vectors[i] - sentence_vectors[j])  # 欧氏距离
            f.write(f"   {sentences[i]}  vs  {sentences[j]}\n")
            f.write(f"      余弦相似度 = {cos:.4f}    欧氏距离 = {euc:.4f}\n\n")

print(f"文本结果已保存到: {result_txt}")

# ====================== 保存图形 ======================
def cosine_similarity(v1, v2):
    """
    计算两个向量的余弦相似度（与 cosine_sim 功能相同）
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)

def pca_2d(data):
    """
    使用 SVD（奇异值分解）将高维数据降维到 2 维
    原理：计算数据中心化后的协方差矩阵，取最大的两个奇异值对应的向量作为主成分
    :param data: 高维数据矩阵 (n_samples, n_features)
    :return: 降维后的 2 维数据 (n_samples, 2)
    """
    mean = np.mean(data, axis=0)  # 计算每个维度的均值
    centered = data - mean  # 中心化：减去均值
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)  # SVD 分解
    return centered @ Vt.T[:, :2]  # 取前两个主成分方向投影

SIM_THRESHOLD = 0.6  # 相似度阈值：只有 >= 0.6 的词对才在 PCA 图中显示连线

# ====================== 单词部分图形 ======================
print("\n正在生成单词部分图形...")

# 构建单词相似度矩阵（每个元素为两个单词的余弦相似度）
word_sim_matrix = np.array([
    [cosine_similarity(word_vectors[i], word_vectors[j]) for j in range(len(words))]
    for i in range(len(words))
])

# 绘制单词相似度热力图
plt.figure(figsize=(12, 10))
plt.imshow(word_sim_matrix, cmap='YlOrRd', interpolation='nearest')  # YlOrRd：黄-橙-红配色
plt.colorbar(label='Cosine Similarity')  # 添加颜色条标注数值
plt.xticks(range(len(words)), words, rotation=45, ha='right')  # x轴标签（英文无需设置 fontsize）
plt.yticks(range(len(words)), words)  # y轴标签
plt.title('English Words - Similarity Heatmap', fontsize=14)
plt.tight_layout()  # 自动调整子图参数，避免标签被截断
word_heatmap_path = os.path.join(result_dir, f"english_words_heatmap_{timestamp}.png")
plt.savefig(word_heatmap_path, dpi=300, bbox_inches='tight')  # dpi=300 高清保存
plt.close()
print(f"单词热力图已保存: {word_heatmap_path}")

# 绘制单词 PCA 二维散点图
word_vectors_2d = pca_2d(word_vectors)  # 将单词向量降维到 2 维

plt.figure(figsize=(12, 10))
plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1], s=200, c='crimson', alpha=0.8, zorder=3)

# 在相似度 >= SIM_THRESHOLD 的词对之间画连线
for i in range(len(words)):
    for j in range(i + 1, len(words)):
        cos = cosine_similarity(word_vectors[i], word_vectors[j])
        if cos >= SIM_THRESHOLD:
            # 线宽和透明度随相似度增加而增加
            linewidth = 1 + (cos - SIM_THRESHOLD) / (1 - SIM_THRESHOLD) * 4
            alpha = 0.4 + (cos - SIM_THRESHOLD) / (1 - SIM_THRESHOLD) * 0.5
            plt.plot([word_vectors_2d[i, 0], word_vectors_2d[j, 0]],
                     [word_vectors_2d[i, 1], word_vectors_2d[j, 1]],
                     'r-', linewidth=linewidth, alpha=alpha, zorder=1)
            # 在连线中点标注相似度数值
            mid_x = (word_vectors_2d[i, 0] + word_vectors_2d[j, 0]) / 2
            mid_y = (word_vectors_2d[i, 1] + word_vectors_2d[j, 1]) / 2
            plt.annotate(f'{cos:.2f}', xy=(mid_x, mid_y), fontsize=8, color='darkred', alpha=0.8)

# 为每个数据点添加英文标签
for i, w in enumerate(words):
    plt.annotate(w, (word_vectors_2d[i, 0] + 0.02, word_vectors_2d[i, 1] + 0.02), fontsize=12, fontweight='bold')
plt.title(f'English Words - PCA 2D (Similarity≥{SIM_THRESHOLD} linked)', fontsize=14)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True, alpha=0.3)
word_pca_path = os.path.join(result_dir, f"english_words_pca_{timestamp}.png")
plt.savefig(word_pca_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"单词PCA图已保存: {word_pca_path}")

# ====================== 句子部分图形 ======================
print("正在生成句子部分图形...")

# 构建句子相似度矩阵
sentence_sim_matrix = np.array([
    [cosine_similarity(sentence_vectors[i], sentence_vectors[j]) for j in range(len(sentences))]
    for i in range(len(sentences))
])

# 绘制句子相似度热力图
plt.figure(figsize=(14, 12))
plt.imshow(sentence_sim_matrix, cmap='YlGnBu', interpolation='nearest')  # YlGnBu：黄-绿-蓝配色
plt.colorbar(label='Cosine Similarity')
plt.xticks(range(len(sentences)), sentences, rotation=45, ha='right')
plt.yticks(range(len(sentences)), sentences)
plt.title('English Sentences - Similarity Heatmap', fontsize=14)
plt.tight_layout()
sentence_heatmap_path = os.path.join(result_dir, f"english_sentences_heatmap_{timestamp}.png")
plt.savefig(sentence_heatmap_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"句子热力图已保存: {sentence_heatmap_path}")

# 绘制句子 PCA 二维散点图
sentence_vectors_2d = pca_2d(sentence_vectors)  # 将句子向量降维到 2 维

plt.figure(figsize=(14, 11))
plt.scatter(sentence_vectors_2d[:, 0], sentence_vectors_2d[:, 1], s=200, c='steelblue', alpha=0.8, zorder=3)

# 在相似度 >= SIM_THRESHOLD 的句子对之间画连线
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        cos = cosine_similarity(sentence_vectors[i], sentence_vectors[j])
        if cos >= SIM_THRESHOLD:
            linewidth = 1 + (cos - SIM_THRESHOLD) / (1 - SIM_THRESHOLD) * 4
            alpha = 0.4 + (cos - SIM_THRESHOLD) / (1 - SIM_THRESHOLD) * 0.5
            plt.plot([sentence_vectors_2d[i, 0], sentence_vectors_2d[j, 0]],
                     [sentence_vectors_2d[i, 1], sentence_vectors_2d[j, 1]],
                     'b-', linewidth=linewidth, alpha=alpha, zorder=1)
            mid_x = (sentence_vectors_2d[i, 0] + sentence_vectors_2d[j, 0]) / 2
            mid_y = (sentence_vectors_2d[i, 1] + sentence_vectors_2d[j, 1]) / 2
            plt.annotate(f'{cos:.2f}', xy=(mid_x, mid_y), fontsize=8, color='darkblue', alpha=0.8)

# 为每个数据点添加句子标签
for i, s in enumerate(sentences):
    plt.annotate(s, (sentence_vectors_2d[i, 0] + 0.02, sentence_vectors_2d[i, 1] + 0.02), fontsize=10)
plt.title(f'English Sentences - PCA 2D (Similarity≥{SIM_THRESHOLD} linked)', fontsize=14)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True, alpha=0.3)
sentence_pca_path = os.path.join(result_dir, f"english_sentences_pca_{timestamp}.png")
plt.savefig(sentence_pca_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"句子PCA图已保存: {sentence_pca_path}")

print(f"\n✅ 英文实验全部完成！所有结果已保存至 {result_dir} 文件夹")
