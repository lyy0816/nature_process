import numpy as np
import matplotlib.pyplot as plt
from gensim.models.fasttext import load_facebook_vectors
import jieba
import os
from datetime import datetime

# ====================== 中文字体设置（解决方块问题） ======================
# 设置支持中文的字体，依次尝试：黑体、微软雅黑、 Arial Unicode MS
# axes.unicode_minus=False 解决负号显示为方块的问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

# ====================== 创建结果文件夹 ======================
result_dir = "result"
os.makedirs(result_dir, exist_ok=True)  # 如果文件夹不存在则创建
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成时间戳，用于区分不同运行的结果

# ====================== 配置 ======================
MODEL_PATH = "cc.zh.300.bin"  # 预训练中文 fastText 模型路径（300维向量）

print("正在加载中文 fastText 模型（cc.zh.300.bin）... 请耐心等待...")
model = load_facebook_vectors(MODEL_PATH)  # 加载预训练模型
print(f"模型加载成功！向量维度: {model.vector_size}\n")

jieba.initialize()  # 初始化 jieba 分词器

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

# ====================== 文本转向量 ======================
def get_word_vector(word, model):
    """
    获取单词的词向量
    :param word: 中文单词
    :param model: 预训练的词向量模型
    :return: 单词对应的向量，如果单词不在模型中则返回全零向量
    """
    return model[word] if word in model else np.zeros(model.vector_size)

def get_sentence_vector(sentence, model):
    """
    获取句子的向量表示（词袋均值法）
    原理：将句子分词后，取所有单词向量的平均值作为句子向量
    :param sentence: 中文句子
    :param model: 预训练的词向量模型
    :return: 句子向量（各词向量的均值）
    """
    words_in_sentence = jieba.lcut(sentence)  # 使用 jieba 对句子进行分词
    # 过滤掉空字符串，并只保留在模型中的词
    vecs = [model[w] for w in words_in_sentence if w.strip() and w in model]
    if vecs:
        return np.mean(vecs, axis=0)  # 对所有词向量取平均值
    return np.zeros(model.vector_size)  # 如果没有有效词向量，返回零向量

# 将所有单词转换为向量（每个单词直接查表获取）
word_vectors = np.array([get_word_vector(w, model) for w in words])

# 将所有句子转换为向量（每个句子由其分词后的词向量均值表示）
sent_vectors = np.array([get_sentence_vector(s, model) for s in sentences])

def cosine_sim(v1, v2):
    """
    计算两个向量的余弦相似度
    余弦相似度 = (A·B) / (|A|×|B|)，取值范围 [-1, 1]，值越大表示越相似
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)

# ====================== 保存文本结果 ======================
result_txt = os.path.join(result_dir, f"chinese_similarity_{timestamp}.txt")

with open(result_txt, "w", encoding="utf-8") as f:
    f.write("=== 中文词向量相似度比对实验结果 ===\n")
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
        f.write(f"   前10维: {sent_vectors[i][:10].tolist()}\n\n")

    f.write("\n2.3 句子对相似度结果:\n")
    # 遍历所有不重复的句子对组合
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            cos = cosine_sim(sent_vectors[i], sent_vectors[j])  # 余弦相似度
            euc = np.linalg.norm(sent_vectors[i] - sent_vectors[j])  # 欧氏距离
            f.write(f"   {sentences[i]}  vs  {sentences[j]}\n")
            f.write(f"      余弦相似度 = {cos:.4f}    欧氏距离 = {euc:.4f}\n\n")

print(f"文本结果已保存到: {result_txt}")

# ====================== 图形展示 ======================
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
sim_matrix_words = np.array([
    [cosine_similarity(word_vectors[i], word_vectors[j]) for j in range(len(words))]
    for i in range(len(words))
])

# 绘制单词相似度热力图
plt.figure(figsize=(12, 10))
plt.imshow(sim_matrix_words, cmap='YlOrRd', interpolation='nearest')  # YlOrRd：黄-橙-红配色
plt.colorbar(label='余弦相似度')  # 添加颜色条标注数值
plt.xticks(range(len(words)), words, rotation=45, ha='right', fontsize=12)  # x轴标签
plt.yticks(range(len(words)), words, fontsize=12)  # y轴标签
plt.title('中文单词 - 余弦相似度热力图', fontsize=16)
heatmap_path = os.path.join(result_dir, f"chinese_words_heatmap_{timestamp}.png")
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')  # dpi=300 高清保存
plt.close()
print(f"单词热力图已保存: {heatmap_path}")

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

# 为每个数据点添加中文标签
for i, w in enumerate(words):
    plt.annotate(w, xy=(word_vectors_2d[i, 0], word_vectors_2d[i, 1]),
                 xytext=(word_vectors_2d[i, 0] + 0.05, word_vectors_2d[i, 1] + 0.05),
                 fontsize=12, fontweight='bold')
plt.title(f'中文单词 - PCA 2D（相似度≥{SIM_THRESHOLD}显示连线）', fontsize=16)
plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.grid(True, alpha=0.3)
pca_path = os.path.join(result_dir, f"chinese_words_pca_{timestamp}.png")
plt.savefig(pca_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"单词PCA图已保存: {pca_path}")

# ====================== 句子部分图形 ======================
print("正在生成句子部分图形...")

# 构建句子相似度矩阵
sim_matrix_sents = np.array([
    [cosine_similarity(sent_vectors[i], sent_vectors[j]) for j in range(len(sentences))]
    for i in range(len(sentences))
])

# 绘制句子相似度热力图
plt.figure(figsize=(14, 12))
plt.imshow(sim_matrix_sents, cmap='YlGnBu', interpolation='nearest')  # YlGnBu：黄-绿-蓝配色
plt.colorbar(label='余弦相似度')
plt.xticks(range(len(sentences)), sentences, rotation=45, ha='right', fontsize=11)
plt.yticks(range(len(sentences)), sentences, fontsize=11)
plt.title('中文句子 - 余弦相似度热力图', fontsize=16)
heatmap_path2 = os.path.join(result_dir, f"chinese_sentences_heatmap_{timestamp}.png")
plt.savefig(heatmap_path2, dpi=300, bbox_inches='tight')
plt.close()
print(f"句子热力图已保存: {heatmap_path2}")

# 绘制句子 PCA 二维散点图
sent_vectors_2d = pca_2d(sent_vectors)  # 将句子向量降维到 2 维
plt.figure(figsize=(14, 11))
plt.scatter(sent_vectors_2d[:, 0], sent_vectors_2d[:, 1], s=200, c='steelblue', alpha=0.8, zorder=3)

# 在相似度 >= SIM_THRESHOLD 的句子对之间画连线
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        cos = cosine_similarity(sent_vectors[i], sent_vectors[j])
        if cos >= SIM_THRESHOLD:
            linewidth = 1 + (cos - SIM_THRESHOLD) / (1 - SIM_THRESHOLD) * 4
            alpha = 0.4 + (cos - SIM_THRESHOLD) / (1 - SIM_THRESHOLD) * 0.5
            plt.plot([sent_vectors_2d[i, 0], sent_vectors_2d[j, 0]],
                     [sent_vectors_2d[i, 1], sent_vectors_2d[j, 1]],
                     'b-', linewidth=linewidth, alpha=alpha, zorder=1)
            mid_x = (sent_vectors_2d[i, 0] + sent_vectors_2d[j, 0]) / 2
            mid_y = (sent_vectors_2d[i, 1] + sent_vectors_2d[j, 1]) / 2
            plt.annotate(f'{cos:.2f}', xy=(mid_x, mid_y), fontsize=8, color='darkblue', alpha=0.8)

# 为每个数据点添加句子标签
for i, s in enumerate(sentences):
    plt.annotate(s, xy=(sent_vectors_2d[i, 0], sent_vectors_2d[i, 1]),
                 xytext=(sent_vectors_2d[i, 0] + 0.03, sent_vectors_2d[i, 1] + 0.03),
                 fontsize=11, fontweight='bold')
plt.title(f'中文句子 - PCA 2D（相似度≥{SIM_THRESHOLD}显示连线）', fontsize=16)
plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.grid(True, alpha=0.3)
pca_path2 = os.path.join(result_dir, f"chinese_sentences_pca_{timestamp}.png")
plt.savefig(pca_path2, dpi=300, bbox_inches='tight')
plt.close()
print(f"句子PCA图已保存: {pca_path2}")

print(f"\n✅ 中文脚本运行完成！所有结果已保存到 {result_dir} 文件夹")
