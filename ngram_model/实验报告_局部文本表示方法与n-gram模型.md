![image-20260419111129061](C:\Users\22630\AppData\Roaming\Typora\typora-user-images\image-20260419111129061.png)



# 基于局部文本表示方法的中英文相似度比对及 n-gram 模型实验报告



| 实验名称： | 基于局部文本表示方法的中英文相似度比对及 n-gram 模型实验 |
| :--------- | ------------------------------------------------------- |
| 学院名称： | 人工智能学院                                            |
| 班级：     | 人工智能231                                             |
| 学号：     | 202308064704                                            |
| 学生姓名： | 刘原野                                                  |
| 指导老师： | 杨华                                                    |

## 一、实验背景

在自然语言处理中，文本表示是将人类语言转换为计算机可处理数值形式的基础步骤。与前一次实验使用预训练 fastText 词向量（全局表示方法）不同，本次实验聚焦于**局部文本表示方法**——即仅基于当前语料库本身来构建文本向量，不依赖外部预训练模型。

局部方法的核心思想是通过统计词频、共现等信息来刻画文本特征，具有实现简单、无需大规模预训练数据、计算资源需求低等优点。同时，n-gram 模型作为一种经典的统计语言模型，在文本生成、语言建模等任务中也有广泛应用。

本实验使用 `sklearn` 库中的 `CountVectorizer` 和 `TfidfVectorizer` 分别实现 One-hot Encoding、BoW（词袋模型）、TF-IDF、字符级 n-gram 四种方法，并独立实现了一个基于 jieba 分词的 n-gram 语言模型，对中英文单词及句子进行相似度比对与文本生成。

---

## 二、实验目的

1. 理解 One-hot Encoding、BoW、TF-IDF、n-gram 四种局部文本表示方法的基本原理；
2. 掌握使用 `sklearn` 库实现文本向量化的方法；
3. 对比四种方法在中英文单词和句子相似度计算中的表现差异；
4. 实现独立的 n-gram 语言模型，掌握文本生成与预测的基本流程；
5. 通过热力图与 PCA 降维散点图直观观察不同方法的特征分布规律。

---

## 三、实验环境

| 项目 | 内容 |
|------|------|
| 操作系统 | Windows 11 Pro |
| Python 版本 | Python 3.x |
| 核心依赖库 | `numpy`、`matplotlib`、`sklearn`、`jieba` |
| 相似度度量 | 余弦相似度 + 欧氏距离 |
| 结果输出目录 | `similarity_comparsion/result_local/`（相似度实验）、`ngram_model/`（n-gram 模型） |

---

## 四、实验原理

### 4.1 One-hot Encoding（独热编码）

One-hot Encoding 是最简单的文本表示方法。它将每个词或文本映射为一个二值向量：

- 向量维度 = 词汇表大小
- 如果某个词出现在文本中，对应位置为 1，否则为 0
- 在 `sklearn` 中通过 `CountVectorizer(binary=True)` 实现

**优点**：实现简单、直观。  
**缺点**：无法表达词频信息，所有词等权重对待，向量稀疏且维度高。

### 4.2 BoW（词袋模型，Bag of Words）

BoW 是 One-hot Encoding 的扩展，它统计每个词在文本中出现的**次数**：

- 向量维度 = 词汇表大小
- 每个位置的值 = 对应词在文本中出现的次数
- 在 `sklearn` 中通过 `CountVectorizer()` 实现

**优点**：比 One-hot 多了词频信息，能区分高频词和低频词。  
**缺点**：忽略了词序信息，无法捕捉语义关系，对常见词（如"的"、"是"）赋予过高权重。

### 4.3 TF-IDF（词频-逆文档频率）

TF-IDF 在 BoW 的基础上引入了**逆文档频率**（IDF）加权，用于降低常见词的权重、提升罕见但有区分度的词的权重：

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

其中：
- $\text{TF}(t, d)$：词 $t$ 在文档 $d$ 中的出现频率
- $\text{IDF}(t) = \log \frac{N}{\text{DF}(t)}$，$N$ 为文档总数，$\text{DF}(t)$ 为包含词 $t$ 的文档数

在 `sklearn` 中通过 `TfidfVectorizer()` 实现。

**优点**：自动降权常见词（如"的"、"the"），提升有区分度的词的权重。  
**缺点**：仍然忽略词序信息，无法捕捉上下文语义。

### 4.4 n-gram 模型（字符级）

n-gram 模型将文本拆分为连续的 n 个字符（或词）的片段：

- **字符级 n-gram**：将文本按字符滑窗切分，例如"苹果好吃"的 2-gram 为："苹果"、"果好"、"好吃"
- 在 `sklearn` 中通过 `CountVectorizer(analyzer='char', ngram_range=(2, 4))` 实现，同时提取 2-gram、3-gram 和 4-gram

**优点**：能够捕捉局部字符共现模式，对拼写变体有一定鲁棒性。  
**缺点**：向量维度随 n 和语料规模快速增长，计算开销较大。

### 4.5 n-gram 语言模型（独立实现）

除了上述基于 `sklearn` 的向量化方法，本实验还独立实现了一个 n-gram 语言模型（`NGramModel` 类），其核心原理为**马尔可夫假设**：

$$P(w_i | w_1, w_2, \ldots, w_{i-1}) \approx P(w_i | w_{i-n+1}, \ldots, w_{i-1})$$

即：一个词出现的概率仅由其前面 $n-1$ 个词决定。模型通过统计训练语料中 n-gram 的出现次数来估计条件概率，支持：

- **字符级预测**：给定前缀，预测下一个字符（适合中文猜字、输入法补全）
- **词级预测**：给定前缀，预测下一个词（适合文本续写）
- **文本生成**：根据前缀自动续写指定长度的文本

---

## 五、实验数据集

### 5.1 单词数据集（27 个词）

| 类别 | 中文 | 英文 |
|------|------|------|
| 水果 | 苹果、香蕉、橙子、葡萄、芒果、桃子 | apple, banana, orange, grape, mango, peach |
| 动物 | 狗、猫、狮子、老虎、马、鸟 | dog, cat, lion, tiger, horse, bird |
| 科技 | 电脑、手机、键盘、屏幕、相机 | computer, phone, keyboard, screen, camera |
| 颜色 | 红色、蓝色、绿色、黄色、紫色 | red, blue, green, yellow, purple |
| 自然 | 河流、高山、森林、海洋、云朵 | river, mountain, forest, ocean, cloud |

### 5.2 句子数据集（17 句）

| 类别 | 中文 | 英文 |
|------|------|------|
| 食物 | 我喜欢吃苹果、我喜欢吃香蕉、草莓酸酸甜甜很好吃 | I love eating apple, I love eating banana, Strawberries are sweet |
| 科技 | 电脑速度很快、笔记本电脑很轻便、手机屏幕真清晰、键盘打字手感很好 | The computer is very fast, The laptop is portable, The phone screen is clear, The keyboard feels great |
| 天气 | 今天天气非常晴朗、今天气温很高很热、天空中飘着白云 | Today is a sunny day, The weather is very hot, The sky is cloudy today |
| 动物 | 狗狗非常友好、猫咪喜欢睡觉、鸟儿歌声很动听 | The dog is very friendly, The cat likes to sleep, The bird sings beautifully |
| 颜色 | 天空是蓝色的、树叶变成绿色 | The sky is blue today, The leaves turn green |
| 自然 | 河流静静流淌、高山非常高耸 | The river flows quietly, The mountain is very high |

### 5.3 n-gram 模型训练语料

```python
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
```

---

## 六、实验代码与关键函数

### 6.1 核心工具函数简介

#### `tokenize_chinese(text)` — 中文分词预处理

```python
def tokenize_chinese(text):
    """对中文文本进行 jieba 分词，用空格连接，供 sklearn 向量器使用"""
    return ' '.join(jieba.lcut(text))
```

- **功能**：调用 jieba 分词器将中文文本切分为词序列，用空格拼接为字符串
- **作用**：sklearn 的 `CountVectorizer` / `TfidfVectorizer` 默认按空格分词，因此中文需要先预分词
- **示例**：`"我喜欢吃苹果"` → `"我 喜欢 吃 苹果"`

#### `pca_2d(data)` — PCA 降维可视化

```python
def pca_2d(data):
    """使用 SVD（奇异值分解）将高维数据降维到 2 维"""
    mean = np.mean(data, axis=0)
    centered = data - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ Vt.T[:, :2]
```

- **功能**：通过 SVD 分解将高维向量投影到前两个主成分方向，用于二维可视化
- **步骤**：① 计算均值 → ② 数据中心化 → ③ SVD 分解 → ④ 取前 2 列投影
- **输出**：`(n_samples, 2)` 的二维坐标矩阵

#### `run_comparison(...)` — 核心实验流程函数

```python
def run_comparison(vectorizer, method_name, texts, labels, data_type, f_out, sim_threshold=0.3):
    """
    运行一种文本表示方法的完整相似度比对实验
    :param vectorizer: sklearn 向量器（CountVectorizer / TfidfVectorizer）
    :param method_name: 方法名称标签（如 "OneHot"、"BoW"）
    :param texts: 待向量化的文本列表（已分词或原始文本）
    :param labels: 原始文本标签（用于图表标注）
    :param data_type: "words" 或 "sentences"，区分单词/句子
    :param f_out: 输出文本文件的文件句柄
    :param sim_threshold: PCA 图中显示连线的相似度阈值
    :return: 向量矩阵和相似度矩阵
    """
```

- **功能**：封装完整的实验流程——向量化 → 计算相似度 → 写入文本结果 → 绘制热力图 → 绘制 PCA 散点图
- **调用方式**：每种方法分别对单词和句子各调用一次，共调用 8 次（4 方法 × 2 数据类型）

#### `cosine_similarity(v1, v2)` — 余弦相似度（sklearn 内置）

```python
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(vecs)  # 计算所有向量两两之间的余弦相似度矩阵
```

- **公式**：$\text{cosine\_sim}(A, B) = \frac{A \cdot B}{\|A\| \times \|B\|}$
- **取值范围**：`[-1, 1]`，值越大表示越相似
- **输出**：`(n, n)` 的相似度矩阵，对角线为 1（自身与自身完全相似）

### 6.2 向量器配置

```python
# One-hot Encoding：binary=True，仅标记是否出现
onehot = CountVectorizer(binary=True)

# BoW：统计词频
bow = CountVectorizer()

# TF-IDF：词频-逆文档频率加权
tfidf = TfidfVectorizer()

# n-gram（字符级）：提取 2~4 字符的 n-gram
ngram_cn = CountVectorizer(analyzer='char', ngram_range=(2, 4))       # 中文
ngram_en = CountVectorizer(analyzer='char_wb', ngram_range=(2, 4))    # 英文（词边界内）
```

| 参数 | 说明 |
|------|------|
| `binary=True` | 仅输出 0/1，不统计频次（One-hot） |
| `analyzer='char'` | 按字符切分（中文 n-gram） |
| `analyzer='char_wb'` | 按词边界内的字符切分，避免跨词拼接（英文 n-gram） |
| `ngram_range=(2, 4)` | 同时提取 2-gram、3-gram、4-gram |

### 6.3 n-gram 语言模型关键方法

#### `NGramModel.train(texts, use_word, use_char)` — 模型训练

```python
def train(self, texts, use_word=True, use_char=True):
    for text in texts:
        # 字符级：对文本按字符滑窗，统计 (context) -> {next: count}
        if use_char:
            chars = list(text)
            for i in range(len(chars) - self.n + 1):
                context = tuple(chars[i:i + self.n - 1])
                next_item = chars[i + self.n - 1]
                self.ngram_counts[context][next_item] += 1

        # 词级：对文本先 jieba 分词，再按词滑窗统计
        if use_word:
            words = list(jieba.cut(text))
            for i in range(len(words) - self.n + 1):
                context = tuple(words[i:i + self.n - 1])
                next_item = words[i + self.n - 1]
                self.ngram_counts[context][next_item] += 1
```

- **核心逻辑**：对每条文本进行滑窗切分，以前 `n-1` 个 token 为上下文，统计下一个 token 的出现频次
- **存储结构**：`defaultdict(Counter)` —— `{ 上下文元组: {下一个词: 频次} }`

#### `NGramModel.predict_next(context, mode, top_k)` — 预测下一个词

```python
def predict_next(self, context, mode="char", top_k=5):
    # 取最后 n-1 个 token 作为上下文
    context_tokens = tokens[-(self.n - 1):]
    context_tuple = tuple(context_tokens)

    if context_tuple in self.ngram_counts:
        counter = self.ngram_counts[context_tuple]
        total = sum(counter.values())
        # 返回 top_k 个候选及其概率
        candidates = [(item, count / total) for item, count in counter.most_common(top_k)]
        return candidates
    else:
        # 未命中时回退：随机返回词汇表中的词
        return [(item, 0.1) for item in random.sample(list(self.vocab), min(top_k, len(self.vocab)))]
```

- **功能**：给定上下文，查表返回概率最高的 top_k 个候选
- **回退策略**：当上下文未在训练数据中出现时，随机返回词汇表中的词作为兜底

#### `NGramModel.generate(prefix, length, mode)` — 文本自动续写

```python
def generate(self, prefix, length=20, mode="char"):
    result = prefix
    for _ in range(length):
        candidates = self.predict_next(result, mode=mode, top_k=3)
        if candidates:
            # 按概率加权采样，而非贪心取最高概率
            next_item = random.choices([c[0] for c in candidates],
                                     weights=[c[1] for c in candidates])[0]
            result += next_item if mode == "char" else " " + next_item
    return result.strip()
```

- **功能**：根据前缀循环预测并拼接，生成指定长度的文本
- **采样策略**：使用 `random.choices` 按概率加权采样（非贪心），增加生成的多样性

### 6.4 英文相似度比对（english.py）

英文版与中文版结构完全一致，区别在于：
- 英文不需要 jieba 分词（sklearn 默认按空格分词）
- n-gram 使用 `analyzer='char_wb'`（词边界内的字符 n-gram，避免跨词拼接）
- 标签和图表标题使用英文

---

## 七、实验结果与分析

### 7.1 向量维度对比

| 方法 | 中文单词维度 | 中文句子维度 | 英文单词维度 | 英文句子维度 |
|------|:-----------:|:-----------:|:-----------:|:-----------:|
| One-hot | 23 | 23 | 27 | 45 |
| BoW | 23 | 41 | 27 | 45 |
| TF-IDF | 23 | 41 | 27 | 45 |
| n-gram | — | — | — | — |

> 说明：One-hot 和 BoW/TF-IDF 的单词维度等于词汇表大小（中文分词后去重，英文默认分词后去重）；句子维度取决于语料库中出现的不同词数。n-gram 的维度由字符组合决定，通常远大于词汇表大小。

### 7.2 中文实验结果

#### 7.2.1 One-hot — 单词

所有单词对的余弦相似度均为 **0.0000**，欧氏距离均为 1.4142。因为每个单词只出现一次，One-hot 向量互相正交，**无法区分语义相近的词**。

![中文单词 One-hot 热力图](../similarity_comparsion/result_local/chinese_OneHot_words_heatmap_20260518_143508.png)

> 图：中文单词 One-hot 余弦相似度热力图。所有非对角线位置均为 0，无法体现语义关系。

![中文单词 One-hot PCA](../similarity_comparsion/result_local/chinese_OneHot_words_pca_20260518_143508.png)

> 图：中文单词 One-hot PCA 二维散点图。各单词均匀分布，无聚类趋势。

#### 7.2.2 BoW — 单词

由于单词仅包含单个词，BoW 的单词向量与 One-hot 完全相同（每个词只出现 1 次），结果一致。

#### 7.2.3 BoW — 句子

| 文本对 | 余弦相似度 | 欧氏距离 |
|--------|:---------:|:-------:|
| 我喜欢吃苹果 vs 我喜欢吃香蕉 | **0.5000** | 1.4142 |
| 我喜欢吃苹果 vs 草莓酸酸甜甜很好吃 | 0.0000 | 2.2361 |
| 我喜欢吃苹果 vs 电脑速度很快 | 0.0000 | 2.2361 |
| 我喜欢吃苹果 vs 猫咪喜欢睡觉 | **0.4082** | 1.7321 |

**分析**："我喜欢吃苹果"与"我喜欢吃香蕉"共享"我"、"喜欢"、"吃"三个词，因此 BoW 余弦相似度达 0.5。与"猫咪喜欢睡觉"共享"喜欢"一词，也有一定相似度。与不同话题的句子则无重叠词，相似度为 0。

![中文句子 BoW 热力图](../similarity_comparsion/result_local/chinese_BoW_sentences_heatmap_20260518_143508.png)

> 图：中文句子 BoW 余弦相似度热力图。食物类句子之间、天气类句子之间有明显的相似度高亮区域。

![中文句子 BoW PCA](../similarity_comparsion/result_local/chinese_BoW_sentences_pca_20260518_143508.png)

> 图：中文句子 BoW PCA 二维散点图。共享词汇多的句子对在图中距离较近。

#### 7.2.4 TF-IDF — 句子

![中文句子 TF-IDF 热力图](../similarity_comparsion/result_local/chinese_TFIDF_sentences_heatmap_20260518_143508.png)

> 图：中文句子 TF-IDF 余弦相似度热力图。TF-IDF 自动降权了"的"、"是"等停用词，使有区分度的词权重更高，语义相关的句子对获得更准确的相似度。

![中文句子 TF-IDF PCA](../similarity_comparsion/result_local/chinese_TFIDF_sentences_pca_20260518_143508.png)

> 图：中文句子 TF-IDF PCA 二维散点图。聚类效果优于 BoW，同类主题的句子更集中。

#### 7.2.5 n-gram — 句子

![中文句子 n-gram 热力图](../similarity_comparsion/result_local/chinese_Ngram_sentences_heatmap_20260518_143508.png)

> 图：中文句子 n-gram 余弦相似度热力图。n-gram 通过字符级共现模式捕捉相似度，能识别共享子串的句子对。

![中文句子 n-gram PCA](../similarity_comparsion/result_local/chinese_Ngram_sentences_pca_20260518_143508.png)

> 图：中文句子 n-gram PCA 二维散点图。

#### 7.2.6 中文四种方法 PCA 对比

| One-hot | BoW |
|:-------:|:---:|
| ![OneHot PCA](../similarity_comparsion/result_local/chinese_OneHot_sentences_pca_20260518_143508.png) | ![BoW PCA](../similarity_comparsion/result_local/chinese_BoW_sentences_pca_20260518_143508.png) |
| 各句均匀分布，无聚类 | 共享词汇的句子开始聚集 |

| TF-IDF | n-gram |
|:------:|:------:|
| ![TFIDF PCA](../similarity_comparsion/result_local/chinese_TFIDF_sentences_pca_20260518_143508.png) | ![Ngram PCA](../similarity_comparsion/result_local/chinese_Ngram_sentences_pca_20260518_143508.png) |
| 聚类效果最好，主题分离明显 | 字符共现带来额外的相似度信息 |

### 7.3 英文实验结果

#### 7.3.1 BoW — 句子

| 文本对 | 余弦相似度 | 欧氏距离 |
|--------|:---------:|:-------:|
| I love eating apple vs I love eating banana | **0.6667** | 1.4142 |
| I love eating apple vs Strawberries are sweet | 0.0000 | 2.4495 |
| I love eating apple vs The computer is very fast | 0.0000 | 2.8284 |

**分析**：英文句子"I love eating apple"与"I love eating banana"共享"I"、"love"、"eating"三个词（共 4 个词中的 3 个），BoW 相似度达 0.6667，高于中文对应句对的 0.5。这是因为英文句子较短，共享词占比更高。

![英文句子 BoW 热力图](../similarity_comparsion/result_local/english_BoW_sentences_heatmap_20260518_143307.png)

> 图：英文句子 BoW 余弦相似度热力图。食物类（apple/banana/strawberries）和科技类句子之间有明显高相似度区域。

![英文句子 BoW PCA](../similarity_comparsion/result_local/english_BoW_sentences_pca_20260518_143307.png)

> 图：英文句子 BoW PCA 二维散点图。同类主题的句子在空间中聚集。

#### 7.3.2 TF-IDF — 句子

![英文句子 TF-IDF 热力图](../similarity_comparsion/result_local/english_TFIDF_sentences_heatmap_20260518_143307.png)

> 图：英文句子 TF-IDF 余弦相似度热力图。降权了"The"、"is"、"very"等停用词后，语义相关的句子对获得更准确的相似度评分。

![英文句子 TF-IDF PCA](../similarity_comparsion/result_local/english_TFIDF_sentences_pca_20260518_143307.png)

> 图：英文句子 TF-IDF PCA 二维散点图。

#### 7.3.3 n-gram — 句子

![英文句子 n-gram 热力图](../similarity_comparsion/result_local/english_Ngram_sentences_heatmap_20260518_143307.png)

> 图：英文句子 n-gram 余弦相似度热力图。字符级 n-gram 能捕捉"The ... is ..."等共享结构模式。

![英文句子 n-gram PCA](../similarity_comparsion/result_local/english_Ngram_sentences_pca_20260518_143307.png)

> 图：英文句子 n-gram PCA 二维散点图。

#### 7.3.4 英文四种方法热力图对比

| One-hot | BoW |
|:-------:|:---:|
| ![OneHot Heatmap](../similarity_comparsion/result_local/english_OneHot_sentences_heatmap_20260518_143307.png) | ![BoW Heatmap](../similarity_comparsion/result_local/english_BoW_sentences_heatmap_20260518_143307.png) |
| One-hot 对角线外全为 0 | BoW 开始出现非零相似度区域 |

| TF-IDF | n-gram |
|:------:|:------:|
| ![TFIDF Heatmap](../similarity_comparsion/result_local/english_TFIDF_sentences_heatmap_20260518_143307.png) | ![Ngram Heatmap](../similarity_comparsion/result_local/english_Ngram_sentences_heatmap_20260518_143307.png) |
| TF-IDF 停用词降权后效果最好 | n-gram 捕捉字符共现模式 |

### 7.4 n-gram 语言模型运行结果

使用 11 条中文语料训练 3-gram 模型后：

**字符级预测示例**：
```
输入: "苹果"
预测: [('果', 1.0)]   → 模型预测下一个字符为"果"的概率为 100%
```

**词级预测示例**：
```
输入: "机器学习"
预测: [('和', 1.0)]   → 模型预测下一个词为"和"的概率为 100%
```

**字符级文本生成**：
```
输入前缀: "今天天气真好"
生成结果: "今天天气真好适合出去玩自然语言处理方向的风景"
```

**词级文本生成**：
```
输入前缀: "自然语言处理"
生成结果: "自然语言处理 是 人工智能 的 重要 分支 人工智能 的 重要 方向"
```

模型能够基于训练语料进行合理的文本续写，但由于训练数据量较小，生成结果较为单一。

---

## 八、四种方法对比总结

| 特性 | One-hot | BoW | TF-IDF | n-gram |
|------|---------|-----|--------|--------|
| 向量维度 | 词汇表大小 | 词汇表大小 | 词汇表大小 | 远大于词汇表 |
| 词频信息 | ✗ | ✓ | ✓ | ✓ |
| 词序信息 | ✗ | ✗ | ✗ | 部分（局部） |
| 停用词降权 | ✗ | ✗ | ✓（自动） | ✗ |
| 语义理解 | ✗ | ✗ | ✗ | ✗ |
| 实现复杂度 | 低 | 低 | 低 | 中 |
| 计算开销 | 低 | 低 | 低 | 高 |
| 适用场景 | 简单分类 | 文本分类 | 信息检索 | 拼写检查、语言建模 |

**关键结论**：

1. **One-hot** 无法区分任何词对的语义相似度（所有余弦相似度为 0），仅适合最基础的文本编码；
2. **BoW** 能识别共享词汇的句子对，但对停用词敏感，容易被"的"、"是"等常见词干扰；
3. **TF-IDF** 自动降权常见词，是局部方法中表现最好的，适合信息检索和文本分类任务；
4. **n-gram** 能捕捉字符级局部模式，适合拼写检查和语言建模，但向量维度高、计算开销大；
5. 所有四种局部方法**无法捕捉语义关系**（如"苹果"与"香蕉"同为水果，在 One-hot/BoW 中相似度为 0），这是预训练词向量（如 fastText）的优势所在。

---

## 九、实验心得

通过本次实验，我深入理解了四种局部文本表示方法的原理和实现方式：

1. **从简单到复杂**：One-hot → BoW → TF-IDF → n-gram，每种方法在前一种的基础上增加了更多信息（词频、逆文档频率、局部序列），但也带来了更高的计算开销；
2. **局部 vs 全局**：与预训练词向量相比，局部方法不依赖外部数据，实现简单，但无法捕捉深层语义关系。在实际应用中，两者常结合使用——用局部方法处理领域特定文本，用预训练模型补充语义信息；
3. **n-gram 的双重角色**：n-gram 既可以作为文本表示方法（用于相似度计算），也可以作为语言模型（用于文本生成）。独立实现的 n-gram 模型让我更直观地理解了马尔可夫假设和条件概率估计的过程；
4. **工具库的价值**：`sklearn` 的 `CountVectorizer` 和 `TfidfVectorizer` 封装了分词、构建词汇表、向量化等步骤，用几行代码就能完成复杂的文本处理流程，极大地提高了开发效率。

---

## 十、文件清单

| 文件 | 说明 |
|------|------|
| `similarity_comparsion/chinese.py` | 中文四种局部方法相似度比对脚本 |
| `similarity_comparsion/english.py` | 英文四种局部方法相似度比对脚本 |
| `similarity_comparsion/result_local/` | 相似度比对结果（34 个文件：32 张图 + 2 份文本报告） |
| `ngram_model/ngram_model.py` | n-gram 语言模型类定义与训练脚本 |
| `ngram_model/call_ngram.py` | n-gram 模型调用演示脚本 |
| `ngram_model/ngram_model.pkl` | 训练好的 n-gram 模型文件 |
