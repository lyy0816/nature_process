import torch
import jieba
import pickle
import os
import joblib

from model import MLPPClassifier

DATA_DIR = r"D:\machine learning\Machine_learning_comprehensive_design\root\THUCNews\data"
RESULT_DIR = os.path.join(os.path.dirname(__file__), "result")

# 加载 TfidfVectorizer
vectorizer = joblib.load(os.path.join(DATA_DIR, "tfidf_vectorizer.pkl"))

# 加载 label2id 并构建 id2label
with open(os.path.join(os.path.dirname(__file__), "label2id.json"), "r", encoding="utf-8") as f:
    label2id = eval(f.read())
id2label = {v: k for k, v in label2id.items()}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MLPPClassifier(input_dim=5000, num_classes=len(label2id)).to(device)
model.load_state_dict(torch.load(os.path.join(RESULT_DIR, "best_model.pth"), map_location=device))
model.eval()
print("模型加载成功！")

print("=== THUCNews 新闻分类预测系统 (MLP) ===")
print("输入新闻内容进行分类，输入 'q' 退出\n")
print(f"可用类别: {list(id2label.values())}")

with open(os.path.join(RESULT_DIR, "predictions.txt"), "a", encoding="utf-8") as log:
    while True:
        text = input("\n请输入新闻文本: ").strip()
        if text.lower() == 'q':
            break
        if not text:
            continue

        # 分词
        tokens = ' '.join(jieba.cut(text))
        # TF-IDF 向量化
        features = vectorizer.transform([tokens]).toarray()[0]

        input_tensor = torch.from_numpy(features).unsqueeze(0).float().to(device)

        with torch.no_grad():
            output = model(input_tensor)
            pred_idx = torch.argmax(output, dim=1).item()
            pred_label = id2label[pred_idx]

        print(f"预测类别: {pred_idx} - {pred_label}")

        log.write(f"输入: {text}\n预测类别: {pred_idx} - {pred_label}\n{'='*60}\n")

print("\n交互结束，结果已保存到 result/predictions.txt")