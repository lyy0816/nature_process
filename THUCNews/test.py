import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
from model import MLPPClassifier
from utils import load_npy
RESULT_DIR = os.path.join(os.path.dirname(__file__), "result")

# ====================== 配置 ======================
DATA_DIR = r"D:\machine learning\Machine_learning_comprehensive_design\root\THUCNews\data"

X_test = load_npy(os.path.join(DATA_DIR, "X_test.npy"))
y_test = load_npy(os.path.join(DATA_DIR, "y_test.npy"))

# 加载 label_encoder
import pickle
with open(os.path.join(RESULT_DIR, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

num_classes = len(label_encoder.classes_)
print(f"类别: {label_encoder.classes_}")

# ====================== 模型 ======================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

model = MLPPClassifier(
    input_dim=X_test.shape[1],
    num_classes=num_classes
).to(device)

# 加载权重
model.load_state_dict(torch.load(os.path.join(RESULT_DIR, "best_model.pth"), map_location=device))
model.eval()
print("模型加载成功！")

# ====================== 测试 ======================
y_test_enc = label_encoder.transform(y_test)

# 批量预测
all_preds = []
all_labels = []

with torch.no_grad():
    for i in range(0, len(X_test), 64):
        batch_x = torch.tensor(X_test[i:i+64], dtype=torch.float32).to(device)
        outputs = model(batch_x)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_test_enc[i:i+64])

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# 整体准确率
acc = (all_preds == all_labels).mean() * 100
print(f"\n测试准确率: {acc:.2f}%")

# 分类报告
print("\n分类报告:")
print(classification_report(all_labels, all_preds, target_names=[str(c) for c in label_encoder.classes_]))

# 混淆矩阵
print("混淆矩阵:")
cm = confusion_matrix(all_labels, all_preds)
print(cm)