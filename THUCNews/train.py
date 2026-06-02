import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
from tqdm import tqdm

from dataset import THUCNewsDataset
from model import MLPPClassifier
from utils import load_npy, RESULT_DIR

# ====================== 配置 ======================
DATA_DIR = r"D:\machine learning\Machine_learning_comprehensive_design\root\THUCNews\data"

X_train = load_npy(os.path.join(DATA_DIR, "X_train.npy"))
y_train = load_npy(os.path.join(DATA_DIR, "y_train.npy"))
X_test  = load_npy(os.path.join(DATA_DIR, "X_test.npy"))
y_test  = load_npy(os.path.join(DATA_DIR, "y_test.npy"))

# 标签编码
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc  = label_encoder.transform(y_test)
num_classes = len(label_encoder.classes_)

print(f"类别数量: {num_classes}")
print(f"类别名称: {label_encoder.classes_}")

# ====================== Dataset & DataLoader ======================
train_dataset = THUCNewsDataset(X_train, y_train_enc, max_len=X_train.shape[1])
test_dataset  = THUCNewsDataset(X_test,  y_test_enc,  max_len=X_test.shape[1])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  
                         num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, 
                         num_workers=0, pin_memory=True)

# ====================== 模型 ======================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

model = MLPPClassifier(
    input_dim=X_train.shape[1],
    num_classes=num_classes
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

import matplotlib.pyplot as plt

# ====================== 训练 ======================
best_acc = 0.0
EPOCHS = 8
train_losses = []
test_accs = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # 测试
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc = 100 * correct / total
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    test_accs.append(acc)
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Test Acc: {acc:.2f}%")
    
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), os.path.join(RESULT_DIR, "best_model.pth"))
        print(f"✅ 保存最佳模型！准确率: {acc:.2f}%")

# 保存 label_encoder 和绘图
import pickle
with open(os.path.join(RESULT_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(label_encoder, f)

# 绘制并保存训练曲线
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS+1), train_losses, 'b-o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS+1), test_accs, 'r-o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy')
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "training_curve.png"), dpi=150)
plt.close()

print(f"\n🎉 训练完成！最佳测试准确率: {best_acc:.2f}%")
print(f"结果文件夹: {os.path.abspath(RESULT_DIR)}")