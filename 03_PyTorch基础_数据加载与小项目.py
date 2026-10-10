# -*- coding: utf-8 -*-
"""
03 · PyTorch基础 · 数据加载与小项目
============================================================
运行方式： python 03_PyTorch基础_数据加载与小项目.py
说明：
  - 首次运行会联网下载模型/依赖，请保持网络畅通。
  - 含 matplotlib 的章节若无图形界面，可把 plt.show() 换成 plt.savefig('fig.png')。
  - 建议先 'pip install -r requirements.txt'。
"""

# ============================================================
# 环境准备：先运行这一格安装本教程所需依赖
# Colab 每新开一个 notebook 都是全新环境，建议先跑这一格
# 已装过的库会自动跳过，不会重复下载
# ============================================================
# 依赖安装（在终端执行下面这行）：pip install -q transformers datasets trl peft accelerate matplotlib


import torch
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)

# ---------- 生成数据 ----------
# 类别 0：中心在 (-2, -2) 的一团点
n = 500
x0 = torch.randn(n, 2) + torch.tensor([-2.0, -2.0])
y0 = torch.zeros(n, dtype=torch.long)      # 标签 0

# 类别 1：中心在 (2, 2) 的一团点
x1 = torch.randn(n, 2) + torch.tensor([2.0, 2.0])
y1 = torch.ones(n, dtype=torch.long)       # 标签 1

# 拼在一起
X = torch.cat([x0, x1], dim=0)   # (1000, 2)
Y = torch.cat([y0, y1], dim=0)   # (1000,)
print("X 形状：", X.shape, " Y 形状：", Y.shape)


# 可视化一下，直观感受两类数据
import matplotlib.pyplot as plt
plt.scatter(X[Y == 0][:, 0], X[Y == 0][:, 1], s=8, label="类别 0")
plt.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1], s=8, label="类别 1")
plt.legend()
plt.title("两团可分的数据")
plt.show()


class MyDataset(Dataset):
    """把我造好的 X、Y 封装成标准数据集。"""

    def __init__(self, features, labels):
        self.features = features   # 特征，形状 (N, 2)
        self.labels = labels       # 标签，形状 (N,)

    def __len__(self):
        # 告诉 DataLoader 一共有多少条数据
        return len(self.labels)

    def __getitem__(self, idx):
        # 返回第 idx 条：特征是 2 维向量，标签是类别编号
        return self.features[idx], self.labels[idx]


dataset = MyDataset(X, Y)
print("数据集大小：", len(dataset))
print("第 0 条数据：", dataset[0])


loader = DataLoader(dataset, batch_size=32, shuffle=True)  # shuffle 打乱顺序，避免模型记住顺序

# 看看一个 batch 长什么样
for batch_x, batch_y in loader:
    print("一个 batch 的特征形状：", batch_x.shape)   # (32, 2)
    print("一个 batch 的标签形状：", batch_y.shape)   # (32,)
    break   # 只看第一个就够


import torch.nn as nn

# 设备管理：有 GPU 就用 GPU，没有就用 CPU（本教程全靠这行自动切换）
device = "cuda" if torch.cuda.is_available() else "cpu"
print("使用设备：", device)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),   # 输入 2 维 → 隐藏层 16 维
            nn.ReLU(),          # 激活函数，引入非线性
            nn.Linear(16, 2),   # 隐藏层 → 输出 2 类
        )

    def forward(self, x):
        return self.net(x)


# 创建模型并搬到设备上（GPU 时需要）
model = Classifier().to(device)
print(model)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)   # Adam 比 SGD 更省心

epochs = 20
for epoch in range(epochs):
    model.train()                # 切换到训练模式
    total_loss = 0.0

    for batch_x, batch_y in loader:
        # 把数据搬到和模型同一个设备
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        logits = model(batch_x)          # 1. 前向：得到每个类别的分数
        loss = loss_fn(logits, batch_y)  # 2. 算损失

        optimizer.zero_grad()            # 3. 清梯度
        loss.backward()                  # 4. 反向
        optimizer.step()                 # 5. 更新

        total_loss += loss.item()

    avg = total_loss / len(loader)
    print(f"第 {epoch+1:2d} 轮  平均损失 = {avg:.4f}")


model.eval()                     # 评估模式
correct, total = 0, 0

with torch.no_grad():            # 关掉梯度计算
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        logits = model(batch_x)
        preds = logits.argmax(dim=1)     # 分数最大的那个类别就是预测结果
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

print(f"准确率：{correct / total * 100:.1f}%")   # 数据可分，应该接近 100%
