# -*- coding: utf-8 -*-
"""
02 · PyTorch基础 · 模型与训练循环
============================================================
运行方式： python 02_PyTorch基础_模型与训练循环.py
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
import matplotlib.pyplot as plt

torch.manual_seed(0)   # 固定随机种子，保证每次运行结果一致（方便教学）

# 真实规律：y = 2x + 1，再叠加一点点噪声，模拟真实世界的不完美
x = torch.linspace(-3, 3, 100)               # 在 [-3,3] 均匀取 100 个点
true_w, true_b = 2.0, 1.0
noise = torch.randn(100) * 0.3               # 高斯噪声
y = true_w * x + true_b + noise

print("x 的形状：", x.shape)
print("y 的形状：", y.shape)
plt.scatter(x.numpy(), y.numpy(), s=8)
plt.title("我们的训练数据（大致是 y = 2x + 1）")
plt.show()


import torch.nn as nn


class LinearRegression(nn.Module):
    """最简单的模型：一个线性层 y = x @ w + b。"""

    def __init__(self):
        super().__init__()                 # 固定写法：初始化父类
        # nn.Linear(输入特征数, 输出特征数)：内部自带权重 w 和偏置 b
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        """前向传播：定义输入 x 如何变成输出。"""
        return self.linear(x)


model = LinearRegression()
print(model)   # 打印模型结构，可以看到 Linear 层的参数

# 看看模型里有哪些可训练参数
for name, p in model.named_parameters():
    print(f"参数 {name}: {p.shape}，初始值 {p.item():.3f}")


# 损失函数：均方误差，衡量预测和真实值的差距，越小越好
loss_fn = nn.MSELoss()

# 优化器：负责"按梯度更新参数"。lr 是学习率（步子迈多大）
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


# ============================================================
# 正式训练：重复 200 轮
# ============================================================
# 调整数据形状：nn.Linear 需要 (样本数, 特征数) 的二维输入
x_train = x.reshape(-1, 1)   # (100,) → (100, 1)
y_train = y.reshape(-1, 1)

losses = []   # 记录每轮的损失，方便画曲线

for epoch in range(200):
    pred = model(x_train)               # 1. 前向预测
    loss = loss_fn(pred, y_train)       # 2. 计算损失

    optimizer.zero_grad()               # 3. 清空上一轮梯度
    loss.backward()                     # 4. 反向传播，求梯度
    optimizer.step()                    # 5. 更新参数

    losses.append(loss.item())

    if (epoch + 1) % 40 == 0:           # 每 40 轮打印一次，避免刷屏
        print(f"第 {epoch+1:3d} 轮  损失 = {loss.item():.4f}")


# ---------- 看看训练结果 ----------
w_learned = model.linear.weight.item()
b_learned = model.linear.bias.item()
print(f"模型学到的 w = {w_learned:.3f}  （真实值 2.0）")
print(f"模型学到的 b = {b_learned:.3f}  （真实值 1.0）")

# 画出损失下降曲线：应该是不断下降的
plt.plot(losses)
plt.xlabel("训练轮数 epoch")
plt.ylabel("损失 loss")
plt.title("损失随训练下降（这就是'在学'的信号）")
plt.show()


# ---------- 保存 ----------
torch.save(model.state_dict(), "linear_model.pt")
print("模型已保存到 linear_model.pt")

# ---------- 加载 ----------
new_model = LinearRegression()                 # 先建一个同结构的空模型
new_model.load_state_dict(torch.load("linear_model.pt"))  # 再把参数灌进去
new_model.eval()                                # 切换到"评估模式"

# 用加载的模型做个预测：当 x = 10 时，y 应该约等于 2*10+1 = 21
with torch.no_grad():                          # 评估时不需要梯度，省显存
    test_x = torch.tensor([[10.0]])
    print("x=10 时预测 y =", new_model(test_x).item())
