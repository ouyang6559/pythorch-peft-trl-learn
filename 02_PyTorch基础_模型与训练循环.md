# 02 · PyTorch基础 · 模型与训练循环

## 本章目标
- 用 `nn.Module` 定义一个模型
- 认识**损失函数**和**优化器**
- 亲手写出**训练循环**（这 5 行代码是未来所有训练的骨架）

## 我们要解决的问题
给一堆数据，学习关系 **y = 2x + 1**。
也就是让模型自己"猜"出 `w≈2`、`b≈1`。虽然简单，但流程和训练 GPT 一模一样。

## 1. 造一点"假数据"

真实项目里数据来自文件/网络，这里我们用公式生成，方便验证模型学得对不对。

```python
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
```

## 2. 先理解"训练"到底在做什么

一句话：**不断调整参数 w、b，让模型预测值尽量接近真实 y。**

四个零件缺一不可：

| 零件 | 作用 | 本章用的 |
| --- | --- | --- |
| 模型 | 做预测 | 线性层 `y = wx + b` |
| 损失函数 | 衡量"错得有多离谱" | 均方误差 `MSELoss` |
| 优化器 | 根据梯度决定怎么调参数 | 梯度下降 `SGD` |
| 训练循环 | 把上述过程重复很多轮 | 我们手写 |

## 3. 用 nn.Module 定义模型

`nn.Module` 是 PyTorch 里所有模型的"父类"。你只需要：
1. 在 `__init__` 里**定义有哪些层/参数**
2. 在 `forward` 里**写数据怎么算**

```python
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
```

## 4. 损失函数与优化器

```python
# 损失函数：均方误差，衡量预测和真实值的差距，越小越好
loss_fn = nn.MSELoss()

# 优化器：负责"按梯度更新参数"。lr 是学习率（步子迈多大）
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
```

## 5. 训练循环 —— 请把这 5 步背下来

```
for epoch in range(轮数):
    pred = model(x)            # 1. 前向：模型预测
    loss = loss_fn(pred, y)    # 2. 算损失：预测有多错
    optimizer.zero_grad()      # 3. 清空旧梯度
    loss.backward()            # 4. 反向：算每个参数的梯度
    optimizer.step()           # 5. 更新参数
```

**这 5 步在未来训练 GPT 时完全一样**，只是模型和损失函数更复杂而已。

```python
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
```

```python
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
```

> 💡 如果损失不降反升，通常是**学习率太大**（步子太大扯着蛋）。可以试着把 `lr` 从 0.1 改成 0.01 或 0.5 观察变化。

## 6. 保存与加载模型

训练好的参数要存下来，下次直接用。PyTorch 推荐保存 `state_dict`（就是"参数字典"）。

```python
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
```

## 7. 小练习

**练习**：把真实规律改成 `y = -3x + 4`，重新造数据并训练，看看模型能不能学到 `w≈-3, b≈4`。

（提示：只需改 `true_w, true_b` 和 `y = ...` 那两行，然后重新跑上面几格。）

---

### 本章小结
- 模型 = `nn.Module` 子类，写 `__init__`（定义层）和 `forward`（定义计算）。
- 训练的本质就是**调参数让损失变小**。
- **训练循环五步**：前向 → 算损失 → 清梯度 → 反向 → 更新。这是万能骨架。
- 用 `state_dict` 保存/加载模型。

👉 **下一章：** 数据多了怎么办？学 `Dataset` 和 `DataLoader`，做一个小分类项目。
