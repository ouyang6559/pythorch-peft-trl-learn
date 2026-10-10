# 01 · PyTorch基础 · 张量与自动求导

## 本章目标
- 会创建、查看、运算**张量（Tensor）**
- 理解张量的**形状（shape）**和**维度**
- 搞懂 PyTorch 的核心魔法：**自动求导（autograd）**

> 这是整个系列的地基。哪怕你以后只调 TRL 的 API，理解张量和梯度，才能在模型出错时知道问题出在哪。

## 1. 什么是张量？

**张量 = 可以放在 GPU 上、还能自动求导的多维数组。** 你把它理解成"升级版的 numpy 数组"就行。

按维度分：

- 0 维张量（标量）：一个数 `3.0`
- 1 维张量（向量）：`[1, 2, 3]`
- 2 维张量（矩阵）：`[[1, 2], [3, 4]]`
- 更高维：比如一张彩色图片是 3 维（通道 × 高 × 宽）

```python
import torch

# ---------- 创建张量的几种常用方式 ----------
a = torch.tensor([1, 2, 3])                 # 从 Python 列表创建（自动推断类型为整数）
b = torch.tensor([1.0, 2.0, 3.0])           # 浮点张量
c = torch.zeros(2, 3)                        # 2 行 3 列，全 0
d = torch.ones(2, 3)                         # 全 1
e = torch.arange(0, 10, 2)                   # 从 0 到 10（不含），步长 2 → [0,2,4,6,8]
f = torch.randn(2, 3)                        # 标准正态分布随机数（常用来初始化参数）
g = torch.rand(2, 3)                         # [0,1) 均匀分布随机数

print("a =", a, "| 类型：", a.dtype)
print("b =", b, "| 类型：", b.dtype)
print("c =\n", c)
print("e =", e)
print("f =\n", f)
```

## 2. 形状与属性：最容易被忽略，却最重要

**报错 90% 都是形状对不上。** 每次写完一段代码，习惯性 `print(x.shape)` 是好习惯。

```python
x = torch.randn(2, 3, 4)   # 造一个 3 维张量：2 × 3 × 4

print("x.shape  =", x.shape)     # 形状：每一维多大
print("x.ndim   =", x.ndim)      # 维度数（这里是 3）
print("x.dtype  =", x.dtype)     # 数据类型（浮点）
print("x.device =", x.device)    # 在 CPU 还是 GPU 上

# 改变形状：view / reshape（元素总数必须不变，2*3*4 = 24 = 6*4）
y = x.reshape(6, 4)
print("reshape 成 (6, 4) 后：", y.shape)

# -1 表示"这一维让 PyTorch 自己算"，非常常用
z = x.reshape(2, -1)
print("reshape 成 (2, -1) 后：", z.shape)   # -1 自动补成 12
```

## 3. 索引与切片

和 Python 列表很像，但**支持多维**。

```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

print("第 0 行：", x[0])          # 取一整行
print("第 1 行第 2 列的元素：", x[1, 2])   # 取单个元素（值为 6）
print("前两行的前两列：\n", x[:2, :2])     # 切片
print("最后一行：", x[-1])         # 负数表示从后往前数

# 冒号 : 表示"这一维全要"，常用于批量取数据
print("所有行的第 1 列：", x[:, 1])
```

## 4. 常用运算与「广播」

两个形状不同的张量做运算时，PyTorch 会自动"广播"（broadcast）对齐，规则和 numpy 一致。

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([10.0, 20.0, 30.0])

print("加法：", a + b)          # 对应元素相加
print("乘法：", a * b)          # 对应元素相乘（不是矩阵乘法！）
print("点积：", torch.dot(a, b))  # 1*10 + 2*20 + 3*30 = 140

# ---------- 矩阵乘法：深度学习里最常见的运算 ----------
m1 = torch.randn(2, 3)   # 2×3
m2 = torch.randn(3, 5)   # 3×5
print("矩阵乘法 (2,3)@(3,5) 的结果形状：", (m1 @ m2).shape)  # → (2,5)

# ---------- 广播示例 ----------
x = torch.ones(2, 3)         # 形状 (2, 3)
y = torch.tensor([1.0, 2.0, 3.0])  # 形状 (3,)
print("(2,3) + (3,) 广播后：\n", x + y)   # y 被扩展成 (2,3) 去加
```

## 5. 与 NumPy 互转

PyTorch 和 NumPy 可以零拷贝互转，处理数据时很方便。

```python
import numpy as np

np_arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(np_arr)     # numpy → 张量（共享内存）
back = t.numpy()                 # 张量 → numpy

print("numpy 数组：", np_arr)
print("转成张量：", t)
print("转回 numpy：", back)

# 注意：共享内存意味着改一个另一个也会变（初学先了解即可）
```

## 6. 核心中的核心：自动求导（autograd）

这是 PyTorch 区别于 numpy 的**最关键能力**。

**它解决什么问题？** 训练模型时，我们要知道"参数往哪个方向调，损失才能变小"，这个方向就是**梯度**。
手算梯度又累又容易错，而 PyTorch 能**自动**帮你算好。

用法只有三步：

1. 把需要求导的张量设成 `requires_grad=True`
2. 做前向运算得到结果
3. 调用 `.backward()`，梯度会自动存进 `.grad`

> 数学上：对 `y = x²` 求导得到 `dy/dx = 2x`。当 x=3 时梯度应为 6。下面验证一下。

```python
# ============================================================
# 自动求导最小例子：y = x^2，求 x=3 处的梯度
# ============================================================
x = torch.tensor(3.0, requires_grad=True)   # 告诉 PyTorch：x 需要求导

y = x ** 2                                   # 前向：y = 3^2 = 9
print("y =", y.item())

y.backward()                                 # 反向：自动计算 dy/dx
# 理论值 dy/dx = 2x = 6
print("x 的梯度 dy/dx =", x.grad.item())     # 应输出 6.0
```

```python
# ============================================================
# 稍实际一点的例子：线性函数 y = w*x + b
# 这在神经网络里就是"一个神经元"的基本形态
# ============================================================
w = torch.tensor(2.0, requires_grad=True)   # 权重
b = torch.tensor(1.0, requires_grad=True)   # 偏置
x = torch.tensor(3.0)                        # 输入（不需要求导）

y = w * x + b          # y = 2*3 + 1 = 7
y.backward()

# d(y)/dw = x = 3 ；d(y)/db = 1
print("对 w 的梯度 =", w.grad.item())   # 3.0
print("对 b 的梯度 =", b.grad.item())   # 1.0
```

> ⚠️ **重要习惯**：PyTorch 的梯度是**累加**的，不是覆盖。
> 所以多轮训练时，每轮开始前要调用 `optimizer.zero_grad()` 或把 `grad` 清零，否则梯度会越加越大。
> 下面手动演示一次"梯度累加"的现象。

```python
x = torch.tensor(3.0, requires_grad=True)

(x ** 2).backward()      # 第一次反向：grad = 6
print("第一次后 grad =", x.grad.item())

(x ** 2).backward()      # 第二次反向：grad 会累加 → 6 + 6 = 12
print("第二次后 grad =", x.grad.item(), "（被累加了！）")

x.grad.zero_()           # 手动清零
print("清零后 grad =", x.grad.item())
```

## 7. 小练习

**练习**：令 `z = 3*a + 2*b`，其中 `a=1.0, b=4.0`。
求 `dz/da` 和 `dz/db` 分别是多少？（提示：先想数学答案，再用代码验证）

<details>
<summary>点开看答案</summary>

数学上：`dz/da = 3`，`dz/db = 2`。
</details>

```python
# ============================================================
# 练习答案
# ============================================================
a = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(4.0, requires_grad=True)

z = 3 * a + 2 * b
z.backward()

print("dz/da =", a.grad.item(), "（应为 3.0）")
print("dz/db =", b.grad.item(), "（应为 2.0）")
```

---

### 本章小结
- **张量**是 PyTorch 的基本数据结构，比 numpy 多了 GPU 支持和自动求导。
- **形状（shape）**永远是排查问题的第一线索。
- **自动求导三步走**：设 `requires_grad=True` → 前向运算 → `.backward()` 得到 `.grad`。
- 记得**梯度会累加**，训练时要清零。

👉 **下一章：** 把这些零件组装起来，亲手写出一个**完整的训练循环**。
