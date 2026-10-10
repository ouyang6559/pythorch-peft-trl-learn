# 05 · PEFT · LoRA原理与最小示例

## 本章目标
- 理解为什么需要 **PEFT**（全量微调太贵）
- 搞懂 **LoRA** 的核心思想
- 用 `peft` 库给 GPT-2 加上 LoRA，并**动手训练几步**

> 前面我们只会"用"模型，从这章开始学"改"模型——而且是用很省资源的方式改。

```python
# ============================================================
# 环境准备：先运行这一格安装本教程所需依赖
# Colab 每新开一个 notebook 都是全新环境，建议先跑这一格
# 已装过的库会自动跳过，不会重复下载
# ============================================================
# 依赖安装（在终端执行下面这行）：pip install -q transformers datasets trl peft accelerate matplotlib
```

## 1. 全量微调的问题

微调 = 拿预训练模型，用我们自己的数据再训练一下。但如果**所有参数都训练**：

| 问题 | 说明 |
| --- | --- |
| 显存爆炸 | GPT-2 有 1.2 亿参数，7B 模型更是几十 GB |
| 存储翻倍 | 每份微调结果都要存一个完整模型（几个 GB） |
| 容易"灾难性遗忘" | 改太多，模型把原来的本事忘了 |

## 2. LoRA 的核心思想（一句话）

**冻结原模型的所有参数，只在旁边加一小对"低秩矩阵"来学习改动。**

数学上：原来权重是 `W`（很大，不动），我们学习增量 `ΔW = B × A`，其中 A、B 是**很小的矩阵**。
最终该层的输出用 `W + B×A`。因为 A、B 很小，可训练参数可能只有**原来的 0.1%~0.5%**，显存和存储都大幅下降。

> 类比：不改发动机本体，只外挂一个小涡轮，就能调整动力。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载基座模型（base model）——注意：这一步不加 LoRA
base_model = AutoModelForCausalLM.from_pretrained("gpt2")

# 统计可训练参数总数
total = sum(p.numel() for p in base_model.parameters())
print(f"基座模型参数量：{total:,}（约 {total/1e6:.1f} 百万）")
```

## 3. 给模型加 LoRA

`peft` 库提供了现成的 `LoraConfig`。几个关键参数：

| 参数 | 含义 | 小白理解 |
| --- | --- | --- |
| `r` | 低秩矩阵的"秩" | 越大能学越多，但参数也越多。常用 8/16 |
| `lora_alpha` | 缩放系数 | 一般设为 `r` 的 1~2 倍 |
| `target_modules` | 给哪些层加 LoRA | GPT-2 的注意力层叫 `c_attn` |
| `lora_dropout` | 随机丢弃，防过拟合 | 常用 0.05 |

```python
from peft import LoraConfig, get_peft_model, TaskType

# 1) 配置 LoRA
lora_config = LoraConfig(
    r=8,                          # 秩
    lora_alpha=16,                # 缩放
    target_modules=["c_attn"],    # GPT-2 的注意力投影层名（其他模型不同，见下）
    lora_dropout=0.05,
    bias="none",                  # 不训练 bias
    task_type=TaskType.CAUSAL_LM, # 任务是因果语言模型
)

# 2) 把 LoRA 套到基座模型上，得到 PEFT 模型
model = get_peft_model(base_model, lora_config)

# 3) 打印可训练参数占比——关键看这个对比！
model.print_trainable_parameters()
```

> 🎯 **注意上面输出的比例**：可训练参数只有零点几个百分点，但我们要的效果就是靠这一小撮参数实现的。
>
> 📌 `target_modules` 因模型而异：GPT-2 是 `c_attn`；LLaMA 常用 `q_proj, v_proj`；BERT 用 `query, value`。不确定时可以传字符串列表去试，报错会提示可用名字。

```python
# 看看模型结构：带 lora 的层会显示 LoraConfig 相关信息
print(model)

# 只有名字里带 "lora" 的参数才需要训练
trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
print("\n前 8 个可训练参数名：")
for n in trainable_names[:8]:
    print(" ", n)
```

## 4. 亲手训练几步（CPU 也能跑！）

我们用几句话当作训练数据，跑一个迷你的"续写微调"。重点是跑通**流程**，不是效果。

```python
from torch.utils.data import DataLoader

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token    # GPT-2 没有 pad，用 eos 代替（固定套路）

# 几条"训练语料"（真实场景里换成你的数据）
texts = [
    "机器学习是人工智能的一个分支。",
    "PyTorch 是一个深度学习框架。",
    "LoRA 是一种高效微调方法。",
    "微调可以让大模型适应特定任务。",
] * 4   # 复制几份，凑够训练量

# 编码成模型能吃的数字，并统一长度
batch = tokenizer(
    texts,
    padding="max_length",
    truncation=True,
    max_length=32,
    return_tensors="pt",
)


class TinyDataset(torch.utils.data.Dataset):
    """把编码结果包装成 Dataset（复习一下第 3 章）。"""

    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return self.encodings["input_ids"].size(0)

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.encodings.items()}


dataset = TinyDataset(batch)
loader = DataLoader(dataset, batch_size=2, shuffle=True)
print("训练样本数：", len(dataset))
```

```python
# ============================================================
# 训练循环（还是那五步，只是模型变成了 PEFT 模型）
# ============================================================
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),  # 只优化 LoRA 参数
    lr=1e-3,
)

model.train()
epochs = 3
for epoch in range(epochs):
    for batch_data in loader:
        # 语言模型的标签就是输入本身（自己预测下一个词）
        outputs = model(**batch_data, labels=batch_data["input_ids"])
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"第 {epoch+1} 轮  损失 = {loss.item():.4f}")
```

## 5. 保存 LoRA 权重（体积超小！）

LoRA 的好处之一：**只保存那一点点增量参数**，通常只有几十 MB 甚至几 MB。

```python
# ---------- 保存 LoRA adapter ----------
model.save_pretrained("my_lora_adapter")
print("LoRA 权重已保存到 my_lora_adapter/ 目录（只有一小撮参数）")

# ---------- 加载：先加载基座模型，再把 LoRA 贴上去 ----------
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("gpt2")
loaded = PeftModel.from_pretrained(base, "my_lora_adapter")
print("LoRA 加载成功！")
```

```python
# ---------- （可选）把 LoRA 合并回基座模型 ----------
# 合并后就是一个普通模型，推理时不再需要 peft，方便部署
merged = loaded.merge_and_unload()
print("已合并。此时模型结构和原始 GPT-2 完全一样。")
```

## 6. 小练习

**练习**：把 `r=8` 改成 `r=1` 和 `r=32`，分别运行第 3 节，观察"可训练参数占比"如何变化。

**思考**：`r` 越大越好吗？（提示：参数越多越容易过拟合，且训练更慢；LoRA 的价值就在于"够用又省"。）

---

### 本章小结
- 全量微调贵，**PEFT** 只训练极少参数。
- **LoRA** = 冻结原权重 + 在旁边学一对低秩矩阵。
- `LoraConfig` → `get_peft_model` 两步搞定，`print_trainable_parameters()` 看省了多少。
- LoRA 权重**体积极小**，可单独保存，也能 `merge_and_unload()` 合并回基座。

👉 **下一章：** 用 **TRL 的 SFTTrainer**，把"准备数据 + 训练 + 保存"全流程一键完成。
