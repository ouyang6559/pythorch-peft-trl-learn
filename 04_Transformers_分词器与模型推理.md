# 04 · Transformers · 分词器与模型推理

## 本章目标
- 认识 Hugging Face `transformers` 库
- 搞懂**分词器（Tokenizer）**：文字怎么变成数字
- 加载一个**预训练模型**并让它做推理（生成 / 情感分类）

> 从这章开始，我们不再自己写模型结构，而是**直接用别人训练好的大模型**（这叫"预训练模型"）。

```python
# ============================================================
# 环境准备：先运行这一格安装本教程所需依赖
# Colab 每新开一个 notebook 都是全新环境，建议先跑这一格
# 已装过的库会自动跳过，不会重复下载
# ============================================================
# 依赖安装（在终端执行下面这行）：pip install -q transformers datasets trl peft accelerate matplotlib
```

## 1. 什么是"预训练模型"？

大模型（如 GPT、BERT）已经在海量文本上学习过语言规律。我们通常不去从头训练，而是：

1. **直接拿来用**（推理）——本章做的事；
2. **稍微改一改再用**（微调）——第 5~7 章做的事。

核心库叫 `transformers`，它帮我们**一行代码下载并加载**模型。

## 2. 模型为什么看不懂中文/英文？——需要 Tokenizer

模型内部只认数字。**分词器负责：文字 ↔ 数字（token id）互相转换。**

流程：`"I love AI"` → 分词 → `["I", " love", " AI"]` → 查表 → `[40, 1842, 9552]` → 喂给模型

```python
from transformers import AutoTokenizer

# 从 Hugging Face 加载 GPT-2 的分词器（第一次运行会自动下载，需要联网）
tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "I love AI"

# 1) 文字 → token (子词)
tokens = tokenizer.tokenize(text)
print("分词结果：", tokens)

# 2) token → 数字 id
ids = tokenizer.encode(text)
print("对应 id ：", ids)

# 3) 数字 id → 还原成文字（解码）
print("解码回去：", tokenizer.decode(ids))
```

### `attention_mask` 是什么？
批量处理时句子长短不一，短的要补 `padding`（填充符）。`attention_mask` 用 0/1 告诉模型**哪些是真词、哪些是填充**，0 的部分会被忽略。

```python
# 批量编码：一次性处理多句话，自动 padding 对齐
batch = tokenizer(
    ["I love AI", "Deep learning is fun"],
    padding=True,          # 短句补齐到最长
    truncation=True,       # 超长则截断
    return_tensors="pt",   # 返回 PyTorch 张量
)

print("input_ids（数字）：\n", batch["input_ids"])
print("attention_mask（哪些是真词）：\n", batch["attention_mask"])
```

## 3. 加载模型做「文本生成」

用 `AutoModelForCausalLM` 加载一个"因果语言模型"（就是 GPT 这类，擅长续写）。然后调用 `generate` 生成文本。

```python
from transformers import AutoModelForCausalLM
import torch

# 加载 GPT-2（约 500MB，第一次下载稍慢）
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()   # 推理模式

# 把提示词编码
prompt = "Artificial intelligence will"
inputs = tokenizer(prompt, return_tensors="pt")

# 生成：不给梯度，省内存
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=30,      # 最多再生成 30 个 token
        do_sample=True,         # 采样（更有多样性）
        top_k=50,               # 只从概率最高的 50 个词里选
        temperature=0.9,        # 温度：越大越随机
        pad_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

> 💡 `generate` 里的采样参数就像"写作风格旋钮"：`temperature` 越小越保守、越大越天马行空。

## 4. 换个任务：文本情感分类

同一个库，换个模型就能做完全不同的任务。这里用现成的**情感分析 pipeline**，几行搞定。

```python
from transformers import pipeline

# pipeline 把"加载模型 + 预处理 + 推理 + 后处理"打包成一行
sentiment = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",  # 已训练好的情感模型
)

results = sentiment(["I love this product!", "This is the worst movie ever."])
for r in results:
    print(r)
```

## 5. 小练习

**练习**：把第 3 节 `generate` 里的 `max_new_tokens` 改成 50，`temperature` 改成 0.3 和 1.5 各跑一次，感受生成结果的差异。

**思考**：为什么 `temperature` 很高的时候，输出容易变成"胡言乱语"？

---

### 本章小结
- `transformers` 让你**一行加载预训练模型**，不用自己搭网络。
- **Tokenizer** 负责文字 ↔ 数字，`attention_mask` 标记真实 token。
- `AutoModelForCausalLM` 做生成，`pipeline` 把常见任务打包成一行。
- 模型可以直接用（推理），也可以继续微调（下一章）。

👉 **下一章：** 认识 **PEFT / LoRA**——只用 1% 的参数就能微调大模型。
