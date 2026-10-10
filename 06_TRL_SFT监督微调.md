# 06 · TRL · SFT监督微调

## 本章目标
- 认识 **TRL**（Transformer Reinforcement Learning）库
- 理解 **SFT（Supervised Fine-Tuning，监督微调）**
- 用 `SFTTrainer` **一键微调**模型（结合第 5 章的 LoRA）

> 上一章我们手写了训练循环。这一章用 TRL，把这些繁琐步骤**打包成几行代码**。

## 1. SFT 是什么？

**SFT = 用"问题 → 标准答案"的数据，教模型学会按我们想要的方式回答。**

比如给它一堆：

```
指令：把下面这句话翻译成英文
回答：Translate the following sentence into English
```

训练后，模型就更擅长按指令做事。这是大模型"对齐"的第一步。

## 2. TRL 是什么？

TRL 建立在 `transformers` 和 `peft` 之上，提供**现成的 Trainer**：

| Trainer | 用途 |
| --- | --- |
| `SFTTrainer` | 监督微调（本章） |
| `DPOTrainer` | 偏好对齐（下一章） |
| `RewardTrainer` 等 | 奖励建模等 |

`Trainer` 帮你自动处理：分批、梯度累积、日志、保存、混合精度……你只需给数据和参数。

## 3. 准备数据

最朴素的 SFT 数据就是一列文本（`text` 字段）。真实项目常用"指令 + 回复"拼成一段文本。

```python
from datasets import Dataset

# 用几条中文样例演示（真实场景可以几百上千条）
data = {
    "text": [
        "### 指令：什么是机器学习？\n### 回答：机器学习是让计算机从数据中自动学习规律的方法。",
        "### 指令：什么是深度学习？\n### 回答：深度学习是使用多层神经网络进行学习的方法。",
        "### 指令：PyTorch 是什么？\n### 回答：PyTorch 是一个开源的深度学习框架。",
        "### 指令：什么是 LoRA？\n### 回答：LoRA 是一种只训练少量参数的高效微调方法。",
        "### 指令：为什么要微调？\n### 回答：微调能让模型更好地完成特定领域的任务。",
    ]
}

# 转成 Hugging Face 的 Dataset 格式（TRL 需要）
dataset = Dataset.from_dict(data)
print(dataset)
print("\n第一条第数据：\n", dataset[0]["text"])
```

## 4. 加载模型 + 套上 LoRA

和上一章一样，我们微调 LoRA 而不是整个模型（省显存）。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig

model_name = "gpt2"   # 想更快可换成更小的模型

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

# LoRA 配置（复习第 5 章）
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
print("模型与 LoRA 配置就绪")
```

## 5. 配置训练参数并开跑

`SFTConfig` 里常用的参数，小白先记住这几个：

| 参数 | 含义 |
| --- | --- |
| `output_dir` | 结果保存目录 |
| `num_train_epochs` | 训练几轮 |
| `per_device_train_batch_size` | 每次喂几条 |
| `learning_rate` | 学习率 |
| `max_steps` | 最多训练多少步（设了就忽略 epoch 数，方便快速演示） |
| `logging_steps` | 每几步打印一次日志 |
| `report_to` | 是否上报到 wandb 等，填 `"none"` 保持干净 |

```python
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./sft_output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="no",        # 演示就不中途保存了
    report_to="none",          # 不上报
    max_length=128,            # 单条文本最大长度
    dataset_text_field="text", # 用哪一列作为训练文本
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,        # 直接传 LoRA 配置，Trainer 会自动套上
    processing_class=tokenizer,     # 新版 TRL 用这个名字；旧版(<0.12)改成 tokenizer=tokenizer
)

print("Trainer 就绪，开始训练……")
trainer.train()
```

> ⏳ **CPU 用户**：GPT-2 在 CPU 上训这几条数据大约需要几十秒到一两分钟，属正常。
> **Colab GPU 用户**：会非常快。若想更快出结果，可把 `max_steps=5` 加进 `SFTConfig`。

## 6. 用微调后的模型生成

训练完，直接调用 `trainer.model` 生成看看效果（数据太少，效果有限，重点看流程）。

```python
import torch

trainer.model.eval()
prompt = "### 指令：什么是 LoRA？\n### 回答："
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    out = trainer.model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False,       # 贪心解码，输出更稳定
        pad_token_id=tokenizer.eos_token_id,
    )
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

```python
# ---------- 保存微调结果 ----------
trainer.save_model("./sft_final")     # 保存（含 LoRA 权重）
tokenizer.save_pretrained("./sft_final")
print("已保存到 ./sft_final")
```

## 7. 版本适配小贴士

不同版本 TRL 参数名有差异，遇到报错对照下面改即可：

| 新版（建议） | 旧版（trl < 0.12） |
| --- | --- |
| `SFTConfig(...)` | `TrainingArguments(...)` |
| `processing_class=tokenizer` | `tokenizer=tokenizer` |
| `max_length` | `max_seq_length` |
| `peft_config=` 传给 SFTTrainer | 需先自己 `get_peft_model` 再传入 `model=` |

> 建议保持 `transformers`、`trl`、`peft` 都升级到较新版本，避免踩坑：`%pip install -U transformers trl peft`。

---

### 本章小结
- **SFT** 用"指令→回答"数据教模型按指令办事，是对齐的第一步。
- `SFTTrainer` 把训练流程打包，几行代码完成微调。
- 配合 `peft_config` 即可**用 LoRA 微调**，省显存。
- 训练完用 `trainer.model.generate` 推理，`trainer.save_model` 保存。

👉 **下一章：** 进入 **DPO 偏好优化**——不只是"答对"，还要"答得好"。
