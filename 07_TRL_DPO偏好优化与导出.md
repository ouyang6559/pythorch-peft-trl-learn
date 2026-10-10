# 07 · TRL · DPO偏好优化与导出

## 本章目标
- 理解 **DPO（Direct Preference Optimization，直接偏好优化）**
- 用 `DPOTrainer` 让模型学会"哪个回答更好"
- 学会**合并 LoRA 并导出**、了解**如何评估**效果

> SFT 教模型"照着答"，DPO 教模型"答得更符合人类偏好"。这是对齐流程的关键一步。

## 1. 为什么还需要 DPO？

SFT 之后，模型能答题了，但**质量参差不齐**——同一个问题，它可能给出好答案，也可能给出差答案。

我们希望：**让模型倾向于生成"更好"的那类回答。** 做法是给它成对的数据：

```
问题 prompt
好的回答 chosen   ← 希望模型多生成这种
差的回答 rejected ← 希望模型少生成这种
```

DPO 直接用这种"好坏对比"数据优化模型，**不需要单独训练奖励模型**，比传统 RLHF 简单很多。

## 2. 准备偏好数据

数据需要三列：`prompt`（问题）、`chosen`（好答案）、`rejected`（差答案）。

```python
from datasets import Dataset

# 构造少量偏好数据（真实项目通常几千条起）
preference_data = {
    "prompt": [
        "什么是机器学习？",
        "什么是 LoRA？",
        "为什么要微调模型？",
    ],
    "chosen": [
        "机器学习是让计算机从数据中自动学习规律、并用于预测或决策的方法。",
        "LoRA 是一种高效微调技术，通过训练极少量低秩参数来调整模型。",
        "微调能让预训练模型适应特定领域任务，提升在该任务上的表现。",
    ],
    "rejected": [
        "不知道。",
        "就是一个东西。",
        "因为好玩。",
    ],
}

dpo_dataset = Dataset.from_dict(preference_data)
print(dpo_dataset)
```

## 3. 加载模型并配置 DPO

`DPOTrainer` 需要一个**参考模型（reference model）**做对比。用 PEFT 时，TRL 会自动帮你处理参考模型，无需手动加载，很省心。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

dpo_config = DPOConfig(
    output_dir="./dpo_output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=1e-4,     # DPO 学习率通常比 SFT 小
    max_steps=10,           # 演示只跑 10 步
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    beta=0.1,               # 控制"偏离参考模型"的力度，常用 0.1~0.5
)

print("DPO 配置就绪")
```

```python
trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=dpo_dataset,
    peft_config=peft_config,       # 使用 LoRA
    processing_class=tokenizer,    # 旧版 TRL 改成 tokenizer=tokenizer
)

print("开始 DPO 训练……")
trainer.train()
```

> ⏳ CPU 上跑 10 步 DPO 需要一两分钟属正常。DPO 每一步要算 4 次前向（chosen/rejected 各两次），确实比 SFT 慢。
>
> 训练日志里有个 `rewards/accuracies`，表示"模型认为 chosen 比 rejected 好的比例"，这个值**越高越好**（理想接近 1.0）。

## 4. 导出：合并 LoRA 并保存成完整模型

部署时，我们常把 LoRA 合并回基座，得到一个"普通"模型文件，便于用任何推理框架加载。

```python
# 训练器里的模型是 PEFT 模型，先合并
merged_model = trainer.model.merge_and_unload()

save_dir = "./dpo_merged_model"
merged_model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"完整模型已导出到 {save_dir}（可像普通 GPT-2 一样加载）")
```

## 5. 评估：怎么知道微调有没有用？

微调不是"跑完就完事"，一定要评估。常见做法：

| 方法 | 说明 | 小白推荐度 |
| --- | --- | --- |
| **人工看** | 手写几个问题，肉眼对比微调前后 | ⭐⭐⭐ 最直观 |
| 困惑度 Perplexity | 模型对文本的"惊讶程度"，越低越好 | ⭐⭐ |
| 任务准确率 | 若有标准答案，算对/错 | ⭐⭐⭐ |
| 偏好胜率 Win Rate | 用另一个模型/人评判新模型是否更优 | ⭐⭐ |

### 动手：对比微调前后的输出

```python
import torch


def ask(model, question, max_new_tokens=40):
    """给模型一个问题，返回它的回答（贪心解码，保证可比）。"""
    model.eval()
    prompt = f"### 指令：{question}\n### 回答："
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    # 只截取"回答："之后的部分
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text.split("### 回答：")[-1].strip()


# 微调后的模型
print("【微调后】", ask(merged_model, "什么是机器学习？"))

# 原始基座模型（对比用）
original = AutoModelForCausalLM.from_pretrained("gpt2")
print("【原始模型】", ask(original, "什么是机器学习？"))
```

> 🔍 我们只用了 5~10 条数据、训练几十步，所以效果**不会**惊艳——这是正常的。
> 真实项目里，数据量要达到**几千到几万条**，训练更久，效果才会明显。
> **本章的价值在于：你完整走通了一遍工业级微调流程。**

## 6. 常见问题排查

| 现象 | 可能原因 | 对策 |
| --- | --- | --- |
| 显存不足（OOM） | batch 太大 / 模型太大 | 调小 `per_device_train_batch_size`，开 `gradient_accumulation_steps` |
| 损失不下降 | 学习率不合适 / 数据太少 | 调学习率，增加数据 |
| 输出全是重复 | 训练过度 / 采样参数差 | 减少步数，调 `temperature`、`repetition_penalty` |
| 报错参数名不认识 | 库版本不匹配 | 见第 6 章"版本适配"，升级或改名 |
| 找不到 `c_attn` | 换模型后层名变了 | 换成该模型的注意力层名（LLaMA 用 `q_proj,v_proj`） |

## 7. 下一步可以去哪？

- 📚 **更多数据**：用真实数据集（如 Alpaca、中文指令集）替换本教程的玩具数据。
- 🚀 **真实模型**：把 `gpt2` 换成 `Qwen2.5-0.5B`、`Llama-3.2-1B` 等支持中文的小模型（需 GPU）。
- 🎛️ **更多算法**：TRL 还提供 PPO、GRPO、Reward Modeling，可继续深入。
- 🖥️ **部署**：合并后的模型可用 `transformers`、vLLM、Ollama 等加载上线。
- 📖 **官方文档**：Hugging Face 的 TRL / PEFT / Transformers 文档是最权威的学习资料。

---

## 🎓 全系列回顾

| 章节 | 一句话记住它 |
| --- | --- |
| 0 | 环境准备 + 学习地图 |
| 1 | 张量是数据，autograd 自动算梯度 |
| 2 | 训练循环五步：前向→损失→清梯度→反向→更新 |
| 3 | Dataset 取一条，DataLoader 批量喂 |
| 4 | Transformers 一行加载模型，Tokenizer 做文字↔数字 |
| 5 | PEFT/LoRA：只训练 1% 参数 |
| 6 | TRL/SFT：几行代码完成监督微调 |
| 7 | TRL/DPO：让模型学会"答得更好" |

**恭喜！你已经走完了从 PyTorch 基础到大模型微调的完整路径。** 🎉
接下来，挑一个你感兴趣的任务，用真实数据跑一遍吧！
