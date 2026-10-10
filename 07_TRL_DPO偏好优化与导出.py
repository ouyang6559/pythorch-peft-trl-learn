# -*- coding: utf-8 -*-
"""
07 · TRL · DPO偏好优化与导出
============================================================
运行方式： python 07_TRL_DPO偏好优化与导出.py
说明：
  - 首次运行会联网下载模型/依赖，请保持网络畅通。
  - 含 matplotlib 的章节若无图形界面，可把 plt.show() 换成 plt.savefig('fig.png')。
  - 建议先 'pip install -r requirements.txt'。
"""

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


trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=dpo_dataset,
    peft_config=peft_config,       # 使用 LoRA
    processing_class=tokenizer,    # 旧版 TRL 改成 tokenizer=tokenizer
)

print("开始 DPO 训练……")
trainer.train()


# 训练器里的模型是 PEFT 模型，先合并
merged_model = trainer.model.merge_and_unload()

save_dir = "./dpo_merged_model"
merged_model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"完整模型已导出到 {save_dir}（可像普通 GPT-2 一样加载）")


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
