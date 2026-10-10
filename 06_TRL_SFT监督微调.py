# -*- coding: utf-8 -*-
"""
06 · TRL · SFT监督微调
============================================================
运行方式： python 06_TRL_SFT监督微调.py
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


# ---------- 保存微调结果 ----------
trainer.save_model("./sft_final")     # 保存（含 LoRA 权重）
tokenizer.save_pretrained("./sft_final")
print("已保存到 ./sft_final")
