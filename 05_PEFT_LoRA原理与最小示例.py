# -*- coding: utf-8 -*-
"""
05 · PEFT · LoRA原理与最小示例
============================================================
运行方式： python 05_PEFT_LoRA原理与最小示例.py
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
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载基座模型（base model）——注意：这一步不加 LoRA
base_model = AutoModelForCausalLM.from_pretrained("gpt2")

# 统计可训练参数总数
total = sum(p.numel() for p in base_model.parameters())
print(f"基座模型参数量：{total:,}（约 {total/1e6:.1f} 百万）")


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


# 看看模型结构：带 lora 的层会显示 LoraConfig 相关信息
print(model)

# 只有名字里带 "lora" 的参数才需要训练
trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
print("\n前 8 个可训练参数名：")
for n in trainable_names[:8]:
    print(" ", n)


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


# ---------- 保存 LoRA adapter ----------
model.save_pretrained("my_lora_adapter")
print("LoRA 权重已保存到 my_lora_adapter/ 目录（只有一小撮参数）")

# ---------- 加载：先加载基座模型，再把 LoRA 贴上去 ----------
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("gpt2")
loaded = PeftModel.from_pretrained(base, "my_lora_adapter")
print("LoRA 加载成功！")


# ---------- （可选）把 LoRA 合并回基座模型 ----------
# 合并后就是一个普通模型，推理时不再需要 peft，方便部署
merged = loaded.merge_and_unload()
print("已合并。此时模型结构和原始 GPT-2 完全一样。")
