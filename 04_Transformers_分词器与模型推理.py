# -*- coding: utf-8 -*-
"""
04 · Transformers · 分词器与模型推理
============================================================
运行方式： python 04_Transformers_分词器与模型推理.py
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


# 批量编码：一次性处理多句话，自动 padding 对齐
batch = tokenizer(
    ["I love AI", "Deep learning is fun"],
    padding=True,          # 短句补齐到最长
    truncation=True,       # 超长则截断
    return_tensors="pt",   # 返回 PyTorch 张量
)

print("input_ids（数字）：\n", batch["input_ids"])
print("attention_mask（哪些是真词）：\n", batch["attention_mask"])


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


from transformers import pipeline

# pipeline 把"加载模型 + 预处理 + 推理 + 后处理"打包成一行
sentiment = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",  # 已训练好的情感模型
)

results = sentiment(["I love this product!", "This is the worst movie ever."])
for r in results:
    print(r)
