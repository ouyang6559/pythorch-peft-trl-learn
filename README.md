# PyTorch + PEFT + TRL 入门系列教程

> 从 PyTorch 基础到大模型微调的完整路径。**全程中文注释、每一步都有讲解，无 GPU 也能跑。**
> 学习顺序：PyTorch 基础 → PEFT/LoRA → TRL 微调（SFT / DPO）。

## 内容形式

| 文件 | 用途 |
| --- | --- |
| `XX_*.md` | **讲义**：讲解 + 代码，Markdown 阅读器直接打开 |
| `XX_*.py` | **可运行 demo**：中文注释，直接 `python XX_*.py` 就能跑 |
| `notebooks/XX_*.ipynb` | **Notebook 版**：可本地 Jupyter 打开，或点下方徽章在 Colab 一键运行 |

## 目录

| 章节 | 讲义 | 代码 | 在线运行（Colab） |
| --- | --- | --- | --- |
| 00 · 学习路线与环境准备<br><sub>学习地图 + 环境自检 + 一键安装，先跑这章</sub> | [md](https://github.com/ouyang6559/pythorch-peft-trl-learn/blob/main/00_%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF%E4%B8%8E%E7%8E%AF%E5%A2%83%E5%87%86%E5%A4%87.md) | [py](https://github.com/ouyang6559/pythorch-peft-trl-learn/blob/main/00_%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF%E4%B8%8E%E7%8E%AF%E5%A2%83%E5%87%86%E5%A4%87.py) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ouyang6559/pythorch-peft-trl-learn/blob/main/notebooks/00_%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF%E4%B8%8E%E7%8E%AF%E5%A2%83%E5%87%86%E5%A4%87.ipynb) |
| 01 · PyTorch基础 · 张量与自动求导<br><sub>张量、形状、广播、autograd 自动求导</sub> | [md](https://github.com/ouyang6559/pythorch-peft-trl-learn/blob/main/01_PyTorch%E5%9F%BA%E7%A1%80_%E5%BC%A0%E9%87%8F%E4%B8%8E%E8%87%AA%E5%8A%A8%E6%B1%82%E5%AF%BC.md) | [py](https://github.com/ouyang6559/pythorch-peft-trl-learn/blob/main/01_PyTorch%E5%9F%BA%E7%A1%80_%E5%BC%A0%E9%87%8F%E4%B8%8E%E8%87%AA%E5%8A%A8%E6%B1%82%E5%AF%BC.py) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ouyang6559/pythorch-peft-trl-learn/blob/main/notebooks/01_PyTorch%E5%9F%BA%E7%A1%80_%E5%BC%A0%E9%87%8F%E4%B8%8E%E8%87%AA%E5%8A%A8%E6%B1%82%E5%AF%BC.ipynb) |
| 02 · PyTorch基础 · 模型与训练循环<br><sub>nn.Module、损失、优化器、训练循环五步</sub> | [md](https://github.com/ouyang6559/pythorch-peft-trl-learn/blob/main/02_PyTorch%E5%9F%BA%E7%A1%80_%E6%A8%A1%E5%9E%8B%E4%B8%8E%E8%AE%AD%E7%BB%83%E5%BE%AA%E7%8E%AF.md) | [py](https://github.com/ouyang6559/pythorch-peft-trl-learn/blob/main/02_PyTorch%E5%9F%BA%E7%A1%80_%E6%A8%A1%E5%9E%8B%E4%B8%8E%E8%AE%AD%E7%BB%83%E5%BE%AA%E7%8E%AF.py) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ouyang6559/pythorch-peft-trl-learn/blob/main/notebooks/02_PyTorch%E5%9F%BA%E7%A1%80_%E6%A8%A1%E5%9E%8B%E4%B8%8E%E8%AE%AD%E7%BB%83%E5%BE%AA%E7%8E%AF.ipynb) |
| 03 · PyTorch基础 · 数据加载与小项目<br><sub>Dataset/DataLoader、二分类小项目、设备管理</sub> | [md](https://github.com/ouyang6559/pythorch-peft-trl-learn/blob/main/03_PyTorch%E5%9F%BA%E7%A1%80_%E6%95%B0%E6%8D%AE%E5%8A%A0%E8%BD%BD%E4%B8%8E%E5%B0%8F%E9%A1%B9%E7%9B%AE.md) | [py](https://github.com/ouyang6559/pythorch-peft-trl-learn/blob/main/03_PyTorch%E5%9F%BA%E7%A1%80_%E6%95%B0%E6%8D%AE%E5%8A%A0%E8%BD%BD%E4%B8%8E%E5%B0%8F%E9%A1%B9%E7%9B%AE.py) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ouyang6559/pythorch-peft-trl-learn/blob/main/notebooks/03_PyTorch%E5%9F%BA%E7%A1%80_%E6%95%B0%E6%8D%AE%E5%8A%A0%E8%BD%BD%E4%B8%8E%E5%B0%8F%E9%A1%B9%E7%9B%AE.ipynb) |
| 04 · Transformers · 分词器与模型推理<br><sub>Tokenizer、GPT-2 生成、情感分析 pipeline</sub> | [md](https://github.com/ouyang6559/pythorch-peft-trl-learn/blob/main/04_Transformers_%E5%88%86%E8%AF%8D%E5%99%A8%E4%B8%8E%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86.md) | [py](https://github.com/ouyang6559/pythorch-peft-trl-learn/blob/main/04_Transformers_%E5%88%86%E8%AF%8D%E5%99%A8%E4%B8%8E%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86.py) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ouyang6559/pythorch-peft-trl-learn/blob/main/notebooks/04_Transformers_%E5%88%86%E8%AF%8D%E5%99%A8%E4%B8%8E%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86.ipynb) |
| 05 · PEFT · LoRA原理与最小示例<br><sub>PEFT/LoRA 原理、只训练 1% 参数、保存与合并</sub> | [md](https://github.com/ouyang6559/pythorch-peft-trl-learn/blob/main/05_PEFT_LoRA%E5%8E%9F%E7%90%86%E4%B8%8E%E6%9C%80%E5%B0%8F%E7%A4%BA%E4%BE%8B.md) | [py](https://github.com/ouyang6559/pythorch-peft-trl-learn/blob/main/05_PEFT_LoRA%E5%8E%9F%E7%90%86%E4%B8%8E%E6%9C%80%E5%B0%8F%E7%A4%BA%E4%BE%8B.py) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ouyang6559/pythorch-peft-trl-learn/blob/main/notebooks/05_PEFT_LoRA%E5%8E%9F%E7%90%86%E4%B8%8E%E6%9C%80%E5%B0%8F%E7%A4%BA%E4%BE%8B.ipynb) |
| 06 · TRL · SFT监督微调<br><sub>SFTTrainer 一键监督微调、版本适配说明</sub> | [md](https://github.com/ouyang6559/pythorch-peft-trl-learn/blob/main/06_TRL_SFT%E7%9B%91%E7%9D%A3%E5%BE%AE%E8%B0%83.md) | [py](https://github.com/ouyang6559/pythorch-peft-trl-learn/blob/main/06_TRL_SFT%E7%9B%91%E7%9D%A3%E5%BE%AE%E8%B0%83.py) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ouyang6559/pythorch-peft-trl-learn/blob/main/notebooks/06_TRL_SFT%E7%9B%91%E7%9D%A3%E5%BE%AE%E8%B0%83.ipynb) |
| 07 · TRL · DPO偏好优化与导出<br><sub>DPOTrainer 偏好优化、合并导出、评估与排错</sub> | [md](https://github.com/ouyang6559/pythorch-peft-trl-learn/blob/main/07_TRL_DPO%E5%81%8F%E5%A5%BD%E4%BC%98%E5%8C%96%E4%B8%8E%E5%AF%BC%E5%87%BA.md) | [py](https://github.com/ouyang6559/pythorch-peft-trl-learn/blob/main/07_TRL_DPO%E5%81%8F%E5%A5%BD%E4%BC%98%E5%8C%96%E4%B8%8E%E5%AF%BC%E5%87%BA.py) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ouyang6559/pythorch-peft-trl-learn/blob/main/notebooks/07_TRL_DPO%E5%81%8F%E5%A5%BD%E4%BC%98%E5%8C%96%E4%B8%8E%E5%AF%BC%E5%87%BA.ipynb) |

## 怎么用

```bash
# 1. 安装依赖（建议 Python 3.10+）
pip install -r requirements.txt

# 2. 按顺序运行任意一章的 demo
python 01_PyTorch基础_张量与自动求导.py
```

- 建议按 00 → 07 顺序学习：先读 `.md` 理解，再跑 `.py` 实践。
- 想在线跑：直接点目录里的 **Open in Colab** 徽章（首次会自动装依赖、联网下模型）。
- 第 04 章起会联网下载模型（首次较慢），无 GPU 也能跑，只是慢一些。
- 含绘图（matplotlib）的章节若无图形界面，可把 `plt.show()` 换成 `plt.savefig('fig.png')`。
- 遇到 API 报错多半是库版本差异，参见第 06 章「版本适配小贴士」。
