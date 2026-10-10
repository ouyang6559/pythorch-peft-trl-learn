# PyTorch + PEFT + TRL 入门系列教程（本地版）

> 从 PyTorch 基础到大模型微调的完整路径。**全程中文注释、每一步都有讲解，无 GPU 也能跑。**

## 每章包含两个文件

| 文件 | 用途 |
| --- | --- |
| `XX_*.md` | **讲义**：讲解 + 代码，直接用 Markdown 阅读器打开即可 |
| `XX_*.py` | **可运行代码**：中文注释，直接 `python XX_*.py` 就能跑 |

## 目录

| 章节 | 讲义 | 代码 |
| --- | --- | --- |
| 00 学习路线与环境准备 | [md](00_学习路线与环境准备.md) | [py](00_学习路线与环境准备.py) |
| 01 PyTorch 张量与自动求导 | [md](01_PyTorch基础_张量与自动求导.md) | [py](01_PyTorch基础_张量与自动求导.py) |
| 02 模型与训练循环 | [md](02_PyTorch基础_模型与训练循环.md) | [py](02_PyTorch基础_模型与训练循环.py) |
| 03 数据加载与小项目 | [md](03_PyTorch基础_数据加载与小项目.md) | [py](03_PyTorch基础_数据加载与小项目.py) |
| 04 Transformers 分词器与推理 | [md](04_Transformers_分词器与模型推理.md) | [py](04_Transformers_分词器与模型推理.py) |
| 05 PEFT / LoRA | [md](05_PEFT_LoRA原理与最小示例.md) | [py](05_PEFT_LoRA原理与最小示例.py) |
| 06 TRL · SFT 监督微调 | [md](06_TRL_SFT监督微调.md) | [py](06_TRL_SFT监督微调.py) |
| 07 TRL · DPO 偏好优化与导出 | [md](07_TRL_DPO偏好优化与导出.md) | [py](07_TRL_DPO偏好优化与导出.py) |

## 怎么用

```bash
# 1. 安装依赖（建议 Python 3.10+）
pip install -r requirements.txt

# 2. 按顺序运行任意一章的代码
python 01_PyTorch基础_张量与自动求导.py
```

- 建议按 00 → 07 顺序学习：先读 `.md` 理解，再跑 `.py` 实践。
- 第 04 章起会联网下载模型（首次较慢），无 GPU 也能跑，只是慢一些。
- 含绘图（matplotlib）的章节若无图形界面，可把 `plt.show()` 换成 `plt.savefig('fig.png')`。
- 遇到 API 报错多半是库版本差异，参见第 06 章「版本适配小贴士」。
