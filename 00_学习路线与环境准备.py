# -*- coding: utf-8 -*-
"""
00 · 学习路线与环境准备
============================================================
运行方式： python 00_学习路线与环境准备.py
说明：
  - 首次运行会联网下载模型/依赖，请保持网络畅通。
  - 含 matplotlib 的章节若无图形界面，可把 plt.show() 换成 plt.savefig('fig.png')。
  - 建议先 'pip install -r requirements.txt'。
"""

# ============================================================
# 环境自检：检查各个库是否已安装、版本是多少
# ============================================================
import sys
print("Python 版本：", sys.version.split()[0])
print("-" * 40)


def show_version(pkg_name, import_name=None):
    """尝试导入一个库并打印版本；导入失败就提示需要安装。"""
    import_name = import_name or pkg_name
    try:
        mod = __import__(import_name)
        ver = getattr(mod, "__version__", "未知版本")
        print(f"[OK]   {pkg_name:<14} 版本 {ver}")
    except Exception:
        print(f"[缺失] {pkg_name:<14} 需要安装（运行下一格即可）")


# PyTorch 生态三件套 + 常用辅助库
show_version("torch")          # 深度学习发动机
show_version("transformers")   # Hugging Face 模型库
show_version("datasets")       # 数据集工具
show_version("trl")            # 微调训练器
show_version("peft")           # 高效微调
show_version("accelerate")     # 训练加速/设备管理

# 看看有没有 GPU。没有也能学，本教程全部用 CPU 能跑的小模型
print("-" * 40)
try:
    import torch
    if torch.cuda.is_available():
        print("检测到 GPU：", torch.cuda.get_device_name(0), "→ 训练会快很多")
    else:
        print("当前是 CPU 模式 → 教程照样能跑，只是慢一点")
except Exception:
    print("PyTorch 还没装好，先运行下面的安装单元格")


# ============================================================
# 一键安装依赖（在 Colab 或本地 Jupyter 里运行）
# 说明：%pip 是 Notebook 专用语法，等价于在终端 pip install
# ============================================================
# 依赖安装（在终端执行下面这行）：pip install -q transformers datasets trl peft accelerate


# ============================================================
# 最小验证：创建一个张量并做一个简单运算
# 张量(Tensor) 就是 PyTorch 里的"多维数组"，后面会详细讲
# ============================================================
import torch

x = torch.tensor([1.0, 2.0, 3.0])   # 创建一个小张量
print("张量 x =", x)
print("求和 =", x.sum().item())      # .item() 把单个元素的张量转成普通数字
print("环境就绪！接下来进入第 1 章。")
