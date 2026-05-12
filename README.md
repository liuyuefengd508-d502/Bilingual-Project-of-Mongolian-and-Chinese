# Bilingual Project of Mongolian and Chinese

蒙汉双语跨域文本检测研究项目（Mongolian–Chinese Cross-Domain Text Detection via UDA）。

## 仓库结构

```
.
├── docs/
│   └── research_plan.md          # 主研究计划（路线 B：跨域 UDA 文本检测）
└── mn-uda-textdet/               # 研究代码仓库
    ├── README.md
    ├── requirements.txt
    ├── environment.yml
    ├── tools/                    # 数据核对、MPS 自检、划分构建
    ├── splits/                   # MMOCR JSONL 划分（7:1:2, seed=42）
    ├── docs/                     # 数据报告、文献综述
    ├── configs/                  # 训练配置（W2 起填充）
    └── datasets/                 # 自定义 PyTorch Dataset（W2 起填充）
```

## 研究路线

- **任务**：在"手写蒙文档案 ↔ 蒙汉双语自然场景"两域上做无监督域自适应（UDA）文本检测。
- **方法**：Teacher-Student 自训练 + 域不变特征对抗对齐 + 蒙文长宽比感知伪标签筛选。
- **硬件**：Mac M5 Pro / Apple Silicon GPU (MPS)。
- **目标**：12 周 → SCI 二区期刊投稿。

详细计划见 [`docs/research_plan.md`](docs/research_plan.md)。

## 数据

数据集 `蒙汉双语/`（手写档案 1000 张 + 自然场景 1501 张）出自：

> *基于 USTB 学位论文的公开 benchmark*，已发表于：
> - IJDAR 2026（手写蒙文档案文本检测）
> - J. King Saud Univ. CS 2025（蒙汉双语自然场景文本检测）
> - Sensors 2024（场景文本中文子集识别）

> 原始数据与论文 PDF **未上传至本仓库**（参见 `.gitignore`）。

## 快速开始

```bash
cd mn-uda-textdet
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tools/check_mps.py
python tools/build_splits.py --root ../蒙汉双语 --seed 42
```

## 进度

- [x] W1: 数据健康检查、文献定位、跨域划分（MMOCR JSONL）
- [ ] W2–W3: DBNet++ / FCENet 单域 oracle + Source-only 跨域下界
- [ ] W4–W6: Teacher-Student + 域对抗对齐主方法
- [ ] W7–W9: 消融、SOTA 对比、可视化
- [ ] W10–W12: 论文撰写与投稿
