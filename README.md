# Bilingual Project of Mongolian and Chinese

蒙汉双语跨域文本检测研究项目（Mongolian–Chinese Cross-Domain Text Detection via UDA）。

## 仓库结构

```
.
├── docs/
│   ├── research_plan.md          # 主研究计划（路线 B：跨域 UDA 文本检测）
│   └── stage_report.md           # 阶段性进度报告（对照里程碑与任务表）
├── .miniforge3/                  # 可选：bootstrap_conda_uda.sh 安装的本地 Miniforge（gitignore）
└── mn-uda-textdet/               # 研究代码（含 `train_uda.py`、`uda_eval_suite.py`、`eval_compare.py`）
    ├── README.md
    ├── requirements.txt
    ├── environment.yml           # 全栈（含 MMOCR）参考
    ├── environment_uda.yml       # UDA 轻量 conda（torch/cv2 经 conda+pip，MPS）
    ├── scripts/                  # bootstrap_conda_uda.sh、run_uda_s2h_long.sh
    ├── tests/                    # unittest（如 H-mean 指标）
    ├── tools/                    # 数据核对、MPS 自检、划分构建、UDA 训练
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

详细计划见 [`docs/research_plan.md`](docs/research_plan.md)。阶段性执行汇总见 [`docs/stage_report.md`](docs/stage_report.md)。

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
- [x] W2–W3（核心交付）: DBNet（R18）双域单域 oracle + 双向 Source-only 跨域下界；四组对照表与可选后处理扫描见 [`mn-uda-textdet/docs/baseline_results.md`](mn-uda-textdet/docs/baseline_results.md)。FCENet 第二骨架、更长训 / R50 强 oracle 仍待选做。
- [ ] W4–W6: **进行中** — [`train_uda.py`](mn-uda-textdet/tools/train_uda.py) 已实现 T7 主干；**[`environment_uda.yml`](mn-uda-textdet/environment_uda.yml) + [`scripts/bootstrap_conda_uda.sh`](mn-uda-textdet/scripts/bootstrap_conda_uda.sh) + [`scripts/run_uda_s2h_long.sh`](mn-uda-textdet/scripts/run_uda_s2h_long.sh)** 提供 Conda 下 torch/OpenCV/MPS 与默认长训入口。相对 source-only **≥3% H** 的正式成表仍待长训 ckpt 与 [`eval_compare.py`](mn-uda-textdet/tools/eval_compare.py) / [`uda_eval_suite.py`](mn-uda-textdet/tools/uda_eval_suite.py) 验收。详见 [`docs/stage_report.md`](docs/stage_report.md) v3。
- [ ] W7–W9: 消融、SOTA 对比、可视化
- [ ] W10–W12: 论文撰写与投稿
