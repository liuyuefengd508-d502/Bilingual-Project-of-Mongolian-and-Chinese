# mn-uda-textdet

蒙汉双语跨域文本检测（Mongolian-Chinese Cross-Domain Text Detection via UDA）。

研究计划详见仓库外的 `../docs/research_plan.md`。

## 目录结构

```
mn-uda-textdet/
├── configs/         # 训练/评测配置（MMOCR 1.x 风格）
├── tools/           # 实用脚本（环境自检、数据核对、转换）
│   ├── check_mps.py        # 检查 PyTorch MPS 可用性
│   └── verify_dataset.py   # 核对 自然场景 total/ vs label/ 一一对应
├── datasets/        # 数据加载器（不存原始数据）
├── splits/          # 数据划分 JSON（train/val/test）
├── docs/
│   └── related_work.md     # 文献综述与 PDF 差异化分析
└── environment.yml  # conda 环境
```

## 环境（Mac M5 Pro / MPS）

无 conda 的本机推荐使用 venv：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python tools/check_mps.py
```

或使用 conda：

```bash
conda env create -f environment.yml
conda activate mn-uda-textdet
python tools/check_mps.py
```

> 注意：MMOCR / MMDet / MMCV 体系**留到 W2 基线阶段再装**，避免与 PyTorch/MPS 早期排错混在一起。

## 数据

- 手写档案：`../蒙汉双语/手写档案/train/` + `train-label/{training,test}.txt`（PaddleOCR 行格式，transcription=0）
- 自然场景：`../蒙汉双语/自然场景/total/` + `label/*.txt`（`x1,y1,...,x4,y4,#`）

> 已知问题：自然场景 1502 张图 vs 1501 个标签，需用 `tools/verify_dataset.py` 核对。

## 路线图

| 阶段 | 周次 | 关键任务 |
|---|---|---|
| 0 | W1 | 数据清洗 / 文献综述 / PDF 差异化定位 |
| 1 | W2–W3 | DBNet++ 单域 oracle + Source-only 跨域下界 |
| 2 | W4–W6 | Teacher-Student + 域对抗对齐主方法 |
| 3 | W7–W9 | 消融 + SOTA 对比 + 可视化 |
| 4 | W10–W12 | 论文撰写 + 投稿 |
