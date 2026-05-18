# mn-uda-textdet

蒙汉双语跨域文本检测（Mongolian-Chinese Cross-Domain Text Detection via UDA）。

研究计划详见仓库外的 `../docs/research_plan.md`。

## 目录结构

```
mn-uda-textdet/
├── configs/         # 训练/评测配置（MMOCR 1.x 风格）
├── tools/           # 实用脚本（环境自检、数据核对、转换）
│   ├── check_mps.py        # 检查 PyTorch MPS 可用性
│   ├── verify_dataset.py   # 核对 自然场景 total/ vs label/ 一一对应
│   ├── sweep_postprocess.py # 单次前向 + 后处理阈值网格（评测加速）
│   ├── train_dbnet.py       # 单域 DBNet 训练
│   ├── train_uda.py         # UDA：源域监督 + 目标域伪标签 + DANN 图像级域头
│   ├── uda_eval_suite.py    # 一次评测多个 ann（表格 + 可选 JSON）
│   └── eval_compare.py      # 两个 ckpt 在同一批 ann 上的 H 对比（ΔH / ΔP / ΔR）
├── splits/          # 数据划分 JSON（train/val/test）
├── tests/           # 无 torch 的指标回归（unittest）
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

### Conda：UDA 专用轻量环境（含 torch + OpenCV + MPS）

不装 MMOCR/MMDet，用于 `train_uda.py` / `train_dbnet.py` 长训：

```bash
cd mn-uda-textdet
bash scripts/bootstrap_conda_uda.sh
# 按脚本末尾提示 source conda.sh 后:
conda activate mn-uda-textdet-uda
```

- 若本机无 conda，脚本会把 **Miniforge** 装到仓库根目录 **`.miniforge3/`**（已写入根目录 `.gitignore`）。
- 环境名：**`mn-uda-textdet-uda`**，定义见 [`environment_uda.yml`](environment_uda.yml)（PyTorch/torchvision 走 **pip** 官方 macOS arm64 wheel，以启用 **MPS**）。
- macOS 上 conda-forge 与 pip torch 可能各带一份 OpenMP，需在训练前设置 **`export KMP_DUPLICATE_LIB_OK=TRUE`**（`scripts/run_uda_s2h_long.sh` 已默认导出）。

一键长训（scene→handwrite，40 epoch，`--data-root` 指向**仓库根**，与 `splits` 中 `蒙汉双语/...` 路径一致）：

```bash
cd mn-uda-textdet
bash scripts/run_uda_s2h_long.sh
# 或: EPOCHS=50 BATCH=4 bash scripts/run_uda_s2h_long.sh
```

日志与权重在 `work_dirs/uda_s2h_long/`（默认，可被环境变量 `OUT` 覆盖）。

> 注意：MMOCR / MMDet / MMCV 体系**留到 W2 基线阶段再装**，避免与 PyTorch/MPS 早期排错混在一起。

## 单元测试（可选）

不依赖 PyTorch；直接加载 `metrics/hmean.py`，需已安装 `numpy` 与 `shapely`（`environment.yml` 已包含）。

```bash
cd mn-uda-textdet && python -m unittest discover -s tests -p "test_*.py" -v
```

## 数据

- 手写档案：`../蒙汉双语/手写档案/train/` + `train-label/{training,test}.txt`（PaddleOCR 行格式，transcription=0）
- 自然场景：`../蒙汉双语/自然场景/total/` + `label/*.txt`（`x1,y1,...,x4,y4,#`）

> 已知问题：自然场景 1502 张图 vs 1501 个标签，需用 `tools/verify_dataset.py` 核对。

## 路线图

| 阶段 | 周次 | 关键任务 |
|---|---|---|
| 0 | W1 | 数据清洗 / 文献综述 / PDF 差异化定位 |
| 1 | W2–W3 | DBNet++ 单域 oracle + Source-only 跨域下界 |
| 2 | W4–W6 | **UDA 主方法**（[`tools/train_uda.py`](tools/train_uda.py)：EMA 教师、目标域伪标签、GRL 域分类器） |
| 3 | W7–W9 | 消融 + SOTA 对比 + 可视化 |
| 4 | W10–W12 | 论文撰写 + 投稿 |

## UDA 训练（`tools/train_uda.py`）

- 检测 H-mean 的批量评测实现集中在 [`metrics/det_eval.py`](metrics/det_eval.py)（`hmean_on_loader` / `hmean_on_ann_file`），供 `train_dbnet`、`train_uda` 与评测脚本复用。

- **源域**：`TextDetJsonDataset(..., labeled=True)`，完整 DBNet 监督。
- **目标域**：`labeled=False`，仅图像；**教师**（EMA）在目标图像上生成概率图 → 多边形 → `generate_dbnet_targets` 伪监督；**学生**在源+目标拼接 batch 上同时算源监督与伪监督。
- **域对抗**：对融合 FPN 特征做全局池化 + 线性域分类器，**梯度反转**强度 `grl_lambda` 与损失权重随 epoch 线性 ramp（`--pseudo-warmup-epochs`、`--da-warmup-epochs`）。
- **评测**：`tools/eval.py` 已支持读取 `latest.pth` / `best_student.pth` 中的 `student` 权重。
- **目标域 val 监控（可选）**：`--val-target splits/handwrite_val.json` 在 epoch 末用与 `train_dbnet` 相同协议算 **H-mean**（开发调参用：真实 UDA 不应在目标 test 上调参）。`--eval-box-thresh` / `--eval-min-score` 等与基线报告对齐。
- **最佳权重**：`--best-on auto`（默认）在提供 `--val-target` 时按 **val_target H-mean** 保存 `best_student.pth`，否则按 **val_src_loss** 最小保存；H 仍为 0 的冷启动不会写入无意义 best。
- **断点续训**：`--resume path/to/latest.pth` 恢复学生/教师/域头与 **optimizer + cosine** 状态；下一轮 epoch 从 `checkpoint.epoch + 1` 开始。

- **快捷配置**：`--preset s2h`（自然→手写：填 `scene_train` / `handwrite_train` / `scene_val`）；`--preset h2s`（手写→自然：对调 train + `handwrite_val`）。显式传入的路径优先于 preset 中的默认项。
- **验证频率**：`--val-every N` 每 N 个 epoch 跑一次 val（最后一 epoch 仍会跑）。
- **早停**：`--early-stop-patience K`（K>0）在按 `--best-on` 的指标连续 K 次「已跑验证」的 epoch 无提升时停止；未跑验证的 epoch 不计入。需存在对应 val 数据（例如 `target_hmean` 要 `--val-target`）。
- **稳定训练**：`--max-grad-norm`（>0 时裁剪学生+域头梯度）、`--pseudo-max-candidates`（教师多边形候选上限）。
- **域头学习率**：`--lr-domain` 单独设置域分类器 LR（默认与 `--lr` 相同）；常取更小值以稳定 GRL 训练。
- **JSONL 日志**：`--log-jsonl path/metrics.jsonl` 每 epoch 追加一行 JSON，便于外部画曲线。

**示例（`--preset`，推荐）**

```bash
python tools/train_uda.py --preset s2h --val-target splits/handwrite_val.json \
  --data-root .. --pretrained \
  --out work_dirs/uda_s2h --epochs 25 --batch 4
```

**基线 vs UDA 对比（同一 ann）**

```bash
python tools/eval_compare.py \
  --ckpt-a work_dirs/scene_r18/best.pth \
  --ckpt-b work_dirs/uda_s2h/best_student.pth \
  --label-a source_only --label-b uda \
  --data-root ..
```

**多 split 评测（UDA 或单域 ckpt）**

```bash
python tools/uda_eval_suite.py --ckpt work_dirs/uda_s2h/best_student.pth --data-root .. \
  --ann splits/scene_test.json:scene splits/handwrite_test.json:handwrite
```

**续训**

```bash
python tools/train_uda.py ... --resume work_dirs/uda_s2h/latest.pth --epochs 40
```

**快速冒烟（每 epoch 只跑几步）**

```bash
python tools/train_uda.py \
  --source-train splits/scene_train.json \
  --target-train splits/handwrite_train.json \
  --val-source splits/scene_val.json \
  --data-root .. --epochs 1 --batch 2 --max-steps 5
```

**训练后评测（学生网络）**

```bash
python tools/eval.py --ckpt work_dirs/uda_s2h/best_student.pth \
  --ann splits/handwrite_test.json --data-root .. --size 640 \
  --box-thresh 0.45 --min-score 0.5
```
