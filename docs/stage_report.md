# 蒙汉双语跨域文本检测 — 阶段性进度报告

**报告性质：** 执行层进度汇总（对照 [research_plan.md](research_plan.md) 的阶段与任务编号）。  
**路线：** B — 手写档案 ↔ 自然场景，无监督域自适应（UDA）文本检测。  
**截至：** 2026-05-13（与 [README.md](../README.md)、[mn-uda-textdet/docs/baseline_results.md](../mn-uda-textdet/docs/baseline_results.md)、子仓 [README.md](../mn-uda-textdet/README.md) 同步）。

---

## 1. 总体结论

- **阶段 0（W1）** 目标已达成：数据盘点与清洗规则明确、MMOCR 风格 JSONL 划分可复现、与 USTB 学位论文及三篇已发表论文的差异化定位已写入综述文档。
- **阶段 1（W2–W3）** 的**核心验收项**（四组 oracle / source-only 对照表）已用 **DBNet + ResNet-18、25 epoch、MPS** 完成；**FCENet 第二骨架、更长训练、ResNet-50 强 oracle** 仍为**选做**。
- **阶段 2（W4–W6）— T7 主方法**：**首版可训练实现已落地**（[`tools/train_uda.py`](../mn-uda-textdet/tools/train_uda.py)），含 EMA 教师、目标域伪标签、图像级 DANN（GRL）、源/目标 val 与 `best_student` 保存、断点续训、`--preset s2h|h2s` 等。**Conda 专用轻量训练环境**（[`environment_uda.yml`](../mn-uda-textdet/environment_uda.yml) + [`scripts/bootstrap_conda_uda.sh`](../mn-uda-textdet/scripts/bootstrap_conda_uda.sh)）与 **一键长训脚本**（[`scripts/run_uda_s2h_long.sh`](../mn-uda-textdet/scripts/run_uda_s2h_long.sh)，默认 40 epoch、`--data-root` 指向仓库根）已就绪；本机已用 **pip torch + MPS** 校验通过。**尚未完成**计划中的「至少一向相对 source-only **≥3% H-mean**」的**长训跑满后的指标成表**与正式验收记录。
- **阶段 3–4**（消融、SOTA、论文）尚未系统开展。

一句话：**基线与跨域动机已闭合；UDA 代码、评测工具链与 Conda 长训环境已就绪；当前缺口是长训完成后的 ΔH 数值验收与论文材料固化。**

---

## 2. 里程碑对照（研究计划 §3）

| 阶段 | 周次（计划） | 完成标志（计划原文） | 当前状态 |
|------|----------------|----------------------|----------|
| 阶段 0 | W1 | 干净划分 JSON；综述与 PDF 差异化 | **已完成** |
| 阶段 1 | W2–W3 | 四组指标：手写内、自然内、两向跨域 | **核心已完成**（DBNet-R18）；FCENet / 强 oracle 选做 |
| 阶段 2 | W4–W6 | UDA 可训练实现；至少一向下界 +≥3% H | **部分完成**：可训管线 + Conda（torch/OpenCV/MPS）+ 长训脚本已具备；**数值验收未完成**（待 `work_dirs` 成 ckpt 后 `eval_compare` / `uda_eval_suite`） |
| 阶段 3 | W7–W9 | 消融、SOTA、可视化；达成功标准 1 | **未开始** |
| 阶段 4 | W10–W12 | 初稿 → 修改 → 投稿 | **未开始** |

---

## 3. 任务级进度（研究计划 §4，摘要）

| ID | 任务 | 状态 | 说明 |
|----|------|------|------|
| T0 | 仓库与环境 | **完成** | `mn-uda-textdet/`、`environment.yml`、`environment_uda.yml`（UDA 轻量 conda）、`scripts/bootstrap_conda_uda.sh`（可选本地 `.miniforge3/`）、`check_mps.py` 等 |
| T1 | 数据与 JSONL 划分 | **完成** | `splits/*.json`、`build_splits_report.json`、`dataset_report.json` |
| T2 | 文献与差异化 | **部分完成** | [related_work.md](../mn-uda-textdet/docs/related_work.md) §8 ≥15 条精读骨架；§3–§6 可随写作深化 |
| T3 | 单域 oracle | **完成（R18）** | 见 [baseline_results.md](../mn-uda-textdet/docs/baseline_results.md) |
| T4 | FCENet 等第二骨架 | **未做** | P1 |
| T5 | Source-only 下界 | **完成** | 双向四格表已闭合 |
| T6 | 风格迁移弱 UDA | **未做** | P1 |
| **T7** | **UDA 主方法** | **进行中** | **`train_uda.py`**：同上；**`run_uda_s2h_long.sh`** 默认长训（`KMP_DUPLICATE_LIB_OK`、`python -u`）；**`tests/test_hmean_metrics.py`**（无 torch 的 H-mean 回归）；**待长训结束后的 ΔH 成表与 ≥3% 验收** |
| T8–T12 | 消融至投稿 | **未做** | 依赖 T7 有效结果 |

---

## 4. 主要产出物（可复现资产）

| 类别 | 路径或说明 |
|------|------------|
| 研究计划与进度 | [docs/research_plan.md](research_plan.md)、本文档、[README.md](../README.md) |
| 基线与四组表 | [mn-uda-textdet/docs/baseline_results.md](../mn-uda-textdet/docs/baseline_results.md)（含阈值扫描结论） |
| 文献 | [mn-uda-textdet/docs/related_work.md](../mn-uda-textdet/docs/related_work.md) |
| 数据划分 | `mn-uda-textdet/splits/{scene,handwrite}_{train,val,test}.json`（`build_splits.py` 生成） |
| 单域训练 / 评测 | `tools/train_dbnet.py`、`tools/eval.py`、`tools/sweep_postprocess.py` |
| **UDA 训练** | **`tools/train_uda.py`**（Teacher–Student + 伪标签 + DANN；`--preset s2h|h2s`；`--resume`；`best_student.pth` / `latest.pth` 含 opt+sched） |
| **UDA Conda 环境** | **`environment_uda.yml`**（环境名 `mn-uda-textdet-uda`；torch/torchvision 经 pip 以启用 MPS）、**`scripts/bootstrap_conda_uda.sh`**、**`scripts/run_uda_s2h_long.sh`** |
| **指标单测** | **`tests/test_hmean_metrics.py`**（`unittest`，不加载 `torch`） |
| **批量评测** | **`tools/uda_eval_suite.py`**（多 ann 表 + 可选 JSON） |
| **基线 vs UDA 对比** | **`tools/eval_compare.py`**（双 ckpt 同 ann，ΔH/ΔP/ΔR） |
| **共享 H-mean 逻辑** | **`metrics/det_eval.py`**（`hmean_on_loader` / `hmean_on_ann_file` / `load_detection_ckpt`） |
| 数据加载 | `datasets/text_det_dataset.py` 支持 **`labeled=False`**（目标域无标签流） |
| 模型 | `models/dbnet.py` 支持 **`return_feat`**；`models/domain_adversarial.py`（GRL + 域头） |
| 实验权重 | `mn-uda-textdet/work_dirs/`（**gitignore**；需本地/网盘归档） |

---

## 5. 关键实验结果（阶段 1，默认后处理）

评测协议：**640**，**IoU=0.5**，**`box_thresh=0.45`，`min_score=0.5`**（详见基线文档）。

| 训练域 | scene_test | handwrite_test |
|--------|------------|----------------|
| Scene-only（自然 oracle） | **H=0.385** | **H=0.000** |
| Handwrite-only（手写 oracle） | **H=0.059** | **H=0.863** |

**补充：** Scene oracle 在测试时阈值网格上 **H 最高约 0.393**（`sweep_postprocess_scene_test.json`）。  
**UDA 正式对比表**：待 `train_uda` 长训完成后，用 **`uda_eval_suite.py`** / **`eval_compare.py`** 相对上表 source-only 填写。

---

## 6. UDA 实现要点（阶段 2 代码现状，便于交接）

- **优化目标**：源域全监督 `db_loss` + 目标域教师伪标签 `db_loss`（权重 warmup）+ 域分类 CE（GRL 强度与权重 warmup）。  
- **伪标签**：教师概率图 → `prob_to_polygons` → 面积/长宽比过滤 → `generate_dbnet_targets`；`--pseudo-max-candidates` 控制候选数。  
- **验证**：`--val-source`（db_loss）、`--val-target`（H-mean，开发调参）；`--val-every` 降频；`--best-on` 控制 `best_student.pth`（避免 H=0 冷启动误存）。  
- **记录**：`history.json`；可选 **`--log-jsonl`** 每 epoch 一行 JSON。

---

## 7. 未完成项与风险（简列）

1. **阶段 2 数值验收**：尚未报告相对 source-only 的 **≥3% H**（双向上或单向上）；**待当前/后续长训 ckpt 固化后**用 `eval_compare.py` / `uda_eval_suite.py` 填表。  
2. **与 USTB 单域 SOTA 的绝对差距**：叙事上坚持「UDA 相对 source-only 的提升」。  
3. **MPS / 算子**：Conda 下 conda-forge 与 pip torch 可能双份 OpenMP，需 **`KMP_DUPLICATE_LIB_OK=TRUE`**（`run_uda_s2h_long.sh` 已导出）；复杂模块仍可能需要 CPU 兜底或云端 GPU（计划书已列）。  
4. **多随机种子**：成功标准要求方差报告；流水线待固化。  
5. **扩展项**：T4、T6、长训/R50、T10 可视化仍为选做。

---

## 8. 下一阶段建议（按优先级）

1. **长训 UDA**：推荐 `cd mn-uda-textdet && bash scripts/run_uda_s2h_long.sh`（或等价 `train_uda.py --preset s2h --pretrained --val-target/--val-source ... --epochs 40+`）；产出默认在 **`work_dirs/uda_s2h_long/`**。  
2. **验收表**：对 **handwrite_test**（及反向 **scene_test**）跑 **`eval_compare.py`**（source-only ckpt vs UDA ckpt），核对是否达到阶段 2 阈值。  
3. **消融预留**（T8）：自训练 / 对抗 / 伪阈值 / EMA / 域头 LR 等开关式记录。  
4. **强基线选做**：FCENet、T6、R50/长训 oracle，支撑论文公平性讨论。

---

## 9. 修订记录

| 日期 | 修订内容 |
|------|----------|
| 2026-05-12 | 首版：阶段 0–1、四组基线、风险与建议 |
| 2026-05-12 | v2：阶段 2（T7）代码与工具链落地情况；更新任务表与产出物清单；调整总体结论与下一阶段 |
| 2026-05-13 | v3：Conda 轻量环境（`environment_uda.yml`）、bootstrap / 长训脚本、H-mean 单测；里程碑与 T0/T7/产出物/建议同步；**数值 ΔH 仍以长训完成为准** |
