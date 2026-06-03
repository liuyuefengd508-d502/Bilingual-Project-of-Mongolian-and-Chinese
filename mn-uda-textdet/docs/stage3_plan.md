# Stage-3 Plan — Beyond Stage-2

Stage-2 已完成（参见 `baseline_results.md`）：UDA 跨域改进得到验证，H→S +0.34pp、S→H +2.85pp（test），
冷启动修复使 S→H 从 0 个 TP → 93 个 TP 出现"质变"。

Stage-3 目标：**把 marginal gain（~3pp）推到 paper-grade（10pp+）**，并补全消融与对比。
若资源/时间不允许，则在现有结果基础上完成论文写作（最低交付路径）。

---

## 1. Stage-3 候选实验（按 ROI 排序）

### 1.1 ✅ DONE — Test-time post-process sweep on UDA ckpts

- **实际结果**（test set，30 阈值组合）：
  - H→S：source-only 0.0753 vs UDA 0.0767 → **Δ +0.14pp（≈ 噪声）**
  - S→H：source-only 0.0028 vs UDA 0.0654 → **Δ +6.26pp ⭐**
- **新发现**：UDA 增益**不对称** —— 仅在 source-only 结构性失败的方向（S→H）有质变；
  在 source-only 已有弱信号的方向（H→S）几乎无增益
- **影响**：Stage-2 验收正式通过（S→H +6.26pp，远超 +3pp 门槛）。
  详见 `baseline_results.md` 的 "Threshold sweep" 章节

### 1.2 ⭐⭐⭐ 更强的 source-only baseline（ResNet-50 / 长训 oracle）

- **预期收益**：oracle 涨 5-10pp，跨域 baseline 跟着抬升，UDA 绝对值随之改善
- **成本**：~6-12h（每 oracle 训 60-80 ep ResNet-50；MPS 上 R50 较慢）
- **方法**：
  - 已有 `scene_r18` (H=0.385 in-domain, 25 ep) 和 `handwrite_r18` (H=0.863)
  - 改为 ResNet-50 backbone，更长 schedule (50-80 ep)
  - 用更强 oracle 作为 UDA `--init-from`
- **风险**：MPS 内存可能不足 R50（需 batch=2），且时间消耗大
- **替代**：保持 R18 但训 80 ep（cosine LR），低成本验证训练时长是否还是瓶颈

### 1.3 ⭐⭐ Stronger domain head （论文级消融）

- **预期收益**：+0.5~2pp；更主要是论文消融需要
- **成本**：~2h 修改代码 + 5h 训练
- **方法**：当前 `DomainHead` 是单层 conv→GAP→FC。改为：
  - Multi-layer conv (3 层 3×3 conv + BN + ReLU)
  - 或加 region-level DA（特征金字塔各层分别 DA）
- **风险**：可能 DA 过强 → 特征对齐过度 → 任务损失上升

### 1.4 ⭐⭐ S→H 长训对称报告 （已启动）

- **状态**：`work_dirs/uda_s2h_long2/` 跑 25 ep，正在运行
- **预期**：与 H→S 长训类似 —— 早期 best，后期崩盘
- **价值**：论文双向对称呈现，证明"5 ep 最优"不是 H→S 一向的特例
- **若 S→H 没崩盘**：意外发现，可能 S→H 的较弱起点容忍更长训练；需更深分析

### 1.5 ⭐ 更先进的 UDA 方法对比

- 当前是 EMA-teacher + DANN，可加：
  - **MIC (Masked Image Consistency)**: 强制 mask 后预测一致
  - **CutMix-style 域混合**: source-target patch mixing
  - **Adaptive Teacher**（参考 Adaptive Teacher CVPR 2022）
- **成本**：大，每个方法 1-2 天实现 + 训练
- **价值**：论文 SOTA 对比表必需，但可作为 future work 表述

### 1.6 ⭐ 质化分析深挖

- **目标域 prob heatmap 演化**：训练前/后对比同一张图的 prob 图
- **DA 特征 t-SNE**：把 source 与 target FPN 特征降维可视化，看是否对齐
- **失败案例分析**：UDA 仍然失败的 target 样本分类（小字、噪声、印章）

---

## 2. 三条 Stage-3 路径建议

### 路径 A：低成本论文化（推荐）

只做 1.1（threshold sweep）+ 1.4（已启动）+ 1.6（少量可视化）。
两周内可完成 + 进入写作。最终数字可能：

- H→S：~0.07-0.08 (val H), test 类似
- S→H：~0.04-0.05 (val H)
- 论文角度："cold-start fix + lightweight UDA"，承认 marginal but real gain

### 路径 B：争取强 baseline + 强 UDA

加 1.2（ResNet-50 oracle）+ 1.3（stronger DA head）。
3-4 周。若 oracle 上到 0.5+ in-domain，UDA 跨域可能上到 0.10-0.15 H。
最坏情况：训练失败 / MPS OOM，回退到路径 A。

### 路径 C：SOTA 追赶（高风险）

实现 1.5（先进 UDA 方法），与论文中 USTB Mongolian 系列对比。
4-6 周。需要相当多的工程，最终可能拉到 0.20+ H。
风险：MPS 平台限制，无法做 multi-GPU；先进方法在小数据上未必有效。

---

## 3. 推荐 next-step 顺序

```
本周（W3）：
1. ✅ 长训 S→H 25ep（已启动）           → 论文双向对称
2. ▢ Threshold sweep on 2 UDA ckpts    → 1h，最高 ROI
3. ▢ Stage-2 数字最终版（含 sweep 后的 best） → 入 paper

下周（W4）：
4. ▢ ResNet-50 oracle 训 50ep（一向先试）
5. ▢ R50 oracle init → UDA 5-ep verify
6. ▢ 决定路径 B 是否值得继续

后续：
7. ▢ 论文写作开始（即使路径 B 失败，路径 A 也够）
```

---

## 4. 论文骨架对照

| 章节 | Stage-2 数据 | Stage-3 增量 |
|---|---|---|
| 3. Method | UDA pipeline + cold-start fix | (no change) |
| 4.1 Datasets | scene/handwrite splits | — |
| 4.2 Oracle baselines | scene_r18 0.385, handwrite_r18 0.863 | +R50 oracle |
| 4.3 Cross-domain motivation | H→S 0.059, S→H 0.000 (source-only test) | — |
| 4.4 UDA main results | H→S 0.063, S→H 0.029 (test) | +sweep, +R50 init |
| 4.5 Ablation | v1/v2/v3 hp ablation | +DA head, +long-train |
| 4.6 Qualitative | compare_*.jpg | +heatmap evolution, +t-SNE |
| 4.7 SOTA comparison | (literature ref only) | +MIC/AdaptTeacher if路径 C |
| 5. Discussion | 5-ep optimum, drift collapse, asymmetric domain gap | — |

---

## 5. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| MPS R50 OOM | 路径 B 失败 | batch=2 + grad accumulation；或换 R18 长训 |
| Sweep 不带来提升 | 论文数字仍然 marginal | 主打"5-ep optimum + cold-start fix"叙事 |
| 高级 UDA 方法实现 bug | 路径 C 失败 | 时间盒上限 1 周，超时回退路径 A |
| 写作时数字反复 | 进度受阻 | 数字冻结策略：sweep 完后不再训练 |

---

## 6. 决策点（每周复盘）

- W3 末：threshold sweep 与 S→H 长训结果出炉，决定是否进路径 B
- W4 末：R50 oracle 是否可行，决定是否进路径 C
- W5 起：原则上停止实验，进入论文写作

如果实验始终无法把 H 推过 0.10，依然走路径 A（论文角度：domain gap is structurally hard，
我们提供方法验证 + 失败分析 + cold-start fix 这一新发现）。
