# 相关工作（Related Work）

> W1 启动版本。已读完 `10127_2023023254_lw.pdf` 摘要、引言、目录、贡献清单、参考文献。

## 1. 目录下学位论文 `10127_2023023254_lw.pdf` 摘要

**论文：** USTB（北京科技大学）硕士学位论文，2026.04 完成；选题"蒙汉双语多场景文本检测关键技术研究"。120 页。

**研究构成（三个子任务，均为单域）：**

1. **手写蒙文档案文本检测**
   - 挑战：旁注干扰、墨迹透印、文本倾斜。
   - 方法：efficient attention 模块分离文本/非文本区域；iterative point-shifting 优化候选框轮廓；自建 benchmark 数据集。
   - 结果：P=99.2 / R=98.8 / **F1=99.0**（自建集，单域内训练评测）。
   - 发表：*Contour thinning for text detection in Mongolian handwritten historical documents*, **IJDAR 2026**（中科院 3 区，已见刊）。

2. **蒙汉双语自然场景文本检测**
   - 挑战：书写布局多变、双语字符形态异质、图像模糊。
   - 方法：text restoration 模块去模糊 → coarse 文本轮廓与 character-level 文本图通过 deep relational reasoning 融合 → Transformer 边界精化。
   - 结果：**F1=84.6**, 16.4 FPS；自建蒙汉双语自然场景 benchmark，比次优方法 +1.3 F1。
   - 发表：*A text clarification and deep relational reasoning method for ... bilingual arbitrary-shaped scene text detection*, **J. King Saud Univ. CS 2025, 37(5)**（中科院 2 区）。

3. **中文子集场景文本识别**
   - 方法：FPN 中加 dual-attention concatenation；标准化字形对齐分支；像素级特征对齐。
   - 结果：CTR 中文集 Avg Acc=79.4%。
   - 发表：*Text Font Correction and Alignment Method for Scene Text Recognition*, **Sensors 2024, 24(24): 7917**（中科院 3 区）。

**关键结论（对本研究的影响）：**
- ✅ 我们手上的两个数据集 = 该论文构建的 benchmark；可作为**公开数据集**直接引用。
- ✅ 作者**两数据集独立训练独立评测**，**未涉及跨域 / UDA / 域自适应 / 域泛化**。
- ✅ 作者方法（Contour thinning / DRR-based detector）可作我们的 **strong source-only baseline 与 oracle 上界**。
- ⚠ 作者所有数据均不带识别 GT（仅手写不带，自然场景仅做了"中文子集识别"），与我们 dataset_report 一致。
- ⚠ 该论文已成为该 benchmark 的"事实基线"，我们的论文必须明确"任务设定不同（UDA）+ 提升相对 source-only 而非相对 oracle"以避免与其单域 SOTA 直接对比落败。

## 2. 我们的差异化贡献（更新）

1. **任务定义新颖：** 在该 benchmark 上**首次定义"手写档案 ↔ 自然场景"跨域文本检测任务**，提出两套评测协议（H→S, S→H），目标域无标签。
2. **方法贡献：** Teacher-Student 自训练 + 域不变特征对抗对齐 + 蒙文长宽比感知伪标签筛选。
3. **实验贡献：** 与该论文（IJDAR 2026 / KSU-CS 2025）方法对比"oracle 上界"，与通用 UDA-OCR 方法对比"跨域提升"。

## 3. 跨域目标检测 / UDA 通用方法（W1 待精读）

- [ ] DA-Faster (CVPR'18) — 域适应目标检测开山
- [ ] Mean-Teacher / Self-training for detection
- [ ] AT-detection (Adaptive Teacher, CVPR'22)
- [ ] PT (Probabilistic Teacher, ICML'22)
- [ ] SIGMA (CVPR'22) — 图匹配 UDA
- [ ] HCL — 层次对比学习 UDA
- [ ] SFOD / LODS — source-free UDA detection

## 4. 文本检测主流方法（W1 待精读）

- [ ] DBNet++ (TPAMI'22) — 我们 backbone 候选 1
- [ ] FCENet (CVPR'21) — 我们 backbone 候选 2
- [ ] PSENet / PAN — 经典分割式
- [ ] DPText-DETR (AAAI'23) — 论文已引用
- [ ] SRFormer — 论文已引用
- [ ] TextDCT — 论文已引用

## 5. UDA × 文本检测（W1–W2 重点综述）

- [ ] Self-training for scene text detection (Zhang et al., 2020)
- [ ] Synth-to-real text detection adaptation
- [ ] Cross-domain scene text detection via pseudo-labeling
- [ ] Few-shot / zero-shot 文本检测的迁移视角

## 6. 蒙文 / 低资源语言 OCR

- [ ] 传统蒙文识别（HMM/MLP 时代）
- [ ] 基于 CRNN 的现代蒙文识别
- [ ] 蒙文场景文本检测（除本论文外是否有他人工作？— W1 待查）
- [ ] 维语/藏语等少数民族文字检测识别（横向参照）

## 7. 风险监控

- **撞车监控：** 若发现作者已有 / 在投跨域 UDA 后续工作（同实验室常见），需立即调整：转 source-free UDA 或 cross-domain text spotting。
- **公平性：** 论文撰写时需 cite 这 3 篇作品并说明数据集来源，避免学术伦理问题。
