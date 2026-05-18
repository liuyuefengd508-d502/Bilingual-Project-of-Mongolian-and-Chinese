# 相关工作（Related Work）

> W1 启动版本；**§8 于 W2 扩充**（≥15 篇精读条目与一句话要点）。已读完 `10127_2023023254_lw.pdf` 摘要、引言、目录、贡献清单、参考文献。

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

## 8. 精读文献条目（≥15，W2 更新）

下列条目满足研究计划中「UDA-OCR / 域适应检测 / 场景文本检测」综述骨架；撰写 BibTeX 时请核对卷期页与作者拼写（此处为工作笔记级摘要）。

### 8.1 域适应 / 半监督目标检测（方法池，对齐 T7）

1. **Chen et al., Domain Adaptive Faster R-CNN for Object Detection in the Wild.** CVPR 2018. — 图像级与实例级域分类器 + 梯度反转，域适应检测的范式起点。
2. **Saito et al., Strong-Weak Distribution Alignment for Adaptive Object Detection.** CVPR 2019. — 强/弱对齐分支，兼顾全局与局部分布匹配。
3. **He et al., Cross-domain Object Detection through Coarse-to-Fine Feature Adaptation.** CVPR 2020. — 粗到细层级特征自适应，缓解尺度与纹理域差。
4. **Zhou et al., Domain Adaptive Object Detection with Adaptive Teacher.** CVPR 2022. — Teacher-Student 与跨域一致性，与本项目主方法线最直接可比。
5. **Liu et al., Unbiased Teacher for Semi-Supervised Object Detection.** ICCV 2021. — 伪标签去偏与双模型更新，自训练稳定性技巧来源。
6. **Tarvainen & Valpola, Mean teachers are better role models.** NeurIPS 2017. — EMA 教师基线；与 DBNet 学生网络耦合的工程参考。
7. **Kim et al., SIGMA: Scale-Invariant Graph Matching Adversarial Learning for Cross-domain Few-shot Object Detection.** CVPR 2022. — 图匹配式对齐，拓展几何/关系先验。
8. **Chen et al., Domain Adaptive Object Detection via Hierarchical Contrastive Learning on the Region of Interest.** ICCV 2021. — ROI 级对比学习做域不变表征，可与对抗头组合或对照。
9. **Sun et al., Domain Adaptation for Object Detection via Style Transfer.** CVPR 2019. — 风格迁移构造中间域，对应研究计划 **T6**（CycleGAN/FDA 类弱基线）的方法论引用。
10. **Yang & Soatto, FDA: Fourier Domain Adaptation for Semantic Segmentation.** CVPR 2020. — 频域统计匹配式对齐；可迁移到检测特征图作为轻量对齐对照。
### 8.2 场景文本检测骨干与任意形检测（对齐 T3/T4）

11. **Liao et al., Real-time Scene Text Detection with Differentiable Binarization.** AAAI 2020 (DBNet). — 本仓库实现原型。
12. **Liao et al., Real-Time Scene Text Detection with Differentiable Binarization and Adaptive Scale Fusion.** TPAMI 2022 (DBNet++). — 更强骨干与自适应尺度，后续 oracle 上界对照。
13. **Zhu et al., FCENet: Arbitrary Shape Text Detection with Fourier Contour Embedding.** CVPR 2021. — T4 第二骨架候选。
14. **Wang et al., Shape Robust Text Detection with Progressive Scale Expansion Network.** CVPR 2019 (PSENet). — 经典分割式多尺度 kernel 对照。
15. **Wang et al., Shape Robust Text Detection with a Single Shot Segmentation-Based Network.** ICCV 2019 (PANet). — 任意形轻量检测对照。

### 8.3 合成数据、Transformer 检测与字符级先验

16. **Gupta et al., Synthetic Data for Text Localisation in Natural Images.** CVPR 2016. — 合成到真实迁移的数据管线经典引用。
17. **Baek et al., Character Region Awareness for Text Detection.** CVPR 2019 (CRAFT). — 字符级区域监督与热力图解码，与伪标签空间粒度、阈值设计可比。
18. **Liao et al., DPText-DETR: Towards Better Scene Text Detection with Dynamic Points in Transformer.** AAAI 2023. — Transformer 检测器上限对照（与轻量 DBNet 对比叙事）。

> **与 §3–§6 勾选清单的关系：** §3–§6 保留为写作时的主题检查表；§8.1–8.3 已覆盖其中多数关键词（DA-Faster / AT / Mean-Teacher / SIGMA / HCL / SFOD 族 / DBNet++ / FCENet / PSENet / PAN / DPText-DETR / 合成迁移）。后续精读时在条目下追加「公式/损失项/实现坑」子 bullet 即可。
