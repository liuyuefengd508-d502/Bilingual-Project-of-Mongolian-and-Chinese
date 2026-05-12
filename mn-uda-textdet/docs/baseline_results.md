# Baseline Results — Stage 1

**Setup**

- Model: DBNet (ResNet-18 + FPN), no DCN.
- Input: 640×640, ImageNet pretrained backbone.
- Augmentation: rotation ±10°, scale [0.75, 1.5], random crop, brightness/contrast ±0.2 (no flip — Mongolian script is direction-sensitive).
- Optimizer: AdamW, lr=1e-3, weight_decay=1e-4, cosine annealing.
- Loss: balanced BCE (prob, OHEM 1:3) + Dice (binary) + L1 (threshold band, w=10).
- Post-processing: `box_thresh=0.45`, `min_score=0.5`, `unclip_ratio=1.5`, IoU=0.5 for H-mean.
- Hardware: Mac M5 Pro, MPS backend, batch=4, 25 epochs.

## Stage-1 cross-domain motivation table

| Train domain | Eval on scene_test | Eval on handwrite_test |
|---|---|---|
| **Scene-only** (oracle on scene) | **H = 0.385** (P=0.55, R=0.30) | **H = 0.000** (P=0, R=0, 152 preds, 0 TP) |
| Handwrite-only | _to be measured_ | _to be measured_ |

**Domain gap (S → H, source-only): H 0.385 → 0.000**, i.e. complete failure.
This is the empirical motivation for the UDA approach in route B of the research plan.

## Scene oracle training trajectory

```
ep  loss     P      R      H
 1  3.1642  0.408  0.026  0.049
 4  2.8948  0.552  0.099  0.168
 8  2.7124  0.465  0.191  0.271
12  2.6105  0.523  0.238  0.327
16  2.5559  0.513  0.274  0.357
20  2.4250  0.569  0.288  0.382
23  2.3840  0.579  0.297  0.393   <- best
25  2.3810  0.570  0.300  0.393   <- final
```

- Loss is still trending down at ep25 → model is **under-trained**, not over-fit.
- Recall (~30%) is the main bottleneck. Increasing epochs and/or harder negatives mining should lift it.

## What this implies for the rest of W2 / W3

1. **Push baseline higher** before declaring "Stage 1 complete":
   - Train 50–80 epochs (LR cosine over more steps).
   - Try ResNet-50 backbone (we have FPN code; ~3× slower per step).
   - Tune `box_thresh` per epoch (or sweep 0.3–0.55 at test time).
2. **Symmetric experiment**: train Handwrite-oracle and run cross-domain in both directions.
3. **Investigate why H→S H=0**:
   - Visualize prob maps on 5 cross-domain samples (`viz_handwrite_crossdomain/`).
   - Likely failure modes: (a) handwrite ink is grayscale / low-contrast → backbone activations weaker; (b) text aspect ratio differs (vertical Mongolian vs. horizontal Chinese signage); (c) absence of natural-image colour priors.
4. **Numbers to beat** (literature reference, not this run):
   - USTB scene paper (J. King Saud Univ. CS 2025): H ≈ 87.6% on a 7:1:2 split similar to ours.
   - USTB handwrite paper (IJDAR 2026): H ≈ 88% on archival Mongolian.
   - Our DBNet ResNet-18 25-ep baseline (H=0.385) is far from these — but those use full DBNet ResNet-50 with multi-stage training, OCR-aware augmentation, and 100+ epochs.

## File map (this experiment)

- `work_dirs/scene_r18/best.pth` — best ckpt (epoch 23).
- `work_dirs/scene_r18/history.json` — full P/R/H per epoch.
- `work_dirs/scene_r18/test_scene.json` — final in-domain test metrics.
- `work_dirs/scene_r18/test_handwrite.json` — cross-domain S→H source-only.
- `work_dirs/scene_r18/viz_scene/` — 6 in-domain qualitative samples.
- `work_dirs/scene_r18/viz_handwrite_crossdomain/` — 6 cross-domain failure cases.
