"""Visualize predictions vs GT for qualitative figures.

Renders side-by-side images:
    [original | GT polygons | predicted polygons + score]

Run:
    python tools/visualize.py --ckpt work_dirs/scene_r18/best.pth \
        --ann splits/scene_test.json --data-root .. --size 640 \
        --num 12 --out work_dirs/scene_r18/viz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets import TextDetJsonDataset
from models import DBNet, prob_to_polygons


GT_COLOR = (0, 200, 0)        # green
PRED_COLOR = (255, 64, 64)    # red
TEXT_BG = (0, 0, 0, 160)


def _denormalize(img_t: torch.Tensor) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = img_t.detach().cpu().numpy().transpose(1, 2, 0)
    arr = (arr * std + mean) * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _draw_polys(img: Image.Image, polys, color, scores=None) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for i, p in enumerate(polys):
        pts = [tuple(map(float, xy)) for xy in p.reshape(-1, 2)]
        if len(pts) >= 2:
            draw.line(pts + [pts[0]], fill=color, width=2)
        if scores is not None and len(pts) > 0:
            x, y = pts[0]
            draw.text((x + 2, y + 2), f"{scores[i]:.2f}", fill=color)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--ann", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path(".."))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--num", type=int, default=12)
    ap.add_argument("--backbone", type=str, default="resnet18")
    ap.add_argument("--box-thresh", type=float, default=0.3)
    ap.add_argument("--min-score", type=float, default=0.6)
    ap.add_argument("--unclip-ratio", type=float, default=1.5)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = (torch.device("mps") if (not args.cpu and torch.backends.mps.is_available())
              else torch.device("cpu"))
    print(f"Device : {device}")

    ds = TextDetJsonDataset(args.ann, args.data_root,
                            image_size=args.size, target_mode="binary")
    n = min(args.num, len(ds))
    print(f"Visualizing {n}/{len(ds)} samples to {args.out}")

    model = DBNet(backbone=args.backbone, pretrained=False).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    rng = np.random.default_rng(0)
    indices = rng.choice(len(ds), size=n, replace=False)

    with torch.no_grad():
        for k, idx in enumerate(indices):
            sample = ds[int(idx)]
            img_t = sample["image"].unsqueeze(0).to(device)
            out = model(img_t)
            prob = out["prob"][0].detach().cpu().numpy()
            preds = prob_to_polygons(
                prob, box_thresh=args.box_thresh,
                min_score=args.min_score, unclip_ratio=args.unclip_ratio,
            )

            img_np = _denormalize(sample["image"])
            img_pil = Image.fromarray(img_np)

            # GT polygons resized to network input space.
            rec = ds.data_list[int(idx)]
            orig_w = rec.get("width", args.size)
            orig_h = rec.get("height", args.size)
            sx = args.size / orig_w
            sy = args.size / orig_h
            gt_polys = []
            for inst in rec.get("instances", []):
                p = np.asarray(inst["polygon"], dtype=np.float32).reshape(-1, 2)
                p[:, 0] *= sx
                p[:, 1] *= sy
                gt_polys.append(p)

            gt_img = _draw_polys(img_pil, gt_polys, GT_COLOR)
            pred_polys = [pp for pp, _ in preds]
            pred_scores = [s for _, s in preds]
            pred_img = _draw_polys(img_pil, pred_polys, PRED_COLOR, pred_scores)

            # Probability map (heatmap) for context.
            prob_vis = (prob * 255).astype(np.uint8)
            prob_vis = np.stack([prob_vis] * 3, axis=-1)
            prob_pil = Image.fromarray(prob_vis).resize(img_pil.size, Image.BILINEAR)

            # Concatenate horizontally: [orig|GT|prob|pred].
            W, H = img_pil.size
            canvas = Image.new("RGB", (W * 4 + 30, H + 30), (32, 32, 32))
            for i, im in enumerate([img_pil, gt_img, prob_pil, pred_img]):
                canvas.paste(im, (i * (W + 10), 30))
            d = ImageDraw.Draw(canvas)
            for i, lab in enumerate(["input", f"GT ({len(gt_polys)})", "prob",
                                     f"pred ({len(pred_polys)})"]):
                d.text((i * (W + 10) + 4, 6), lab, fill=(240, 240, 240))

            stem = Path(rec["img_path"]).stem
            save_path = args.out / f"{k:02d}_{stem}.jpg"
            canvas.save(save_path, quality=85)

    print(f"Done -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
