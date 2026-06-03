"""Run DBNet forward once, then sweep box_thresh / min_score (post-process only).

Avoids reloading the model and recomputing conv features for each threshold pair.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets import TextDetJsonDataset, collate
from metrics import HMeanEvaluator
from models import DBNet, batch_prob_to_polygons


def pick_device(force_cpu: bool = False) -> torch.device:
    if not force_cpu and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--ann", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path(".."))
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--backbone", type=str, default="resnet18")
    ap.add_argument("--iou-thresh", type=float, default=0.5)
    ap.add_argument("--unclip-ratio", type=float, default=1.5)
    ap.add_argument("--box-thresh-grid", type=str, default="0.30,0.35,0.40,0.45,0.50,0.55")
    ap.add_argument("--min-score-grid", type=str, default="0.40,0.45,0.50,0.55,0.60")
    ap.add_argument("--out", type=Path, default=None, help="Write JSON with all grid results.")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    box_list = [float(x) for x in args.box_thresh_grid.split(",")]
    ms_list = [float(x) for x in args.min_score_grid.split(",")]

    device = pick_device(force_cpu=args.cpu)
    ds = TextDetJsonDataset(args.ann, args.data_root,
                            image_size=args.size, target_mode="dbnet")
    loader = DataLoader(
        ds, batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=collate,
        drop_last=False,
    )

    model = DBNet(backbone=args.backbone, pretrained=False).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    weights = state.get("student") or state.get("model") or state
    model.load_state_dict(weights)
    model.eval()

    # Cache probability maps + per-sample GT in network input scale.
    all_probs: list[np.ndarray] = []
    per_sample_gt: list[tuple[list[np.ndarray], list[bool]]] = []
    sample_idx = 0
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            out = model(img)
            probs = out["prob"].detach().cpu().numpy()
            B = probs.shape[0]
            for b in range(B):
                all_probs.append(probs[b])
                rec = ds.data_list[sample_idx]
                sample_idx += 1
                orig_w = rec.get("width", args.size)
                orig_h = rec.get("height", args.size)
                sx = args.size / orig_w
                sy = args.size / orig_h
                gt_polys: list[np.ndarray] = []
                gt_ignore: list[bool] = []
                for inst in rec.get("instances", []):
                    p = np.asarray(inst["polygon"], dtype=np.float32).reshape(-1, 2)
                    p[:, 0] *= sx
                    p[:, 1] *= sy
                    gt_polys.append(p)
                    gt_ignore.append(bool(inst.get("ignore", False)))
                per_sample_gt.append((gt_polys, gt_ignore))

    grid_results: list[dict] = []
    for bt in box_list:
        for ms in ms_list:
            ev = HMeanEvaluator(iou_thresh=args.iou_thresh)
            for prob, (gt_polys, gt_ignore) in zip(all_probs, per_sample_gt):
                preds = batch_prob_to_polygons(
                    prob[np.newaxis, ...],
                    box_thresh=bt, min_score=ms, unclip_ratio=args.unclip_ratio,
                )[0]
                pred_polys = [pp for pp, _ in preds]
                pred_scores = [s for _, s in preds]
                ev.add(gt_polys, gt_ignore, pred_polys, pred_scores)
            m = ev.compute()
            grid_results.append({
                "box_thresh": bt, "min_score": ms, **{k: float(v) if isinstance(v, (float, np.floating)) else v
                                                          for k, v in m.items()},
            })

    grid_results.sort(key=lambda r: r["hmean"], reverse=True)
    print(f"Device: {device} | samples: {len(all_probs)} | grid: {len(grid_results)} combos")
    print("Top 8 by H-mean:")
    for r in grid_results[:8]:
        print(f"  H={r['hmean']:.4f} P={r['precision']:.4f} R={r['recall']:.4f} "
              f"box_thresh={r['box_thresh']} min_score={r['min_score']}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ckpt": str(args.ckpt),
            "ann": str(args.ann),
            "size": args.size,
            "unclip_ratio": args.unclip_ratio,
            "iou_thresh": args.iou_thresh,
            "sorted_by_hmean_desc": grid_results,
        }
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
