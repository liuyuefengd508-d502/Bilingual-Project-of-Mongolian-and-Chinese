"""Standalone evaluator: load a DBNet checkpoint, run on a JSON ann file,
report Precision / Recall / H-mean.

Useful for cross-domain "source-only" lower bound:
    # Train on scene
    python tools/train_dbnet.py --train splits/scene_train.json \
        --val splits/scene_val.json --out work_dirs/scene_oracle ...

    # Eval on handwrite test (no retraining)
    python tools/eval.py --ckpt work_dirs/scene_oracle/best.pth \
        --ann splits/handwrite_test.json --data-root .. --size 640
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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


def evaluate(
    model: torch.nn.Module,
    ds: TextDetJsonDataset,
    loader: DataLoader,
    device: torch.device,
    image_size: int,
    box_thresh: float = 0.3,
    min_score: float = 0.6,
    unclip_ratio: float = 1.5,
    iou_thresh: float = 0.5,
) -> dict:
    model.eval()
    evaluator = HMeanEvaluator(iou_thresh=iou_thresh)
    sample_idx = 0
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            out = model(img)
            probs = out["prob"].detach().cpu().numpy()
            preds = batch_prob_to_polygons(
                probs, box_thresh=box_thresh, min_score=min_score,
                unclip_ratio=unclip_ratio,
            )
            B = probs.shape[0]
            for b in range(B):
                rec = ds.data_list[sample_idx]
                sample_idx += 1
                orig_w = rec.get("width", image_size)
                orig_h = rec.get("height", image_size)
                sx = image_size / orig_w
                sy = image_size / orig_h
                gt_polys: list[np.ndarray] = []
                gt_ignore: list[bool] = []
                for inst in rec.get("instances", []):
                    p = np.asarray(inst["polygon"], dtype=np.float32).reshape(-1, 2)
                    p[:, 0] *= sx
                    p[:, 1] *= sy
                    gt_polys.append(p)
                    gt_ignore.append(bool(inst.get("ignore", False)))
                pred_polys = [pp for pp, _ in preds[b]]
                pred_scores = [s for _, s in preds[b]]
                evaluator.add(gt_polys, gt_ignore, pred_polys, pred_scores)
    return evaluator.compute()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--ann", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path(".."))
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--backbone", type=str, default="resnet18")
    ap.add_argument("--box-thresh", type=float, default=0.3)
    ap.add_argument("--min-score", type=float, default=0.6)
    ap.add_argument("--unclip-ratio", type=float, default=1.5)
    ap.add_argument("--iou-thresh", type=float, default=0.5)
    ap.add_argument("--save", type=Path, default=None,
                    help="Optional path to write metrics JSON.")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = pick_device(force_cpu=args.cpu)
    print(f"Device : {device}")
    print(f"Ckpt   : {args.ckpt}")
    print(f"Ann    : {args.ann}")

    ds = TextDetJsonDataset(args.ann, args.data_root,
                            image_size=args.size, target_mode="dbnet")
    loader = DataLoader(
        ds, batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate, drop_last=False,
    )
    print(f"Samples: {len(ds)}")

    model = DBNet(backbone=args.backbone, pretrained=False).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    print(f"Loaded ckpt epoch={state.get('epoch', '?')} "
          f"train_metrics={state.get('metrics', {})}")

    t0 = time.time()
    metrics = evaluate(
        model, ds, loader, device, args.size,
        box_thresh=args.box_thresh, min_score=args.min_score,
        unclip_ratio=args.unclip_ratio, iou_thresh=args.iou_thresh,
    )
    dt = time.time() - t0
    print(f"\n=== Result ({dt:.1f}s) ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:>10}: {v:.4f}")
        else:
            print(f"  {k:>10}: {v}")

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ckpt": str(args.ckpt),
            "ann": str(args.ann),
            "size": args.size,
            "post": {
                "box_thresh": args.box_thresh,
                "min_score": args.min_score,
                "unclip_ratio": args.unclip_ratio,
                "iou_thresh": args.iou_thresh,
            },
            "metrics": metrics,
        }
        args.save.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        print(f"\nSaved -> {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
