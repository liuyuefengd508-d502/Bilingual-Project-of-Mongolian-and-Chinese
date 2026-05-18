"""Batch-evaluate a UDA / DBNet student checkpoint on multiple JSON splits.

Writes a small table to stdout and optional JSON for paper tables.

Example:
    python tools/uda_eval_suite.py --ckpt work_dirs/uda_s2h/best_student.pth \\
        --data-root .. --size 640 \\
        --ann splits/scene_test.json:scene_test splits/handwrite_test.json:handwrite_test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metrics import hmean_on_ann_file, load_detection_ckpt
from models import DBNet


def pick_device(force_cpu: bool = False) -> torch.device:
    if not force_cpu and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path(".."))
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--backbone", type=str, default="resnet18")
    ap.add_argument("--box-thresh", type=float, default=0.45)
    ap.add_argument("--min-score", type=float, default=0.5)
    ap.add_argument("--unclip-ratio", type=float, default=1.5)
    ap.add_argument("--iou-thresh", type=float, default=0.5)
    ap.add_argument(
        "--ann",
        type=str,
        nargs="+",
        default=[
            "splits/scene_test.json:scene_test",
            "splits/handwrite_test.json:handwrite_test",
        ],
        help="Each item is path[:tag]; tag defaults to stem of path.",
    )
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = pick_device(force_cpu=args.cpu)
    model = DBNet(backbone=args.backbone, pretrained=False).to(device)
    load_detection_ckpt(model, args.ckpt, device)

    kw = dict(
        data_root=args.data_root,
        device=device,
        image_size=args.size,
        box_thresh=args.box_thresh,
        min_score=args.min_score,
        unclip_ratio=args.unclip_ratio,
        iou_thresh=args.iou_thresh,
        batch=args.batch,
    )

    rows: list[dict[str, object]] = []
    print(f"ckpt={args.ckpt} device={device} post=box{args.box_thresh}/min{args.min_score}/u{args.unclip_ratio}/iou{args.iou_thresh}")
    print(f"{'tag':<16} {'P':>8} {'R':>8} {'H':>8}  ann")
    print("-" * 72)
    for spec in args.ann:
        if ":" in spec:
            ann_path, tag = spec.split(":", 1)
        else:
            ann_path, tag = spec, Path(spec).stem
        ann = Path(ann_path)
        m = hmean_on_ann_file(model, ann, **kw)
        row = {"tag": tag, "ann": str(ann), **{k: float(v) if isinstance(v, (float, int)) else v for k, v in m.items()}}
        rows.append(row)
        print(f"{tag:<16} {m['precision']:.4f} {m['recall']:.4f} {m['hmean']:.4f}  {ann}")

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps({"ckpt": str(args.ckpt), "runs": rows}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nWrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
