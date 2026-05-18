"""Compare two detection checkpoints on the same annotation files (H-mean delta)."""
from __future__ import annotations

import argparse
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
    ap.add_argument("--ckpt-a", type=Path, required=True, help="e.g. source-only baseline")
    ap.add_argument("--ckpt-b", type=Path, required=True, help="e.g. UDA student")
    ap.add_argument("--label-a", type=str, default="A")
    ap.add_argument("--label-b", type=str, default="B")
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
    )
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = pick_device(force_cpu=args.cpu)
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

    model_a = DBNet(backbone=args.backbone, pretrained=False).to(device)
    model_b = DBNet(backbone=args.backbone, pretrained=False).to(device)
    load_detection_ckpt(model_a, args.ckpt_a, device)
    load_detection_ckpt(model_b, args.ckpt_b, device)

    print(f"A={args.ckpt_a}\nB={args.ckpt_b}")
    print(f"post: box={args.box_thresh} min={args.min_score} iou={args.iou_thresh}")
    hdr = f"{'tag':<14} | {'H '+args.label_a:>8} {'H '+args.label_b:>8} | {'dH':>8} | {'dP':>8} | {'dR':>8}"
    print(hdr)
    print("-" * len(hdr))
    for spec in args.ann:
        if ":" in spec:
            ann_path, tag = spec.split(":", 1)
        else:
            ann_path, tag = spec, Path(spec).stem
        ann = Path(ann_path)
        ma = hmean_on_ann_file(model_a, ann, **kw)
        mb = hmean_on_ann_file(model_b, ann, **kw)
        d_h = float(mb["hmean"]) - float(ma["hmean"])
        d_p = float(mb["precision"]) - float(ma["precision"])
        d_r = float(mb["recall"]) - float(ma["recall"])
        print(
            f"{tag:<14} | {ma['hmean']:>8.4f} {mb['hmean']:>8.4f} | "
            f"{d_h:>+8.4f} | {d_p:>+8.4f} | {d_r:>+8.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
