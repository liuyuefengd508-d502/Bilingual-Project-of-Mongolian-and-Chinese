"""DBNet smoke test: full DB targets + DBNet head + 3-term loss, on MPS.

Run:
    python tools/smoke_dbnet.py --ann splits/handwrite_train.json \
        --data-root .. --iters 3 --batch 2 --size 320
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets import TextDetJsonDataset, collate
from models import DBNet, db_loss


def pick_device(force_cpu: bool = False) -> torch.device:
    if not force_cpu and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", type=Path, default=Path("splits/handwrite_train.json"))
    ap.add_argument("--data-root", type=Path, default=Path(".."))
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = pick_device(force_cpu=args.cpu)
    print(f"Device : {device}, Torch: {torch.__version__}")

    ds = TextDetJsonDataset(
        args.ann, args.data_root, image_size=args.size, target_mode="dbnet",
    )
    print(f"Dataset: {len(ds)} samples from {args.ann}")

    loader = DataLoader(
        ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate, drop_last=True,
    )

    model = DBNet(backbone="resnet18", pretrained=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    t0 = time.time()
    for step, batch in enumerate(loader, start=1):
        if step > args.iters:
            break
        img = batch["image"].to(device)
        target = {
            "gt": batch["gt"].to(device),
            "gt_mask": batch["gt_mask"].to(device),
            "thresh_map": batch["thresh_map"].to(device),
            "thresh_mask": batch["thresh_mask"].to(device),
        }

        out = model(img)
        losses = db_loss(out, target)

        opt.zero_grad()
        losses["loss"].backward()
        opt.step()

        with torch.no_grad():
            pred_bin = (out["binary"] > 0.5).float().sum().item()
            gt_pos = target["gt"].sum().item()
        print(f"  step {step:>2}/{args.iters} "
              f"loss={losses['loss'].item():.4f} "
              f"(p={losses['loss_prob'].item():.3f}, "
              f"b={losses['loss_binary'].item():.3f}, "
              f"t={losses['loss_thresh'].item():.3f}) "
              f"gt_pos={gt_pos:.0f} pred_pos={pred_bin:.0f}")

    dt = time.time() - t0
    print(f"\nOK. {args.iters} iters in {dt:.2f}s ({dt/args.iters:.2f}s/it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
