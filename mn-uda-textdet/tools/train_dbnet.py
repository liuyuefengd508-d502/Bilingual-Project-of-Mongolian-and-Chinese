"""Train DBNet (ResNet18+FPN) on one domain. Single-domain baseline (oracle).

Run example:
    python tools/train_dbnet.py \
        --train splits/scene_train.json \
        --val   splits/scene_val.json \
        --data-root .. --epochs 20 --batch 4 --size 640 \
        --out work_dirs/scene_dbnet_r18
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
from models import DBNet, db_loss, batch_prob_to_polygons


def pick_device(force_cpu: bool = False) -> torch.device:
    if not force_cpu and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_loaders(args) -> tuple[DataLoader, DataLoader, TextDetJsonDataset, TextDetJsonDataset]:
    train_ds = TextDetJsonDataset(args.train, args.data_root,
                                  image_size=args.size, target_mode="dbnet")
    val_ds = TextDetJsonDataset(args.val, args.data_root,
                                image_size=args.size, target_mode="dbnet")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate, drop_last=False,
    )
    return train_loader, val_loader, train_ds, val_ds


def _val_one_epoch(model: torch.nn.Module, loader: DataLoader, val_ds: TextDetJsonDataset,
                   device: torch.device, args) -> dict:
    model.eval()
    evaluator = HMeanEvaluator(iou_thresh=args.iou_thresh)
    # Pre-index ground-truth polygons (already resized to args.size by Dataset).
    # We re-iterate val_ds.data_list in the same order as the loader (shuffle=False).
    sample_idx = 0
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            out = model(img)
            probs = out["prob"].detach().cpu().numpy()
            preds = batch_prob_to_polygons(
                probs, box_thresh=args.box_thresh, min_score=args.min_score,
                unclip_ratio=args.unclip_ratio,
            )
            B = probs.shape[0]
            for b in range(B):
                rec = val_ds.data_list[sample_idx]
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
                pred_polys = [pp for pp, _ in preds[b]]
                pred_scores = [s for _, s in preds[b]]
                evaluator.add(gt_polys, gt_ignore, pred_polys, pred_scores)
    return evaluator.compute()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path(".."))
    ap.add_argument("--out", type=Path, default=Path("work_dirs/run"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--backbone", type=str, default="resnet18")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--box-thresh", type=float, default=0.3)
    ap.add_argument("--min-score", type=float, default=0.6)
    ap.add_argument("--unclip-ratio", type=float, default=1.5)
    ap.add_argument("--iou-thresh", type=float, default=0.5)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = pick_device(force_cpu=args.cpu)
    print(f"Device : {device}")
    print(f"Out    : {args.out.resolve()}")

    train_loader, val_loader, _, val_ds = _build_loaders(args)
    print(f"Train  : {len(train_loader.dataset)} | Val: {len(val_ds)}")

    model = DBNet(backbone=args.backbone, pretrained=args.pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    history: list[dict] = []
    best_hmean = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = {"loss": 0.0, "n": 0}
        for step, batch in enumerate(train_loader, start=1):
            img = batch["image"].to(device)
            target = {k: batch[k].to(device)
                      for k in ("gt", "gt_mask", "thresh_map", "thresh_mask")}
            out = model(img)
            losses = db_loss(out, target)
            opt.zero_grad()
            losses["loss"].backward()
            opt.step()
            running["loss"] += losses["loss"].item() * img.shape[0]
            running["n"] += img.shape[0]
            if step % args.log_every == 0:
                print(f"  ep{epoch:>3} st{step:>4} loss={losses['loss'].item():.4f}")
        sched.step()
        avg_loss = running["loss"] / max(1, running["n"])
        dt_train = time.time() - t0

        t1 = time.time()
        metrics = _val_one_epoch(model, val_loader, val_ds, device, args)
        dt_val = time.time() - t1

        entry = {
            "epoch": epoch, "lr": opt.param_groups[0]["lr"],
            "train_loss": avg_loss,
            "train_sec": dt_train, "val_sec": dt_val,
            **metrics,
        }
        history.append(entry)
        print(f"[ep {epoch:>3}/{args.epochs}] loss={avg_loss:.4f} "
              f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
              f"H={metrics['hmean']:.4f} "
              f"(train {dt_train:.1f}s, val {dt_val:.1f}s)")

        # Save history each epoch.
        (args.out / "history.json").write_text(
            json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
        # Save latest + best.
        torch.save({"model": model.state_dict(), "epoch": epoch,
                    "args": vars(args), "metrics": metrics},
                   args.out / "latest.pth")
        if metrics["hmean"] > best_hmean:
            best_hmean = metrics["hmean"]
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "args": vars(args), "metrics": metrics},
                       args.out / "best.pth")
            print(f"  -> new best H-mean {best_hmean:.4f} (saved best.pth)")

    print(f"\nDone. Best H-mean = {best_hmean:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
