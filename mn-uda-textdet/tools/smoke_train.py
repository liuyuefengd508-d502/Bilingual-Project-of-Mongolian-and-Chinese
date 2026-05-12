"""End-to-end MPS smoke test.

Verifies that:
    raw images + JSONL labels  ->  DataLoader  ->  ResNet18+head on MPS
    forward + BCE loss + backward + optimizer step all succeed.

Run from the repo root (mn-uda-textdet/):
    python tools/smoke_train.py \
        --ann splits/handwrite_train.json \
        --data-root .. \
        --iters 3 --batch 2 --size 320
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet18

# Make sibling `datasets/` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets import TextDetJsonDataset, collate  # noqa: E402


class TinySegHead(nn.Module):
    """ResNet18 backbone (no pretrained weights download here) + 1-channel head."""

    def __init__(self) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        # Drop avgpool + fc; keep through layer4 -> stride 32 feature.
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1  # 1/4
        self.layer2 = backbone.layer2  # 1/8
        self.layer3 = backbone.layer3  # 1/16
        self.layer4 = backbone.layer4  # 1/32
        self.head = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        logit = self.head(h)  # (B,1,H/32,W/32)
        # Upsample to input resolution for BCE against full-res mask.
        logit = F.interpolate(logit, size=x.shape[-2:], mode="bilinear",
                              align_corners=False)
        return logit.squeeze(1)


def pick_device(force_cpu: bool = False) -> torch.device:
    if not force_cpu and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", type=Path, default=Path("splits/handwrite_train.json"))
    ap.add_argument("--data-root", type=Path, default=Path(".."),
                    help="Anchor matching tools/build_splits.py --path-anchor.")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--num-workers", type=int, default=0,
                    help="Keep 0 on macOS to avoid fork issues with MPS.")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = pick_device(force_cpu=args.cpu)
    print(f"Device : {device}")
    print(f"Torch  : {torch.__version__}")

    ds = TextDetJsonDataset(args.ann, args.data_root, image_size=args.size)
    print(f"Dataset: {len(ds)} samples from {args.ann}")

    loader = DataLoader(
        ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate, drop_last=True,
    )

    model = TinySegHead().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    pos_weight = torch.tensor([5.0], device=device)

    model.train()
    t0 = time.time()
    for step, batch in enumerate(loader, start=1):
        if step > args.iters:
            break
        img = batch["image"].to(device)
        mask = batch["mask"].to(device)

        logits = model(img)                            # (B,H,W)
        loss = F.binary_cross_entropy_with_logits(
            logits, mask, pos_weight=pos_weight
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            pred = torch.sigmoid(logits) > 0.5
            pos = mask.sum().item()
            pred_pos = pred.sum().item()
        print(f"  step {step:>2}/{args.iters} "
              f"loss={loss.item():.4f} "
              f"mask_pos={pos:.0f} pred_pos={pred_pos:.0f}")

    dt = time.time() - t0
    print(f"\nOK. {args.iters} iters in {dt:.2f}s ({dt/args.iters:.2f}s/it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
