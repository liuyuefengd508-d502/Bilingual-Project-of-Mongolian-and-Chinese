"""Minimal PyTorch Dataset for 蒙汉双语 text detection.

Reads MMOCR-style JSON produced by tools/build_splits.py and yields
(image_tensor, text_mask) pairs at a fixed resolution.

Notes:
- Does NOT depend on MMOCR/MMCV; uses only PIL + NumPy + torch.
- Polygons are rasterized to a binary mask (1 = text, 0 = bg) at the same
  resolution as the (resized) image. This is a placeholder supervision used by
  the smoke test; DBNet++ shrink / threshold maps will be added at W2 baseline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TextDetJsonDataset(Dataset):
    """Read MMOCR-style JSON; img_path is relative to `data_root`."""

    def __init__(
        self,
        ann_file: str | Path,
        data_root: str | Path,
        image_size: int = 640,
        ignore_flag_key: str = "ignore",
    ) -> None:
        self.ann_file = Path(ann_file)
        self.data_root = Path(data_root)
        self.image_size = int(image_size)
        self.ignore_flag_key = ignore_flag_key

        with self.ann_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        self.data_list: Sequence[dict] = payload["data_list"]
        self.metainfo: dict = payload.get("metainfo", {})

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> dict:
        rec = self.data_list[idx]
        img_path = (self.data_root / rec["img_path"]).resolve()
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # Resize image to square (letterbox-free; simple resize for smoke test).
        size = self.image_size
        img_resized = img.resize((size, size), Image.BILINEAR)

        # Rasterize polygons into a binary mask at target resolution.
        mask = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(mask)
        sx = size / orig_w
        sy = size / orig_h
        for inst in rec.get("instances", []):
            if inst.get(self.ignore_flag_key, False):
                continue
            poly = inst["polygon"]
            xy = [
                (poly[i] * sx, poly[i + 1] * sy)
                for i in range(0, len(poly), 2)
            ]
            if len(xy) >= 3:
                draw.polygon(xy, fill=1)

        # To tensors.
        img_arr = np.asarray(img_resized, dtype=np.float32) / 255.0
        for c, (m, s) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
            img_arr[..., c] = (img_arr[..., c] - m) / s
        img_t = torch.from_numpy(img_arr.transpose(2, 0, 1)).contiguous()  # (3,H,W)
        mask_t = torch.from_numpy(np.asarray(mask, dtype=np.float32))      # (H,W)

        return {
            "image": img_t,
            "mask": mask_t,
            "img_path": str(img_path),
            "orig_size": (orig_h, orig_w),
        }


def collate(batch: list[dict]) -> dict:
    images = torch.stack([b["image"] for b in batch], dim=0)
    masks = torch.stack([b["mask"] for b in batch], dim=0)
    return {
        "image": images,
        "mask": masks,
        "img_paths": [b["img_path"] for b in batch],
        "orig_sizes": [b["orig_size"] for b in batch],
    }
