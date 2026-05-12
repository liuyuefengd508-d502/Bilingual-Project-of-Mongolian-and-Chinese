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

from .dbnet_targets import generate_dbnet_targets


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TextDetJsonDataset(Dataset):
    """Read MMOCR-style JSON; img_path is relative to `data_root`.

    target_mode:
        'binary' : produce a single binary text mask (legacy / smoke test).
        'dbnet'  : produce DBNet supervision (gt, gt_mask, thresh_map, thresh_mask).
    """

    def __init__(
        self,
        ann_file: str | Path,
        data_root: str | Path,
        image_size: int = 640,
        ignore_flag_key: str = "ignore",
        target_mode: str = "binary",
        shrink_ratio: float = 0.4,
    ) -> None:
        self.ann_file = Path(ann_file)
        self.data_root = Path(data_root)
        self.image_size = int(image_size)
        self.ignore_flag_key = ignore_flag_key
        if target_mode not in ("binary", "dbnet"):
            raise ValueError(f"Unknown target_mode: {target_mode}")
        self.target_mode = target_mode
        self.shrink_ratio = float(shrink_ratio)

        with self.ann_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        self.data_list: Sequence[dict] = payload["data_list"]
        self.metainfo: dict = payload.get("metainfo", {})

    def __len__(self) -> int:
        return len(self.data_list)

    def _load_and_resize(self, rec: dict) -> tuple[Image.Image, list[np.ndarray], list[bool]]:
        img = Image.open((self.data_root / rec["img_path"]).resolve()).convert("RGB")
        orig_w, orig_h = img.size
        size = self.image_size
        img_resized = img.resize((size, size), Image.BILINEAR)
        sx = size / orig_w
        sy = size / orig_h
        polys: list[np.ndarray] = []
        igs: list[bool] = []
        for inst in rec.get("instances", []):
            poly = inst["polygon"]
            xy = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            xy[:, 0] *= sx
            xy[:, 1] *= sy
            polys.append(xy)
            igs.append(bool(inst.get(self.ignore_flag_key, False)))
        return img_resized, polys, igs

    def __getitem__(self, idx: int) -> dict:
        rec = self.data_list[idx]
        img_resized, polys, igs = self._load_and_resize(rec)
        size = self.image_size

        # Normalize image to tensor.
        img_arr = np.asarray(img_resized, dtype=np.float32) / 255.0
        for c, (m, s) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
            img_arr[..., c] = (img_arr[..., c] - m) / s
        img_t = torch.from_numpy(img_arr.transpose(2, 0, 1)).contiguous()

        out = {
            "image": img_t,
            "img_path": str((self.data_root / rec["img_path"]).resolve()),
            "orig_size": (rec.get("height", size), rec.get("width", size)),
        }

        if self.target_mode == "binary":
            mask = Image.new("L", (size, size), 0)
            draw = ImageDraw.Draw(mask)
            for poly, ig in zip(polys, igs):
                if ig or poly.shape[0] < 3:
                    continue
                draw.polygon([tuple(p) for p in poly], fill=1)
            out["mask"] = torch.from_numpy(np.asarray(mask, dtype=np.float32))
        else:  # dbnet
            tgt = generate_dbnet_targets(
                polys, igs, size, size, shrink_ratio=self.shrink_ratio,
            )
            out["gt"] = torch.from_numpy(tgt["gt"])
            out["gt_mask"] = torch.from_numpy(tgt["gt_mask"])
            out["thresh_map"] = torch.from_numpy(tgt["thresh_map"])
            out["thresh_mask"] = torch.from_numpy(tgt["thresh_mask"])

        return out


def collate(batch: list[dict]) -> dict:
    out: dict = {"image": torch.stack([b["image"] for b in batch], dim=0)}
    for key in ("mask", "gt", "gt_mask", "thresh_map", "thresh_mask"):
        if key in batch[0]:
            out[key] = torch.stack([b[key] for b in batch], dim=0)
    out["img_paths"] = [b["img_path"] for b in batch]
    out["orig_sizes"] = [b["orig_size"] for b in batch]
    return out
