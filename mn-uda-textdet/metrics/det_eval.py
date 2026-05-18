"""Shared DBNet detection evaluation (H-mean on a labeled JSON loader)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import TextDetJsonDataset, collate
from models import batch_prob_to_polygons

from .hmean import HMeanEvaluator


def load_detection_ckpt(model: nn.Module, ckpt_path: Path, device: torch.device) -> None:
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(raw, dict):
        if "student" in raw:
            model.load_state_dict(raw["student"])
        elif "model" in raw:
            model.load_state_dict(raw["model"])
        else:
            model.load_state_dict(raw)
    else:
        model.load_state_dict(raw)


@torch.no_grad()
def hmean_on_loader(
    model: nn.Module,
    loader: DataLoader,
    ds: TextDetJsonDataset,
    device: torch.device,
    image_size: int,
    box_thresh: float,
    min_score: float,
    unclip_ratio: float,
    iou_thresh: float,
) -> dict[str, Any]:
    """Same protocol as ``train_dbnet._val_one_epoch`` / ``uda_eval_suite``."""
    model.eval()
    evaluator = HMeanEvaluator(iou_thresh=iou_thresh)
    sample_idx = 0
    for batch in loader:
        img = batch["image"].to(device)
        out = model(img, return_feat=False)
        probs = out["prob"].detach().cpu().numpy()
        preds = batch_prob_to_polygons(
            probs,
            box_thresh=box_thresh,
            min_score=min_score,
            unclip_ratio=unclip_ratio,
        )
        bsz = probs.shape[0]
        for b in range(bsz):
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


def hmean_on_ann_file(
    model: nn.Module,
    ann: Path,
    data_root: Path,
    device: torch.device,
    image_size: int,
    box_thresh: float,
    min_score: float,
    unclip_ratio: float,
    iou_thresh: float,
    batch: int,
    num_workers: int = 0,
) -> dict[str, Any]:
    ds = TextDetJsonDataset(
        ann, data_root, image_size=image_size, target_mode="dbnet", augment=False,
    )
    loader = DataLoader(
        ds, batch_size=batch, shuffle=False, num_workers=num_workers, collate_fn=collate,
        drop_last=False,
    )
    return hmean_on_loader(
        model, loader, ds, device, image_size,
        box_thresh, min_score, unclip_ratio, iou_thresh,
    )
