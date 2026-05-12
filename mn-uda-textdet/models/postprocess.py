"""DBNet post-processing: probability map -> polygons.

Pipeline (per image):
    1. Threshold the probability map at `box_thresh` to get a binary mask.
    2. Find external contours (cv2.findContours).
    3. Filter by min size / min mean score.
    4. Expand each contour back to text region (inverse of DBNet shrink).
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon


def _unclip(poly: np.ndarray, unclip_ratio: float = 1.5) -> np.ndarray | None:
    p = Polygon(poly)
    if not p.is_valid or p.area <= 0:
        return None
    distance = p.area * unclip_ratio / p.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(poly.astype(np.int64).tolist(),
                   pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    res = offset.Execute(distance)
    if not res:
        return None
    return np.asarray(res[0], dtype=np.int32)


def _box_score(prob: np.ndarray, contour: np.ndarray) -> float:
    h, w = prob.shape
    xmin = max(0, int(contour[:, 0].min()))
    xmax = min(w - 1, int(contour[:, 0].max()))
    ymin = max(0, int(contour[:, 1].min()))
    ymax = min(h - 1, int(contour[:, 1].max()))
    if xmax <= xmin or ymax <= ymin:
        return 0.0
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    local = contour.copy()
    local[:, 0] -= xmin
    local[:, 1] -= ymin
    cv2.fillPoly(mask, [local.astype(np.int32)], 1)
    return float(cv2.mean(prob[ymin:ymax + 1, xmin:xmax + 1], mask=mask)[0])


def prob_to_polygons(
    prob: np.ndarray,
    box_thresh: float = 0.3,
    min_score: float = 0.6,
    min_size: int = 3,
    unclip_ratio: float = 1.5,
    max_candidates: int = 1000,
) -> list[tuple[np.ndarray, float]]:
    """Return list of (polygon (N,2) int32, score)."""
    h, w = prob.shape
    bitmap = (prob > box_thresh).astype(np.uint8)
    contours, _ = cv2.findContours(bitmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[tuple[np.ndarray, float]] = []
    for cnt in contours[:max_candidates]:
        if cnt.shape[0] < 4:
            continue
        cnt = cnt.reshape(-1, 2)
        if min(cnt[:, 0].max() - cnt[:, 0].min(),
               cnt[:, 1].max() - cnt[:, 1].min()) < min_size:
            continue
        score = _box_score(prob, cnt)
        if score < min_score:
            continue
        expanded = _unclip(cnt, unclip_ratio=unclip_ratio)
        if expanded is None or expanded.shape[0] < 3:
            continue
        out.append((expanded, score))
    return out


def batch_prob_to_polygons(
    probs: np.ndarray,
    **kwargs,
) -> list[list[tuple[np.ndarray, float]]]:
    """probs: (B,H,W) numpy array in [0,1]."""
    return [prob_to_polygons(p, **kwargs) for p in probs]
