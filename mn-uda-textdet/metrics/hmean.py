"""ICDAR-style H-mean evaluator for polygon text detection.

Match rule:
    A predicted polygon is a true positive iff its IoU with some GT polygon
    is >= iou_thresh, with one-to-one greedy matching (highest IoU first).
    GT polygons marked as `ignore` are excluded from matching but predictions
    overlapping them are not counted as FP if IoU>=iou_thresh.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from shapely.geometry import Polygon


def _to_shapely(poly: np.ndarray) -> Polygon | None:
    if poly is None or len(poly) < 3:
        return None
    p = Polygon(np.asarray(poly, dtype=np.float64).reshape(-1, 2))
    if not p.is_valid:
        p = p.buffer(0)
    if p.is_empty or p.area <= 0:
        return None
    return p


def polygon_iou(a: np.ndarray, b: np.ndarray) -> float:
    pa = _to_shapely(a)
    pb = _to_shapely(b)
    if pa is None or pb is None:
        return 0.0
    inter = pa.intersection(pb).area
    if inter <= 0:
        return 0.0
    return inter / (pa.area + pb.area - inter + 1e-9)


class HMeanEvaluator:
    def __init__(self, iou_thresh: float = 0.5) -> None:
        self.iou_thresh = float(iou_thresh)
        self.reset()

    def reset(self) -> None:
        self._tp = 0
        self._fp = 0
        self._fn = 0
        self._n_pred = 0
        self._n_gt = 0

    def add(
        self,
        gt_polys: Sequence[np.ndarray],
        gt_ignore: Sequence[bool],
        pred_polys: Sequence[np.ndarray],
        pred_scores: Sequence[float],
    ) -> None:
        gt = [_to_shapely(p) for p in gt_polys]
        pred = [(s, _to_shapely(p)) for p, s in zip(pred_polys, pred_scores)]
        pred = [(s, p) for s, p in pred if p is not None]
        pred.sort(key=lambda x: -x[0])  # high score first

        gt_used = [False] * len(gt)
        active_gt_count = sum(
            1 for g, ig in zip(gt, gt_ignore) if g is not None and not ig
        )
        self._n_gt += active_gt_count
        self._n_pred += len(pred)

        for _, pp in pred:
            best_iou = 0.0
            best_idx = -1
            for j, gg in enumerate(gt):
                if gg is None or gt_used[j]:
                    continue
                inter = pp.intersection(gg).area
                if inter <= 0:
                    continue
                iou = inter / (pp.area + gg.area - inter + 1e-9)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j
            if best_idx >= 0 and best_iou >= self.iou_thresh:
                gt_used[best_idx] = True
                if not gt_ignore[best_idx]:
                    self._tp += 1
                # If matched to an ignore GT, neither TP nor FP.
            else:
                self._fp += 1

        # Unmatched non-ignore GTs are FN.
        for used, ig in zip(gt_used, gt_ignore):
            if not used and not ig:
                self._fn += 1

    def compute(self) -> dict[str, float]:
        tp, fp, fn = self._tp, self._fp, self._fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        hmean = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
        return {
            "precision": precision,
            "recall": recall,
            "hmean": hmean,
            "tp": tp, "fp": fp, "fn": fn,
            "n_pred": self._n_pred, "n_gt": self._n_gt,
        }
