"""DBNet-style label generation.

For each text polygon (resized into the network input space):
    shrink_mask : 1 inside shrunk polygon, 0 outside  (a.k.a. probability map)
    threshold_map : distance-based map in [thresh_min, thresh_max] in a band
                    of width `shrink_ratio` around the polygon boundary,
                    used to supervise the threshold branch (DBNet++ paper).
    threshold_mask : 1 where threshold loss is computed (the band region).

Refs:
    Liao et al., "Real-time Scene Text Detection with Differentiable
    Binarization" (AAAI 2020) and its TPAMI 2022 extension (DBNet++).
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon


def _polygon_area_perimeter(poly: np.ndarray) -> tuple[float, float]:
    p = Polygon(poly)
    if not p.is_valid or p.area <= 0:
        return 0.0, 0.0
    return float(p.area), float(p.length)


def _offset_polygon(poly: np.ndarray, distance: float) -> list[np.ndarray]:
    """Shrink (distance<0 effectively via JT_MITER) or expand polygon."""
    subj = poly.astype(np.int64).tolist()
    pc = pyclipper.PyclipperOffset()
    pc.AddPath(subj, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    res = pc.Execute(distance)
    return [np.asarray(p, dtype=np.int32) for p in res]


def generate_dbnet_targets(
    polygons: Sequence[np.ndarray],
    ignore_flags: Sequence[bool],
    height: int,
    width: int,
    shrink_ratio: float = 0.4,
    thresh_min: float = 0.3,
    thresh_max: float = 0.7,
    min_polygon_size: float = 8.0,
) -> dict[str, np.ndarray]:
    """Build DBNet supervision maps for one image.

    Args:
        polygons: list of (N_i, 2) int/float arrays, in resized image coords.
        ignore_flags: same length as polygons; True -> exclude from positives,
                      mark in ignore mask.
        height, width: target map resolution (= input resolution).
        shrink_ratio: D = A * (1 - r^2) / L (DBNet eq.).
        thresh_min, thresh_max: range of threshold map values in the border band.
        min_polygon_size: skip tiny polygons (in resized pixels).

    Returns:
        dict with:
            'gt'        : (H,W) float32 in {0,1}  - shrunk text mask
            'gt_mask'   : (H,W) float32 in {0,1}  - 1 where loss is computed
            'thresh_map': (H,W) float32 in [thresh_min, thresh_max]
            'thresh_mask': (H,W) float32 in {0,1} - band region
    """
    gt = np.zeros((height, width), dtype=np.float32)
    gt_mask = np.ones((height, width), dtype=np.float32)
    thresh_map = np.zeros((height, width), dtype=np.float32)
    thresh_mask = np.zeros((height, width), dtype=np.float32)

    for poly, ig in zip(polygons, ignore_flags):
        poly = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        if poly.shape[0] < 3:
            continue
        # Clip to image bounds.
        poly[:, 0] = np.clip(poly[:, 0], 0, width - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, height - 1)
        area, perimeter = _polygon_area_perimeter(poly)
        if area <= 0 or perimeter <= 0 or min(*np.ptp(poly, axis=0)) < min_polygon_size:
            cv2.fillPoly(gt_mask, [poly.astype(np.int32)], 0.0)
            continue
        if ig:
            cv2.fillPoly(gt_mask, [poly.astype(np.int32)], 0.0)
            continue

        # DBNet shrink distance.
        distance = area * (1 - shrink_ratio ** 2) / perimeter
        shrunk = _offset_polygon(poly, -distance)
        if not shrunk:
            cv2.fillPoly(gt_mask, [poly.astype(np.int32)], 0.0)
            continue
        for sp in shrunk:
            cv2.fillPoly(gt, [sp.astype(np.int32)], 1.0)

        # Threshold map: band between shrunk and expanded polygons.
        expanded = _offset_polygon(poly, +distance)
        if not expanded:
            continue
        band = expanded[0]
        # Local ROI to limit distance computation cost.
        xmin, ymin = band.min(axis=0)
        xmax, ymax = band.max(axis=0)
        xmin = max(0, int(xmin) - 1)
        ymin = max(0, int(ymin) - 1)
        xmax = min(width - 1, int(xmax) + 1)
        ymax = min(height - 1, int(ymax) + 1)
        if xmax <= xmin or ymax <= ymin:
            continue

        # Distance from each pixel in ROI to the polygon edges.
        roi_w = xmax - xmin + 1
        roi_h = ymax - ymin + 1
        local_poly = poly.copy()
        local_poly[:, 0] -= xmin
        local_poly[:, 1] -= ymin

        ys, xs = np.mgrid[0:roi_h, 0:roi_w]
        pts = np.stack([xs, ys], axis=-1).astype(np.float32)  # (h,w,2)
        # Edge segments.
        seg_a = local_poly
        seg_b = np.roll(local_poly, -1, axis=0)
        # Compute min distance from each point to all segments.
        d = _distance_to_segments(pts, seg_a, seg_b)  # (h,w)
        # Normalize so that boundary -> 1, shrink/expand line -> 0.
        d_norm = 1.0 - np.clip(d / distance, 0.0, 1.0)
        d_norm = thresh_min + (thresh_max - thresh_min) * d_norm

        # Apply only within band mask.
        band_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        band_local = band.copy()
        band_local[:, 0] -= xmin
        band_local[:, 1] -= ymin
        cv2.fillPoly(band_mask, [band_local.astype(np.int32)], 1)
        # Subtract shrunk interior so band is only the border ring.
        for sp in shrunk:
            sp_local = sp.astype(np.int32).copy()
            sp_local[:, 0] -= xmin
            sp_local[:, 1] -= ymin
            cv2.fillPoly(band_mask, [sp_local], 0)

        sel = band_mask.astype(bool)
        tm_slice = thresh_map[ymin:ymax + 1, xmin:xmax + 1]
        msk_slice = thresh_mask[ymin:ymax + 1, xmin:xmax + 1]
        tm_slice[sel] = np.maximum(tm_slice[sel], d_norm[sel])
        msk_slice[sel] = 1.0

    return {
        "gt": gt,
        "gt_mask": gt_mask,
        "thresh_map": thresh_map,
        "thresh_mask": thresh_mask,
    }


def _distance_to_segments(
    pts: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray
) -> np.ndarray:
    """Min distance from each (h,w,2) point to the closest of N segments.

    Args:
        pts: (H, W, 2)
        seg_a, seg_b: (N, 2)
    Returns:
        (H, W) min distance.
    """
    H, W, _ = pts.shape
    flat = pts.reshape(-1, 2)                # (HW, 2)
    ab = seg_b - seg_a                       # (N, 2)
    ab_len2 = (ab ** 2).sum(axis=1)          # (N,)
    # For each point p and each segment: t = ((p-a) . ab) / |ab|^2 clipped.
    # Distance = ||p - (a + t*ab)||.
    best = np.full((flat.shape[0],), np.inf, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for i in range(seg_a.shape[0]):
            if ab_len2[i] < 1e-6:
                # Degenerate segment: treat as a point.
                d = np.linalg.norm(flat - seg_a[i], axis=1)
            else:
                ap = flat - seg_a[i]                       # (HW, 2)
                t = (ap @ ab[i]) / ab_len2[i]              # (HW,)
                t = np.clip(t, 0.0, 1.0)
                proj = seg_a[i] + t[:, None] * ab[i]       # (HW, 2)
                d = np.linalg.norm(flat - proj, axis=1)    # (HW,)
            np.minimum(best, d, out=best)
    return best.reshape(H, W)
