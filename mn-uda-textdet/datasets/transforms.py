"""Detection-aware augmentations: rotate / scale / crop / color jitter.

All operations work on (PIL.Image RGB, list[np.ndarray polygons]) and keep
polygons in image pixel coordinates. The final crop produces an output of
exactly `out_size x out_size`, matching the network input resolution.

Notes:
    - No horizontal/vertical flip by default: Mongolian script is direction-
      sensitive (vertical, top-to-bottom).
    - Polygons are NOT clipped against image bounds inside this module; the
      Dataset's target generator clips them when rasterizing.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image, ImageEnhance


@dataclass
class AugmentConfig:
    out_size: int = 640
    rotate_deg: float = 10.0          # +/- range
    scale_min: float = 0.5
    scale_max: float = 2.0
    color_brightness: float = 0.2
    color_contrast: float = 0.2
    pad_value: tuple[int, int, int] = (114, 114, 114)


def _rotate_polys(polys: list[np.ndarray], angle_deg: float,
                  cx: float, cy: float) -> list[np.ndarray]:
    if angle_deg == 0.0:
        return polys
    rad = math.radians(angle_deg)
    cos, sin = math.cos(rad), math.sin(rad)
    M = np.array([[cos, -sin], [sin, cos]], dtype=np.float32)
    out: list[np.ndarray] = []
    for p in polys:
        shifted = p - np.array([cx, cy], dtype=np.float32)
        rot = shifted @ M.T
        out.append(rot + np.array([cx, cy], dtype=np.float32))
    return out


def _scale_polys(polys: list[np.ndarray], sx: float, sy: float) -> list[np.ndarray]:
    if sx == 1.0 and sy == 1.0:
        return polys
    s = np.array([sx, sy], dtype=np.float32)
    return [p * s for p in polys]


def _translate_polys(polys: list[np.ndarray], dx: float, dy: float) -> list[np.ndarray]:
    if dx == 0.0 and dy == 0.0:
        return polys
    t = np.array([dx, dy], dtype=np.float32)
    return [p + t for p in polys]


def random_augment(
    img: Image.Image,
    polys: list[np.ndarray],
    ignore_flags: list[bool],
    cfg: AugmentConfig,
    rng: random.Random,
) -> tuple[Image.Image, list[np.ndarray], list[bool]]:
    W, H = img.size
    cx, cy = W / 2.0, H / 2.0

    # 1) Random rotation around image center.
    angle = rng.uniform(-cfg.rotate_deg, cfg.rotate_deg)
    if abs(angle) > 1e-3:
        img = img.rotate(angle, resample=Image.BILINEAR, expand=False,
                         fillcolor=cfg.pad_value)
        # PIL rotates by `angle` degrees counter-clockwise; we apply the
        # corresponding inverse to polygon points so they follow the image.
        polys = _rotate_polys(polys, -angle, cx, cy)

    # 2) Color jitter (brightness + contrast).
    if cfg.color_brightness > 0:
        f = 1.0 + rng.uniform(-cfg.color_brightness, cfg.color_brightness)
        img = ImageEnhance.Brightness(img).enhance(f)
    if cfg.color_contrast > 0:
        f = 1.0 + rng.uniform(-cfg.color_contrast, cfg.color_contrast)
        img = ImageEnhance.Contrast(img).enhance(f)

    # 3) Random scale (uniform in [scale_min, scale_max]).
    s = rng.uniform(cfg.scale_min, cfg.scale_max)
    new_W = max(1, int(round(W * s)))
    new_H = max(1, int(round(H * s)))
    img = img.resize((new_W, new_H), Image.BILINEAR)
    polys = _scale_polys(polys, s, s)
    W, H = new_W, new_H

    # 4) Random crop / pad to out_size square.
    out = cfg.out_size
    canvas = Image.new("RGB", (out, out), cfg.pad_value)
    if W <= out and H <= out:
        # Pad: place image at random position inside canvas.
        ox = rng.randint(0, out - W)
        oy = rng.randint(0, out - H)
        canvas.paste(img, (ox, oy))
        polys = _translate_polys(polys, ox, oy)
    else:
        # Crop a out_size window.
        max_x = max(0, W - out)
        max_y = max(0, H - out)
        # Bias crop toward containing some text by sampling around a random
        # polygon centroid 50% of the time.
        if polys and rng.random() < 0.5:
            target = polys[rng.randrange(len(polys))]
            txc, tyc = float(target[:, 0].mean()), float(target[:, 1].mean())
            ox = int(np.clip(txc - out / 2, 0, max_x))
            oy = int(np.clip(tyc - out / 2, 0, max_y))
        else:
            ox = rng.randint(0, max_x) if max_x > 0 else 0
            oy = rng.randint(0, max_y) if max_y > 0 else 0
        right = min(W, ox + out)
        bottom = min(H, oy + out)
        crop = img.crop((ox, oy, right, bottom))
        canvas.paste(crop, (0, 0))
        polys = _translate_polys(polys, -ox, -oy)

    # 5) Drop polygons completely outside the canvas (any vertex inside keeps it).
    kept_polys: list[np.ndarray] = []
    kept_igs: list[bool] = []
    for p, ig in zip(polys, ignore_flags):
        x_in = (p[:, 0] >= 0) & (p[:, 0] < out)
        y_in = (p[:, 1] >= 0) & (p[:, 1] < out)
        if (x_in & y_in).any():
            kept_polys.append(p.astype(np.float32))
            kept_igs.append(ig)
    return canvas, kept_polys, kept_igs
