"""Build paper/figures/datasets_overview.jpg by sampling one image from
each domain split and stacking them with text labels.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
HW_ANN = REPO_ROOT / "mn-uda-textdet/splits/handwrite_test.json"
SC_ANN = REPO_ROOT / "mn-uda-textdet/splits/scene_test.json"
DATA_ROOT = REPO_ROOT
OUT = Path(__file__).resolve().parent / "datasets_overview.jpg"

TARGET_W = 1200
PANEL_H = 600
LABEL_BAND = 36


def load_first_image(ann_path: Path, seed: int) -> Image.Image:
    data = json.loads(ann_path.read_text())["data_list"]
    rng = random.Random(seed)
    rec = rng.choice(data)
    img = Image.open(DATA_ROOT / rec["img_path"]).convert("RGB")
    img.thumbnail((TARGET_W, PANEL_H), Image.Resampling.LANCZOS)
    return img


def draw_panel(img: Image.Image, label: str) -> Image.Image:
    canvas = Image.new("RGB", (TARGET_W, PANEL_H + LABEL_BAND), (24, 24, 24))
    d = ImageDraw.Draw(canvas)
    d.text((10, 8), label, fill=(255, 255, 255))
    x = (TARGET_W - img.width) // 2
    y = LABEL_BAND + (PANEL_H - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def main() -> None:
    hw = load_first_image(HW_ANN, seed=11)
    sc = load_first_image(SC_ANN, seed=11)
    panel_hw = draw_panel(hw, "Handwriting domain (H): 1003 archival pages, 20,115 polygons")
    panel_sc = draw_panel(sc, "Scene domain (S): 1502 photographs, 18,598 quadrilaterals")
    final = Image.new("RGB", (TARGET_W, 2 * (PANEL_H + LABEL_BAND) + 8), (16, 16, 16))
    final.paste(panel_hw, (0, 0))
    final.paste(panel_sc, (0, PANEL_H + LABEL_BAND + 8))
    final.save(OUT, quality=88)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
