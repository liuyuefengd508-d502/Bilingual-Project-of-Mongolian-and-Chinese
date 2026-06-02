"""把 baseline 与 UDA 的可视化结果拼成纵向对照图。

输入目录结构（每目录 6 张同名 .jpg，索引相同 → 同一张测试图）:
    work_dirs/figs/baseline_<dir>/00_*.jpg ... 05_*.jpg
    work_dirs/figs/uda_<dir>/00_*.jpg      ... 05_*.jpg

输出:
    work_dirs/figs/compare_<dir>.jpg   (6 行 x 2 列拼接)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def stack_pair(baseline_dir: Path, uda_dir: Path, out: Path) -> None:
    base_imgs = sorted(baseline_dir.glob("*.jpg"))
    uda_imgs = sorted(uda_dir.glob("*.jpg"))
    assert len(base_imgs) == len(uda_imgs), f"{len(base_imgs)} vs {len(uda_imgs)}"

    rows = []
    for b, u in zip(base_imgs, uda_imgs):
        bi = Image.open(b)
        ui = Image.open(u)
        # Same input image → same height. Stack vertically with label band.
        W = max(bi.width, ui.width)
        H = bi.height + ui.height + 60
        canvas = Image.new("RGB", (W, H), (32, 32, 32))
        d = ImageDraw.Draw(canvas)
        d.text((10, 5), f"Source-only ({b.stem})", fill=(255, 255, 255))
        canvas.paste(bi, (0, 30))
        d.text((10, bi.height + 35), f"UDA ({u.stem})", fill=(120, 240, 120))
        canvas.paste(ui, (0, bi.height + 60))
        rows.append(canvas)

    # Vertically stack the 6 pair-rows
    total_h = sum(r.height for r in rows) + 10 * (len(rows) - 1)
    max_w = max(r.width for r in rows)
    final = Image.new("RGB", (max_w, total_h), (16, 16, 16))
    y = 0
    for r in rows:
        final.paste(r, (0, y))
        y += r.height + 10
    out.parent.mkdir(parents=True, exist_ok=True)
    final.save(out, quality=85)
    print(f"  -> {out}  ({final.size})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs-root", type=Path,
                    default=Path("work_dirs/figs"))
    args = ap.parse_args()

    pairs = [
        ("h2s_scene",       "baseline_h2s_scene",       "uda_h2s_scene"),
        ("s2h_handwrite",   "baseline_s2h_handwrite",   "uda_s2h_handwrite"),
    ]
    for tag, base, uda in pairs:
        print(f"=== {tag} ===")
        stack_pair(args.figs_root / base, args.figs_root / uda,
                   args.figs_root / f"compare_{tag}.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
