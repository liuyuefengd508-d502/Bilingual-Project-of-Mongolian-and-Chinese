"""Verify dataset integrity for 蒙汉双语.

Checks:
1. 自然场景 total/*.jpg <-> label/*.txt one-to-one mapping (1502 vs 1501).
2. 手写档案 train/*.jpg <-> train-label/{training,test}.txt coverage.
3. Basic statistics: per-image polygon count, polygon vertex counts, transcription tags.

Run:
    python tools/verify_dataset.py --root ../蒙汉双语
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def check_scene(scene_root: Path) -> dict:
    img_dir = scene_root / "total"
    lbl_dir = scene_root / "label"
    imgs = {p.stem: p for p in img_dir.glob("*.jpg")}
    lbls = {p.stem: p for p in lbl_dir.glob("*.txt")}
    only_img = sorted(set(imgs) - set(lbls))
    only_lbl = sorted(set(lbls) - set(imgs))

    # polygon stats
    poly_counts, vertex_counts, tag_counter = [], [], Counter()
    for stem, path in lbls.items():
        n = 0
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            tag = parts[-1].strip()
            tag_counter[tag] += 1
            coords = parts[:-1]
            vertex_counts.append(len(coords) // 2)
            n += 1
        poly_counts.append(n)

    return {
        "subset": "scene",
        "image_count": len(imgs),
        "label_count": len(lbls),
        "missing_labels": only_img,
        "orphan_labels": only_lbl,
        "polygon_count": {
            "total": sum(poly_counts),
            "mean_per_image": (sum(poly_counts) / len(poly_counts)) if poly_counts else 0,
            "max": max(poly_counts) if poly_counts else 0,
            "min": min(poly_counts) if poly_counts else 0,
        },
        "vertex_count_distribution": dict(Counter(vertex_counts)),
        "transcription_tags_top10": tag_counter.most_common(10),
    }


def check_handwrite(hw_root: Path) -> dict:
    img_dir = hw_root / "train"
    lbl_dir = hw_root / "train-label"
    imgs = {p.stem: p for p in img_dir.glob("*.jpg")}

    referenced = set()
    poly_counts = []
    for split_name in ("training.txt", "test.txt"):
        f = lbl_dir / split_name
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            # PaddleOCR row: <relpath>\t<json_polygons>
            try:
                relpath, payload = line.split("\t", 1)
            except ValueError:
                continue
            stem = Path(relpath).stem
            referenced.add(stem)
            try:
                polys = json.loads(payload)
                poly_counts.append(len(polys))
            except json.JSONDecodeError:
                continue

    return {
        "subset": "handwrite",
        "image_count": len(imgs),
        "labeled_image_count": len(referenced),
        "unlabeled_images": sorted(set(imgs) - referenced),
        "labels_referencing_missing_images": sorted(referenced - set(imgs)),
        "polygon_count": {
            "total": sum(poly_counts),
            "mean_per_image": (sum(poly_counts) / len(poly_counts)) if poly_counts else 0,
            "max": max(poly_counts) if poly_counts else 0,
            "min": min(poly_counts) if poly_counts else 0,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("../蒙汉双语"),
                    help="Path to 蒙汉双语 directory.")
    ap.add_argument("--out", type=Path, default=Path("docs/dataset_report.json"))
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    report = {
        "root": str(root),
        "scene": check_scene(root / "自然场景"),
        "handwrite": check_handwrite(root / "手写档案"),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # console summary
    s = report["scene"]
    h = report["handwrite"]
    print("=== 自然场景 ===")
    print(f"  images={s['image_count']} labels={s['label_count']}")
    print(f"  missing_labels={len(s['missing_labels'])} -> {s['missing_labels'][:5]}")
    print(f"  orphan_labels ={len(s['orphan_labels'])} -> {s['orphan_labels'][:5]}")
    print(f"  polygons total={s['polygon_count']['total']} "
          f"mean/img={s['polygon_count']['mean_per_image']:.2f}")
    print(f"  vertex distrib={s['vertex_count_distribution']}")
    print(f"  tag top10={s['transcription_tags_top10']}")
    print("=== 手写档案 ===")
    print(f"  images={h['image_count']} labeled={h['labeled_image_count']} "
          f"unlabeled={len(h['unlabeled_images'])}")
    print(f"  polygons total={h['polygon_count']['total']} "
          f"mean/img={h['polygon_count']['mean_per_image']:.2f}")
    print(f"\nFull report -> {args.out}")


if __name__ == "__main__":
    main()
