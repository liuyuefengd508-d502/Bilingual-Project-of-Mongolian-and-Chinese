"""Convert raw 蒙汉双语 labels into MMOCR-style TextDet annotations and split 7:1:2.

Outputs:
    splits/handwrite_{train,val,test}.json   # MMOCR TextDetDataset
    splits/scene_{train,val,test}.json
    splits/build_splits_report.json          # counts, dropped items, etc.

MMOCR TextDetDataset schema (v1.x):
{
  "metainfo": {"dataset_type": "TextDetDataset", "task_name": "textdet",
               "category": [{"id": 0, "name": "text"}]},
  "data_list": [
    {
      "img_path": "<abs path or relative to data_root>",
      "height": <int>, "width": <int>,
      "instances": [
        {"polygon": [x1,y1,x2,y2,...], "bbox": [xmin,ymin,xmax,ymax],
         "bbox_label": 0, "ignore": false}
      ]
    }, ...
  ]
}

Run:
    python tools/build_splits.py --root ../蒙汉双语 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

from PIL import Image


META = {
    "dataset_type": "TextDetDataset",
    "task_name": "textdet",
    "category": [{"id": 0, "name": "text"}],
}


def polygon_bbox(poly: list[float]) -> list[float]:
    xs = poly[0::2]
    ys = poly[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size  # (W, H)


# ---------------- Scene ----------------

def parse_scene_label(txt: Path) -> list[list[float]]:
    polys: list[list[float]] = []
    for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        # last field is the transcription tag (e.g. "#"), ignore it
        coords = parts[:-1]
        if len(coords) < 8 or len(coords) % 2 != 0:
            continue
        try:
            poly = [float(x) for x in coords]
        except ValueError:
            continue
        polys.append(poly)
    return polys


def build_scene(scene_root: Path, path_anchor: Path) -> list[dict]:
    img_dir = scene_root / "total"
    lbl_dir = scene_root / "label"
    items: list[dict] = []
    skipped: list[str] = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        lbl = lbl_dir / f"{img_path.stem}.txt"
        if not lbl.exists():
            skipped.append(img_path.name)
            continue
        polys = parse_scene_label(lbl)
        if not polys:
            skipped.append(img_path.name)
            continue
        w, h = image_size(img_path)
        instances = [
            {
                "polygon": p,
                "bbox": polygon_bbox(p),
                "bbox_label": 0,
                # `#` in this dataset is a placeholder, not "ignore" — treat as positive.
                "ignore": False,
            }
            for p in polys
        ]
        items.append({
            "img_path": str(img_path.resolve().relative_to(path_anchor)),
            "height": h, "width": w,
            "instances": instances,
        })
    return items, skipped


# ---------------- Handwrite ----------------

def parse_handwrite_label_file(path: Path) -> dict[str, list[list[float]]]:
    """Each line: `<relpath>\t<json_polygons>` where json_polygons is a list of
    {"transcription": ..., "points": [[x,y],...]}.
    Returns mapping stem -> list of flat polygons.
    """
    out: dict[str, list[list[float]]] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            relpath, payload = line.split("\t", 1)
        except ValueError:
            continue
        stem = Path(relpath).stem
        try:
            objs = json.loads(payload)
        except json.JSONDecodeError:
            continue
        polys: list[list[float]] = []
        for o in objs:
            pts = o.get("points") or []
            if len(pts) < 4:
                continue
            flat = [float(c) for xy in pts for c in xy]
            polys.append(flat)
        out[stem] = polys
    return out


def build_handwrite(hw_root: Path, path_anchor: Path) -> tuple[list[dict], list[str]]:
    img_dir = hw_root / "train"
    lbl_dir = hw_root / "train-label"
    label_map: dict[str, list[list[float]]] = {}
    for fn in ("training.txt", "test.txt"):
        f = lbl_dir / fn
        if f.exists():
            label_map.update(parse_handwrite_label_file(f))

    items: list[dict] = []
    skipped: list[str] = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        polys = label_map.get(img_path.stem)
        if not polys:
            skipped.append(img_path.name)
            continue
        w, h = image_size(img_path)
        instances = [
            {
                "polygon": p,
                "bbox": polygon_bbox(p),
                "bbox_label": 0,
                "ignore": False,
            }
            for p in polys
        ]
        items.append({
            "img_path": str(img_path.resolve().relative_to(path_anchor)),
            "height": h, "width": w,
            "instances": instances,
        })
    return items, skipped


# ---------------- Split ----------------

def split_712(items: list[dict], seed: int) -> tuple[list, list, list]:
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(round(n * 0.7))
    n_val = int(round(n * 0.1))
    train = [items[i] for i in idx[:n_train]]
    val = [items[i] for i in idx[n_train:n_train + n_val]]
    test = [items[i] for i in idx[n_train + n_val:]]
    return train, val, test


def dump(path: Path, data_list: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metainfo": META, "data_list": list(data_list)}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("../蒙汉双语"),
                    help="Path to the raw dataset root (蒙汉双语/).")
    ap.add_argument("--path-anchor", type=Path, default=None,
                    help="Anchor directory; img_path will be stored relative to it. "
                         "Default = parent of --root (so paths look like '蒙汉双语/...').")
    ap.add_argument("--out", type=Path, default=Path("splits"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")
    anchor = (args.path_anchor or root.parent).resolve()

    scene_items, scene_skipped = build_scene(root / "自然场景", anchor)
    hw_items, hw_skipped = build_handwrite(root / "手写档案", anchor)

    s_tr, s_va, s_te = split_712(scene_items, args.seed)
    h_tr, h_va, h_te = split_712(hw_items, args.seed)

    dump(args.out / "scene_train.json", s_tr)
    dump(args.out / "scene_val.json", s_va)
    dump(args.out / "scene_test.json", s_te)
    dump(args.out / "handwrite_train.json", h_tr)
    dump(args.out / "handwrite_val.json", h_va)
    dump(args.out / "handwrite_test.json", h_te)

    report = {
        "seed": args.seed,
        "path_anchor": str(anchor),
        "scene": {
            "kept": len(scene_items), "skipped": scene_skipped,
            "split": {"train": len(s_tr), "val": len(s_va), "test": len(s_te)},
            "polygons": sum(len(x["instances"]) for x in scene_items),
        },
        "handwrite": {
            "kept": len(hw_items), "skipped": hw_skipped,
            "split": {"train": len(h_tr), "val": len(h_va), "test": len(h_te)},
            "polygons": sum(len(x["instances"]) for x in hw_items),
        },
    }
    (args.out / "build_splits_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Scene ===")
    print(f"  kept={report['scene']['kept']} skipped={len(scene_skipped)} -> {scene_skipped[:5]}")
    print(f"  split={report['scene']['split']} polygons={report['scene']['polygons']}")
    print("=== Handwrite ===")
    print(f"  kept={report['handwrite']['kept']} skipped={len(hw_skipped)} -> {hw_skipped[:5]}")
    print(f"  split={report['handwrite']['split']} polygons={report['handwrite']['polygons']}")
    print(f"\nOutputs -> {args.out.resolve()}")


if __name__ == "__main__":
    main()
