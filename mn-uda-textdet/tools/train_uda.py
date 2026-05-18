"""Unsupervised domain adaptation for DBNet (Teacher–Student + pseudo labels + DANN).

Source domain: full DBNet supervision from JSON polygons.
Target domain: no labels in the dataloader; teacher EMA generates pseudo polygons
from probability maps, rasterized to DBNet targets. Optional image-level domain
adversary on fused FPN features (gradient reversal).

Example (natural scene -> handwritten archive, target unlabeled):
    python tools/train_uda.py --preset s2h --data-root .. --out work_dirs/uda_s2h --epochs 25 --batch 4

Or pass explicit JSON paths instead of ``--preset`` (see ``--source-train`` / ``--target-train``).

``best_student.pth`` is written when validation improves (see ``--best-on``).
Use ``--resume work_dirs/uda_s2h/latest.pth`` to continue training with the same
optimizer and cosine schedule state.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets import TextDetJsonDataset, collate
from datasets.dbnet_targets import generate_dbnet_targets
from metrics import hmean_on_loader
from models import DBNet, ImageDomainHead, db_loss
from models.postprocess import prob_to_polygons


def pick_device(force_cpu: bool = False) -> torch.device:
    if not force_cpu and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _cycle_loader(loader: DataLoader) -> Iterator[dict[str, Any]]:
    while True:
        for batch in loader:
            yield batch


def _slice_out(out: dict[str, torch.Tensor], start: int, end: int) -> dict[str, torch.Tensor]:
    d: dict[str, torch.Tensor] = {}
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            d[k] = v[start:end]
    return d


def _ramp(epoch_1based: int, warmup_epochs: int, total_epochs: int, max_w: float) -> float:
    """Linear ramp from 0 (inclusive of warmup) to max_w after warmup."""
    if warmup_epochs <= 0:
        return max_w
    if epoch_1based <= warmup_epochs:
        return 0.0
    span = max(1, total_epochs - warmup_epochs)
    t = (epoch_1based - warmup_epochs) / float(span)
    return max_w * min(1.0, t)


def _pseudo_targets_from_teacher(
    prob_np: np.ndarray,
    image_size: int,
    shrink_ratio: float,
    box_thresh: float,
    min_score: float,
    unclip_ratio: float,
    max_aspect: float,
    min_poly_area: float,
    max_candidates: int,
) -> dict[str, torch.Tensor] | None:
    polys_scored = prob_to_polygons(
        prob_np,
        box_thresh=box_thresh,
        min_score=min_score,
        unclip_ratio=unclip_ratio,
        max_candidates=max_candidates,
    )
    polys: list[np.ndarray] = []
    for poly, _score in polys_scored:
        poly = poly.astype(np.float32)
        area = float(cv2.contourArea(poly.astype(np.int32)))
        if area < min_poly_area:
            continue
        xs, ys = poly[:, 0], poly[:, 1]
        w = float(xs.max() - xs.min() + 1.0)
        h = float(ys.max() - ys.min() + 1.0)
        ar = max(w, h) / max(min(w, h), 1.0)
        if ar > max_aspect:
            continue
        polys.append(poly)
    if not polys:
        return None
    tgt = generate_dbnet_targets(
        polys,
        [False] * len(polys),
        image_size,
        image_size,
        shrink_ratio=shrink_ratio,
    )
    return {k: torch.from_numpy(v).unsqueeze(0) for k, v in tgt.items()}


def _pseudo_loss_per_sample(
    student_slice: dict[str, torch.Tensor],
    teacher_prob_hw: np.ndarray,
    device: torch.device,
    image_size: int,
    shrink_ratio: float,
    pseudo_box: float,
    pseudo_score: float,
    unclip_ratio: float,
    max_aspect: float,
    min_poly_area: float,
    max_candidates: int,
) -> torch.Tensor | None:
    tdict = _pseudo_targets_from_teacher(
        teacher_prob_hw,
        image_size,
        shrink_ratio,
        pseudo_box,
        pseudo_score,
        unclip_ratio,
        max_aspect,
        min_poly_area,
        max_candidates,
    )
    if tdict is None:
        return None
    target = {k: v.to(device) for k, v in tdict.items()}
    return db_loss(student_slice, target)["loss"]


@torch.no_grad()
def _ema_update(student: torch.nn.Module, teacher: torch.nn.Module, m: float) -> None:
    for p_s, p_t in zip(student.parameters(), teacher.parameters()):
        p_t.data.mul_(m).add_(p_s.data, alpha=1.0 - m)


def _save_checkpoint(
    path: Path,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    domain_head: torch.nn.Module,
    epoch: int,
    args: argparse.Namespace,
    opt: torch.optim.Optimizer,
    sched: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "domain_head": domain_head.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
        },
        path,
    )


def apply_preset(args: argparse.Namespace) -> None:
    """Fill split paths when ``--preset`` is set; only ``None`` fields are written."""
    root = Path("splits")
    if args.preset == "s2h":
        if args.source_train is None:
            args.source_train = root / "scene_train.json"
        if args.target_train is None:
            args.target_train = root / "handwrite_train.json"
        if args.val_source is None:
            args.val_source = root / "scene_val.json"
    elif args.preset == "h2s":
        if args.source_train is None:
            args.source_train = root / "handwrite_train.json"
        if args.target_train is None:
            args.target_train = root / "scene_train.json"
        if args.val_source is None:
            args.val_source = root / "handwrite_val.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=("s2h", "h2s"),
        help="Shortcut: scene→handwrite (s2h) or handwrite→scene (h2s); fills train + val-source paths under splits/ when those args are omitted.",
    )
    ap.add_argument("--source-train", type=Path, default=None)
    ap.add_argument("--target-train", type=Path, default=None)
    ap.add_argument("--val-source", type=Path, default=None,
                    help="Optional source-domain val JSON for quick val logging.")
    ap.add_argument("--data-root", type=Path, default=Path(".."))
    ap.add_argument("--out", type=Path, default=Path("work_dirs/uda_run"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-domain", type=float, default=None,
                    help="Learning rate for domain head; defaults to --lr.")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--backbone", type=str, default="resnet18")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")

    ap.add_argument("--ema-momentum", type=float, default=0.999)
    ap.add_argument("--pseudo-weight-max", type=float, default=0.5,
                    help="Max multiplier on mean pseudo-label DBNet loss.")
    ap.add_argument("--pseudo-warmup-epochs", type=int, default=3)
    ap.add_argument("--pseudo-box-thresh", type=float, default=0.45)
    ap.add_argument("--pseudo-min-score", type=float, default=0.55)
    ap.add_argument("--pseudo-max-aspect", type=float, default=80.0,
                    help="Drop pseudo boxes with axis-aligned aspect ratio above this.")
    ap.add_argument("--pseudo-min-area", type=float, default=64.0,
                    help="Min polygon area in resized pixels.")
    ap.add_argument("--unclip-ratio", type=float, default=1.5)
    ap.add_argument(
        "--pseudo-max-candidates",
        type=int,
        default=400,
        help="Cap teacher polygon candidates per image (prob_to_polygons max_candidates).",
    )
    ap.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.0,
        help="If >0, clip joint gradient norm of student + domain_head after backward.",
    )

    ap.add_argument("--da-weight-max", type=float, default=0.1,
                    help="Weight on domain classification loss (0 to disable schedule).")
    ap.add_argument("--da-warmup-epochs", type=int, default=2)
    ap.add_argument("--grl-max", type=float, default=1.0,
                    help="Gradient reversal strength (ramps with epoch after da warmup).")

    ap.add_argument("--shrink-ratio", type=float, default=0.4)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="If >0, cap optimizer steps per epoch (smoke tests).")
    ap.add_argument("--val-target", type=Path, default=None,
                    help="Optional labeled target-domain val JSON (dev only: tune UDA on target val H-mean).")
    ap.add_argument("--eval-box-thresh", type=float, default=0.45)
    ap.add_argument("--eval-min-score", type=float, default=0.5)
    ap.add_argument("--eval-unclip-ratio", type=float, default=1.5)
    ap.add_argument("--eval-iou-thresh", type=float, default=0.5)
    ap.add_argument(
        "--val-every",
        type=int,
        default=1,
        help="Run val_* metrics every N epochs (always runs on the last epoch).",
    )
    ap.add_argument(
        "--log-jsonl",
        type=Path,
        default=None,
        help="Append one JSON object per epoch (metrics row) for external plotting.",
    )
    ap.add_argument("--resume", type=Path, default=None,
                    help="Resume from a checkpoint saved by this script (latest.pth or best_student.pth).")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="Initialize student+teacher from a single-domain DBNet ckpt "
                         "(e.g. scene_r18/best.pth) to break the UDA cold-start. "
                         "Ignored if --resume is set.")
    ap.add_argument(
        "--best-on",
        type=str,
        default="auto",
        choices=("auto", "target_hmean", "source_val_loss", "none"),
        help="Criterion for best_student.pth. auto: target val H if --val-target else min source val loss.",
    )
    ap.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop after this many validation rounds without improvement (0=off). "
             "Only counts epochs where validation ran; uses the same metric as --best-on.",
    )
    args = ap.parse_args()
    args.val_every = max(1, int(args.val_every))
    apply_preset(args)
    if args.source_train is None or args.target_train is None:
        ap.error("Need --source-train and --target-train, or use --preset s2h|h2s.")

    lr_domain = float(args.lr_domain) if args.lr_domain is not None else float(args.lr)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    device = pick_device(force_cpu=args.cpu)
    print(f"Device : {device}")
    print(f"Out    : {args.out.resolve()}")

    src_train = TextDetJsonDataset(
        args.source_train, args.data_root, image_size=args.size,
        target_mode="dbnet", augment=True, seed=args.seed, labeled=True,
    )
    tgt_train = TextDetJsonDataset(
        args.target_train, args.data_root, image_size=args.size,
        target_mode="dbnet", augment=True, seed=args.seed + 1, labeled=False,
    )
    src_loader = DataLoader(
        src_train, batch_size=args.batch, shuffle=True, num_workers=args.num_workers,
        collate_fn=collate, drop_last=True,
    )
    tgt_loader = DataLoader(
        tgt_train, batch_size=args.batch, shuffle=True, num_workers=args.num_workers,
        collate_fn=collate, drop_last=True,
    )
    steps_per_epoch = max(len(src_loader), len(tgt_loader))
    steps_this_epoch = (
        min(steps_per_epoch, args.max_steps) if args.max_steps > 0 else steps_per_epoch
    )
    total_sched_steps = args.epochs * steps_this_epoch
    print(f"Source train: {len(src_train)} | Target train: {len(tgt_train)}")
    print(f"Steps/epoch : {steps_this_epoch} (capped from {steps_per_epoch})")

    val_loader = None
    val_ds = None
    if args.val_source is not None:
        val_ds = TextDetJsonDataset(
            args.val_source, args.data_root, image_size=args.size,
            target_mode="dbnet", augment=False, labeled=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate, drop_last=False,
        )

    val_target_loader = None
    val_target_ds = None
    if args.val_target is not None:
        val_target_ds = TextDetJsonDataset(
            args.val_target, args.data_root, image_size=args.size,
            target_mode="dbnet", augment=False, labeled=True,
        )
        val_target_loader = DataLoader(
            val_target_ds, batch_size=args.batch, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate, drop_last=False,
        )

    student = DBNet(backbone=args.backbone, pretrained=args.pretrained).to(device)
    teacher = DBNet(backbone=args.backbone, pretrained=False).to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    fpn_ch = 64
    domain_head = ImageDomainHead(in_channels=fpn_ch, hidden=256, num_domains=2).to(device)

    opt = torch.optim.AdamW(
        [
            {"params": student.parameters(), "lr": args.lr},
            {"params": domain_head.parameters(), "lr": lr_domain},
        ],
        weight_decay=args.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_sched_steps)

    history_path = args.out / "history.json"
    history: list[dict] = []

    start_epoch = 1
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        student.load_state_dict(ckpt["student"])
        teacher.load_state_dict(ckpt["teacher"])
        domain_head.load_state_dict(ckpt["domain_head"])
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        if "sched" in ckpt:
            sched.load_state_dict(ckpt["sched"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        if history_path.exists():
            try:
                history = json.loads(history_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                history = []
        print(f"Resume : {args.resume} -> start epoch {start_epoch}")
    else:
        if args.init_from is not None:
            init_ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
            init_sd = init_ckpt.get("model", init_ckpt)
            missing, unexpected = student.load_state_dict(init_sd, strict=False)
            teacher.load_state_dict(student.state_dict())
            print(f"Init   : {args.init_from} -> student+teacher "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")
        else:
            teacher.load_state_dict(student.state_dict())

    if start_epoch > args.epochs:
        print(f"Nothing to do: start_epoch {start_epoch} > epochs {args.epochs}")
        return 0

    if args.best_on == "auto":
        best_mode = (
            "target_hmean"
            if val_target_loader is not None
            else ("source_val_loss" if val_loader is not None else "none")
        )
    else:
        best_mode = args.best_on

    best_target_h = -1.0
    best_src_val = float("inf")
    for row in history:
        if "val_target_hmean" in row:
            v = float(row["val_target_hmean"])
            if v > 0.0 and v > best_target_h:
                best_target_h = v
        if "val_src_loss" in row:
            best_src_val = min(best_src_val, float(row["val_src_loss"]))

    it_src = _cycle_loader(src_loader)
    it_tgt = _cycle_loader(tgt_loader)
    patience_ctr = 0

    for epoch in range(start_epoch, args.epochs + 1):
        w_pseudo = _ramp(epoch, args.pseudo_warmup_epochs, args.epochs, args.pseudo_weight_max)
        w_da = _ramp(epoch, args.da_warmup_epochs, args.epochs, args.da_weight_max)
        grl_lambda = _ramp(epoch, args.da_warmup_epochs, args.epochs, args.grl_max)

        student.train()
        domain_head.train()
        teacher.eval()

        running: dict[str, float] = {
            "loss": 0.0, "src": 0.0, "pseudo": 0.0, "da": 0.0, "n": 0.0,
            "n_pseudo": 0.0,
        }
        t0 = time.time()
        for step in range(1, steps_this_epoch + 1):
            batch_s = next(it_src)
            batch_t = next(it_tgt)
            img_s = batch_s["image"].to(device)
            img_t = batch_t["image"].to(device)
            B = img_s.shape[0]
            assert img_t.shape[0] == B

            x_cat = torch.cat([img_s, img_t], dim=0)
            out = student(x_cat, return_feat=True)
            feat = out.pop("feat")
            out_s = _slice_out(out, 0, B)
            out_t = _slice_out(out, B, 2 * B)

            tgt_src = {k: batch_s[k].to(device) for k in ("gt", "gt_mask", "thresh_map", "thresh_mask")}
            loss_s = db_loss(out_s, tgt_src)["loss"]

            with torch.no_grad():
                t_out = teacher(img_t)
                t_prob = t_out["prob"].detach()

            pseudo_terms: list[torch.Tensor] = []
            for b in range(B):
                pb = t_prob[b].float().cpu().numpy()
                sl = {k: v[b : b + 1] for k, v in out_t.items()}
                lp = _pseudo_loss_per_sample(
                    sl, pb, device, args.size, args.shrink_ratio,
                    args.pseudo_box_thresh, args.pseudo_min_score, args.unclip_ratio,
                    args.pseudo_max_aspect, args.pseudo_min_area, args.pseudo_max_candidates,
                )
                if lp is not None:
                    pseudo_terms.append(lp)
            if pseudo_terms:
                loss_p = torch.stack(pseudo_terms).mean()
                n_ps = float(len(pseudo_terms))
            else:
                loss_p = out_t["prob"].sum() * 0.0
                n_ps = 0.0

            dom_logits = domain_head(feat, grl_lambda)
            dom_y = torch.cat(
                [torch.zeros(B, dtype=torch.long, device=device),
                 torch.ones(B, dtype=torch.long, device=device)],
                dim=0,
            )
            loss_da = F.cross_entropy(dom_logits, dom_y)

            loss = loss_s + w_pseudo * loss_p + w_da * loss_da

            opt.zero_grad()
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(student.parameters()) + list(domain_head.parameters()),
                    args.max_grad_norm,
                )
            opt.step()
            sched.step()

            _ema_update(student, teacher, args.ema_momentum)

            running["loss"] += float(loss.detach().item())
            running["src"] += float(loss_s.detach().item())
            running["pseudo"] += float(loss_p.detach().item()) if pseudo_terms else 0.0
            running["da"] += float(loss_da.detach().item())
            running["n"] += 1.0
            running["n_pseudo"] += n_ps

            if step % args.log_every == 0:
                print(f"  ep{epoch:>3} st{step:>4}/{steps_this_epoch} "
                      f"loss={loss.item():.4f} src={loss_s.item():.4f} "
                      f"pseudo={loss_p.item():.4f} da={loss_da.item():.4f} "
                      f"w_p={w_pseudo:.3f} w_da={w_da:.3f} grl={grl_lambda:.3f} "
                      f"pseudo_frac={n_ps/B:.2f}")

        dt = time.time() - t0
        n = max(1.0, running["n"])
        entry = {
            "epoch": epoch,
            "lr": opt.param_groups[0]["lr"],
            "train_sec": dt,
            "loss": running["loss"] / n,
            "loss_src": running["src"] / n,
            "loss_pseudo": running["pseudo"] / n,
            "loss_da": running["da"] / n,
            "w_pseudo": w_pseudo,
            "w_da": w_da,
            "grl_lambda": grl_lambda,
            "pseudo_hit_rate": running["n_pseudo"] / (n * float(args.batch)),
        }
        do_val = (val_loader is not None or val_target_loader is not None) and (
            epoch % args.val_every == 0 or epoch == args.epochs
        )
        if do_val:
            student.eval()
            if val_loader is not None:
                v_loss = 0.0
                v_n = 0
                with torch.no_grad():
                    for vb in val_loader:
                        vi = vb["image"].to(device)
                        vo = student(vi, return_feat=False)
                        vt = {k: vb[k].to(device) for k in ("gt", "gt_mask", "thresh_map", "thresh_mask")}
                        v_loss += float(db_loss(vo, vt)["loss"].item())
                        v_n += 1
                entry["val_src_loss"] = v_loss / max(1, v_n)
                print(f"         val_src_loss={entry['val_src_loss']:.4f}")
            if val_target_loader is not None:
                mt = hmean_on_loader(
                    student, val_target_loader, val_target_ds, device,
                    args.size,
                    args.eval_box_thresh, args.eval_min_score,
                    args.eval_unclip_ratio, args.eval_iou_thresh,
                )
                entry["val_target_hmean"] = float(mt["hmean"])
                entry["val_target_precision"] = float(mt["precision"])
                entry["val_target_recall"] = float(mt["recall"])
                print(
                    f"         val_tgt H={entry['val_target_hmean']:.4f} "
                    f"P={entry['val_target_precision']:.4f} R={entry['val_target_recall']:.4f}"
                )
            student.train()
            domain_head.train()
        else:
            entry["val_skipped"] = True

        history.append(entry)
        print(f"[ep {epoch:>3}/{args.epochs}] "
              f"loss={entry['loss']:.4f} src={entry['loss_src']:.4f} "
              f"pseudo={entry['loss_pseudo']:.4f} da={entry['loss_da']:.4f} "
              f"({dt:.1f}s)")

        history_path.write_text(
            json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

        if args.log_jsonl is not None:
            args.log_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with args.log_jsonl.open("a", encoding="utf-8") as jf:
                jf.write(json.dumps(entry, ensure_ascii=False) + "\n")

        _save_checkpoint(
            args.out / "latest.pth", student, teacher, domain_head,
            epoch, args, opt, sched,
        )

        improved = False
        if best_mode == "target_hmean" and val_target_loader is not None:
            h = entry.get("val_target_hmean")
            if h is not None and h > best_target_h:
                # Avoid saving a useless checkpoint when H is still 0 at cold start.
                if not (best_target_h < 0 and h <= 0.0):
                    best_target_h = h
                    improved = True
        elif best_mode == "source_val_loss" and val_loader is not None:
            vs = entry.get("val_src_loss")
            if vs is not None and vs < best_src_val:
                best_src_val = vs
                improved = True

        if improved:
            _save_checkpoint(
                args.out / "best_student.pth", student, teacher, domain_head,
                epoch, args, opt, sched,
            )
            if best_mode == "target_hmean":
                print(f"  -> new best val_target H-mean {best_target_h:.4f} (best_student.pth)")
            else:
                print(f"  -> new best val_src_loss {best_src_val:.4f} (best_student.pth)")

        if args.early_stop_patience > 0 and do_val and best_mode != "none":
            metric_ok = False
            if best_mode == "target_hmean" and val_target_loader is not None:
                metric_ok = entry.get("val_target_hmean") is not None
            elif best_mode == "source_val_loss" and val_loader is not None:
                metric_ok = entry.get("val_src_loss") is not None
            if metric_ok:
                if improved:
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= args.early_stop_patience:
                        print(
                            f"\nEarly stopping: no improvement for {args.early_stop_patience} "
                            f"validation checkpoints (--best-on={best_mode})."
                        )
                        break

    if best_mode == "target_hmean" and val_target_loader is not None:
        if best_target_h < 0:
            print("\nDone. (val_target H-mean never exceeded 0; best_student.pth not written.)")
        else:
            print(f"\nDone. Best val_target H-mean = {best_target_h:.4f}")
    elif best_mode == "source_val_loss" and val_loader is not None:
        print(f"\nDone. Best val_src_loss = {best_src_val:.4f}")
    else:
        print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
