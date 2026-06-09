"""Generate paper/figures/long_run_curves.pdf from training logs.

Parses val_tgt H-mean per epoch from the H->S and S->H 25-epoch UDA logs.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
LOGS = {
    "H \u2192 S": REPO_ROOT / "mn-uda-textdet/work_dirs/uda_h2s_long.log",
    "S \u2192 H": REPO_ROOT / "mn-uda-textdet/work_dirs/uda_s2h_long2.log",
}

VAL_RE = re.compile(r"val_tgt H=(\d+\.\d+)")


def parse_h_curve(log_path: Path) -> list[float]:
    if not log_path.exists():
        raise FileNotFoundError(log_path)
    h_vals: list[float] = []
    for line in log_path.read_text().splitlines():
        m = VAL_RE.search(line)
        if m:
            h_vals.append(float(m.group(1)))
    return h_vals


def main() -> None:
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for label, path in LOGS.items():
        h = parse_h_curve(path)
        epochs = list(range(1, len(h) + 1))
        ax.plot(epochs, h, marker="o", markersize=3.5, linewidth=1.6,
                label=label)
        best_idx = int(max(range(len(h)), key=h.__getitem__))
        ax.annotate(f"best={h[best_idx]:.4f}@ep{best_idx+1}",
                    xy=(best_idx + 1, h[best_idx]),
                    xytext=(best_idx + 3, h[best_idx] + 0.005),
                    fontsize=8,
                    arrowprops=dict(arrowstyle="->", lw=0.6))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Target validation $H_{mean}$")
    ax.set_xlim(0, 26)
    ax.set_ylim(-0.005, 0.08)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    plt.tight_layout()

    out = Path(__file__).resolve().parent / "long_run_curves.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
