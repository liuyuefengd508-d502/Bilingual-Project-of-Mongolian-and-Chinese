#!/usr/bin/env bash
# 在 conda 环境 mn-uda-textdet-uda 中跑 scene→handwrite UDA 长训。
# 数据根目录为仓库根（与 splits 中 img_path「蒙汉双语/...」一致）。
set -euo pipefail

MNUDA="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$MNUDA/.." && pwd)"
MINIFORGE_ROOT="${MINIFORGE_ROOT:-$REPO_ROOT/.miniforge3}"
ENV_NAME="mn-uda-textdet-uda"

activate_conda_env() {
  if [[ -n "${CONDA_DEFAULT_ENV:-}" && "$CONDA_DEFAULT_ENV" == "$ENV_NAME" ]]; then
    return 0
  fi
  for root in "$MINIFORGE_ROOT" "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/miniconda3"; do
    if [[ -f "$root/etc/profile.d/conda.sh" ]]; then
      # shellcheck source=/dev/null
      source "$root/etc/profile.d/conda.sh"
      conda activate "$ENV_NAME"
      return 0
    fi
  done
  echo "未找到 conda。请先运行: bash scripts/bootstrap_conda_uda.sh" >&2
  return 1
}

activate_conda_env || exit 1

# conda-forge numpy/opencv 与 pip torch 可能各带一份 libomp；不设会 Abort
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1

cd "$MNUDA"
OUT="${OUT:-work_dirs/uda_s2h_long}"
EPOCHS="${EPOCHS:-40}"
BATCH="${BATCH:-4}"

exec python -u tools/train_uda.py \
  --preset s2h \
  --val-target splits/handwrite_val.json \
  --val-source splits/scene_val.json \
  --data-root "$REPO_ROOT" \
  --pretrained \
  --out "$OUT" \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --log-jsonl "$OUT/metrics.jsonl"
