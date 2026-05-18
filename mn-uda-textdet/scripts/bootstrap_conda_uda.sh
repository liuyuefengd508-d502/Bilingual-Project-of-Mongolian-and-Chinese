#!/usr/bin/env bash
# 在仓库根目录安装 Miniforge（若尚无 conda），并创建 mn-uda-textdet-uda 环境。
# 用法（在 mn-uda-textdet 下）:
#   bash scripts/bootstrap_conda_uda.sh
# 可选环境变量:
#   MINIFORGE_ROOT  默认: 仓库根下的 .miniforge3
#   SKIP_MINIFORGE  若设为 1，则假定已安装 conda，仅用 CONDA_EXE 或 PATH 中的 conda
set -euo pipefail

MNUDA="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$MNUDA/.." && pwd)"
MINIFORGE_ROOT="${MINIFORGE_ROOT:-$REPO_ROOT/.miniforge3}"
case "$(uname -m)" in
  arm64) INSTALLER_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh" ;;
  x86_64) INSTALLER_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh" ;;
  *) echo "不支持的架构: $(uname -m)" >&2; exit 1 ;;
esac

resolve_conda() {
  if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
    echo "$CONDA_EXE"
    return
  fi
  for c in \
    "$MINIFORGE_ROOT/bin/conda" \
    "$HOME/miniforge3/bin/conda" \
    "$HOME/mambaforge/bin/conda" \
    "$HOME/miniconda3/bin/conda"; do
    if [[ -x "$c" ]]; then
      echo "$c"
      return
    fi
  done
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return
  fi
  echo ""
}

CONDA_BIN="$(resolve_conda)"

if [[ -z "$CONDA_BIN" && "${SKIP_MINIFORGE:-0}" != "1" ]]; then
  echo "[bootstrap] 未找到 conda，将 Miniforge 安装到: $MINIFORGE_ROOT"
  mkdir -p "$REPO_ROOT"
  tmp="$(mktemp /tmp/miniforgeXXXXXX.sh)"
  curl -fsSL -o "$tmp" "$INSTALLER_URL"
  bash "$tmp" -b -p "$MINIFORGE_ROOT"
  rm -f "$tmp"
  CONDA_BIN="$MINIFORGE_ROOT/bin/conda"
fi

if [[ -z "$CONDA_BIN" || ! -x "$CONDA_BIN" ]]; then
  echo "错误: 仍无法找到 conda。请手动安装 Miniforge 后重试，或设置 CONDA_EXE。" >&2
  exit 1
fi

echo "[bootstrap] 使用 conda: $CONDA_BIN"
# shellcheck source=/dev/null
source "$(dirname "$CONDA_BIN")/../etc/profile.d/conda.sh"

ENV_NAME="mn-uda-textdet-uda"
CONDA_ROOT="$(cd "$(dirname "$CONDA_BIN")/.." && pwd)"
ENV_DIR="$CONDA_ROOT/envs/$ENV_NAME"
MAMBA="$(dirname "$CONDA_BIN")/mamba"
run_mamba_or_conda() {
  if [[ -x "$MAMBA" ]]; then
    "$MAMBA" "$@"
  else
    "$CONDA_BIN" "$@"
  fi
}

if [[ -d "$ENV_DIR" ]]; then
  echo "[bootstrap] 更新环境 $ENV_NAME"
  run_mamba_or_conda env update -n "$ENV_NAME" -f "$MNUDA/environment_uda.yml" --prune
else
  echo "[bootstrap] 创建环境 $ENV_NAME (首次可能较慢)"
  run_mamba_or_conda env create -y -f "$MNUDA/environment_uda.yml"
fi

conda activate "$ENV_NAME"
# 避免 conda-forge OpenMP 与 PyTorch 同时加载时 abort（macOS 常见）
export KMP_DUPLICATE_LIB_OK=TRUE
python - <<'PY'
import importlib
mods = ("torch", "cv2", "pyclipper", "shapely")
for m in mods:
    importlib.import_module(m)
print("OK:", mods, "| torch", __import__("torch").__version__, "| MPS", __import__("torch").backends.mps.is_available())
PY

CONDA_ROOT="$(cd "$(dirname "$CONDA_BIN")/.." && pwd)"
echo "[bootstrap] 完成。激活命令:"
echo "  source \"$CONDA_ROOT/etc/profile.d/conda.sh\" && conda activate $ENV_NAME"
echo "长训示例:"
echo "  bash scripts/run_uda_s2h_long.sh"
