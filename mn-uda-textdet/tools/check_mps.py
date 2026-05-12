"""Check PyTorch MPS (Apple Silicon) availability and run a tiny smoke test."""
import platform
import sys


def main() -> int:
    print(f"Python : {sys.version.split()[0]}")
    print(f"System : {platform.platform()}")
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch not installed. `pip install torch torchvision`.")
        return 1

    print(f"Torch  : {torch.__version__}")
    available = torch.backends.mps.is_available()
    built = torch.backends.mps.is_built()
    print(f"MPS available : {available}")
    print(f"MPS built     : {built}")

    if not available:
        print("MPS unavailable; will fall back to CPU. Training will be slow.")
        return 0

    device = torch.device("mps")
    x = torch.randn(1024, 1024, device=device)
    y = x @ x.T
    print(f"MPS matmul OK, output shape={tuple(y.shape)}, mean={y.mean().item():.4f}")

    # Probe ops that are known to be problematic on MPS for OCR detectors.
    issues = []
    try:
        from torchvision.ops import deform_conv2d  # noqa: F401
        issues.append("torchvision.ops.deform_conv2d is importable, but DCN on MPS "
                      "is not supported by stock kernels — disable DCN in configs.")
    except Exception:
        pass
    if issues:
        print("\nNotes:")
        for s in issues:
            print(f"  - {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
