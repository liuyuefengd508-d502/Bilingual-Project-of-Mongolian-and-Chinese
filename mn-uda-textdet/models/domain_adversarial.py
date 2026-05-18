"""Image-level domain classifier with gradient reversal (DANN-style).

Applied to a spatial feature map (e.g. DBNet fused FPN tensor); uses global
average pooling then a small MLP. Used by ``tools/train_uda.py``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float) -> torch.Tensor:
    return _GradientReversalFn.apply(x, lambda_)  # type: ignore[return-value]


class ImageDomainHead(nn.Module):
    """Binary domain logits from a (B, C, H, W) feature map."""

    def __init__(self, in_channels: int = 64, hidden: int = 256, num_domains: int = 2) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_domains),
        )

    def forward(self, feat: torch.Tensor, grl_lambda: float) -> torch.Tensor:
        h = grad_reverse(feat, grl_lambda)
        x = self.pool(h).flatten(1)
        return self.fc(x)
