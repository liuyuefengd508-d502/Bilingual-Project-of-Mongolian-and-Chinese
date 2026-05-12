"""Minimal DBNet implementation (ResNet18 backbone + FPN + DB head).

Differs from the original paper only in scope:
- Backbone: ResNet18 (smaller, MPS-friendly), no DCN.
- FPN with 4 levels (1/4, 1/8, 1/16, 1/32), output 1/4 resolution feature.
- DB head: probability map P, threshold map T, approximate binary map
  B_hat = 1 / (1 + exp(-k * (P - T))) (k=50 by default).

References:
    Liao et al., AAAI 2020 / TPAMI 2022.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50


class _UpAdd(nn.Module):
    """Upsample top to bottom size, add, then 3x3 conv."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, top: torch.Tensor, bottom: torch.Tensor) -> torch.Tensor:
        up = F.interpolate(top, size=bottom.shape[-2:], mode="bilinear",
                           align_corners=False)
        return self.conv(up + bottom)


class DBNet(nn.Module):
    """ResNet18+FPN DBNet for binary text detection.

    Output stride = 4.  Two heads each predict a 1-channel map at input
    resolution (upsampled bilinearly from stride-4 features).
    """

    def __init__(self, backbone: str = "resnet18", fpn_channels: int = 64,
                 k: float = 50.0, pretrained: bool = False) -> None:
        super().__init__()
        self.k = k

        if backbone == "resnet18":
            net = resnet18(weights="DEFAULT" if pretrained else None)
            chans = [64, 128, 256, 512]
        elif backbone == "resnet50":
            net = resnet50(weights="DEFAULT" if pretrained else None)
            chans = [256, 512, 1024, 2048]
        else:
            raise ValueError(backbone)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        # Lateral 1x1 convs to fpn_channels.
        self.lat1 = nn.Conv2d(chans[0], fpn_channels, 1)
        self.lat2 = nn.Conv2d(chans[1], fpn_channels, 1)
        self.lat3 = nn.Conv2d(chans[2], fpn_channels, 1)
        self.lat4 = nn.Conv2d(chans[3], fpn_channels, 1)
        self.smooth3 = _UpAdd(fpn_channels)
        self.smooth2 = _UpAdd(fpn_channels)
        self.smooth1 = _UpAdd(fpn_channels)

        # Fuse 4 levels to one 1/4 feature.
        self.fuse = nn.Sequential(
            nn.Conv2d(fpn_channels * 4, fpn_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
        )

        self.prob_head = self._make_head(fpn_channels)
        self.thresh_head = self._make_head(fpn_channels)

    @staticmethod
    def _make_head(in_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h0 = self.stem(x)
        c1 = self.layer1(h0)   # 1/4
        c2 = self.layer2(c1)   # 1/8
        c3 = self.layer3(c2)   # 1/16
        c4 = self.layer4(c3)   # 1/32

        p4 = self.lat4(c4)
        p3 = self.smooth3(p4, self.lat3(c3))
        p2 = self.smooth2(p3, self.lat2(c2))
        p1 = self.smooth1(p2, self.lat1(c1))   # 1/4

        # Upsample p2/p3/p4 to p1 resolution and concat.
        size = p1.shape[-2:]
        p2u = F.interpolate(p2, size=size, mode="bilinear", align_corners=False)
        p3u = F.interpolate(p3, size=size, mode="bilinear", align_corners=False)
        p4u = F.interpolate(p4, size=size, mode="bilinear", align_corners=False)
        feat = self.fuse(torch.cat([p1, p2u, p3u, p4u], dim=1))

        prob_logits = self.prob_head(feat)      # (B,1,H/4,W/4)
        thresh_logits = self.thresh_head(feat)

        # Upsample to input resolution.
        prob_logits = F.interpolate(prob_logits, size=x.shape[-2:],
                                    mode="bilinear", align_corners=False).squeeze(1)
        thresh_logits = F.interpolate(thresh_logits, size=x.shape[-2:],
                                      mode="bilinear", align_corners=False).squeeze(1)

        prob = torch.sigmoid(prob_logits)
        thresh = torch.sigmoid(thresh_logits)
        binary = torch.sigmoid(self.k * (prob - thresh))
        return {
            "prob_logits": prob_logits,
            "prob": prob,
            "thresh": thresh,
            "binary": binary,
        }


def _balanced_bce(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor,
                  negative_ratio: float = 3.0, eps: float = 1e-6) -> torch.Tensor:
    """OHEM-style BCE for probability map (gt in {0,1}, mask in {0,1})."""
    pos = ((gt > 0.5) & (mask > 0.5)).float()
    neg = ((gt <= 0.5) & (mask > 0.5)).float()
    n_pos = int(pos.sum().item())
    n_neg = int(min(neg.sum().item(), n_pos * negative_ratio))
    bce = F.binary_cross_entropy_with_logits(pred, gt, reduction="none") * mask
    if n_pos == 0:
        return bce.mean()
    pos_loss = (bce * pos).sum() / (pos.sum() + eps)
    if n_neg == 0:
        return pos_loss
    neg_loss_flat = (bce * neg).flatten()
    topk = torch.topk(neg_loss_flat, n_neg).values.sum() / (n_neg + eps)
    return pos_loss + topk


def _dice_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor,
               eps: float = 1e-6) -> torch.Tensor:
    pred = pred * mask
    gt = gt * mask
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() + eps
    return 1.0 - 2.0 * inter / union


def db_loss(out: dict[str, torch.Tensor], target: dict[str, torch.Tensor],
            w_prob: float = 1.0, w_binary: float = 1.0,
            w_thresh: float = 10.0) -> dict[str, torch.Tensor]:
    prob_logits = out["prob_logits"]
    binary = out["binary"]
    thresh = out["thresh"]

    l_prob = _balanced_bce(prob_logits, target["gt"], target["gt_mask"])
    l_binary = _dice_loss(binary, target["gt"], target["gt_mask"])
    # Threshold map: L1 in band region.
    diff = (thresh - target["thresh_map"]).abs() * target["thresh_mask"]
    l_thresh = diff.sum() / (target["thresh_mask"].sum() + 1e-6)

    total = w_prob * l_prob + w_binary * l_binary + w_thresh * l_thresh
    return {
        "loss": total,
        "loss_prob": l_prob.detach(),
        "loss_binary": l_binary.detach(),
        "loss_thresh": l_thresh.detach(),
    }
