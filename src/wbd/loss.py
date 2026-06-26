"""Loss functions for BIO frame classification."""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import IGNORE_ID, NUM_CLASSES


class FocalLoss(nn.Module):
    def __init__(self, gamma: float, weight: torch.Tensor | None, ignore_index: int):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0

        valid_logits = logits[valid]
        valid_target = target[valid]
        ce = F.cross_entropy(
            valid_logits,
            valid_target,
            reduction="none",
        )
        pt = torch.exp(-ce)
        loss = (1.0 - pt) ** self.gamma * ce
        if self.weight is not None:
            alpha = self.weight.to(device=logits.device, dtype=loss.dtype)[valid_target]
            loss = alpha * loss
        return loss.mean()


def parse_class_weights(value: str | None, device: torch.device) -> torch.Tensor | None:
    if not value:
        return None
    weights = [float(x.strip()) for x in value.split(",")]
    if len(weights) != NUM_CLASSES:
        raise ValueError("--class-weights must have exactly 3 values for O,B,I.")
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_criterion(args: argparse.Namespace, device: torch.device) -> nn.Module:
    weights = parse_class_weights(args.class_weights, device)
    if args.loss == "ce":
        return nn.CrossEntropyLoss(weight=weights, ignore_index=IGNORE_ID)
    return FocalLoss(gamma=args.focal_gamma, weight=weights, ignore_index=IGNORE_ID)


def flatten_for_loss(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
