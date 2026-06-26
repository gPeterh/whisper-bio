"""BIO token and paper-style segmentation metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .config import IGNORE_ID


@dataclass
class BatchMetrics:
    loss: float = 0.0
    len: float = 0.0
    token_acc: float = 0.0
    boundary_hits: float = 0.0
    boundary_pred: float = 0.0
    boundary_ref: float = 0.0
    segmentation_precision: float = 0.0
    segmentation_recall: float = 0.0
    segmentation_f_value: float = 0.0
    segmentation_os: float = 0.0
    segmentation_r_value: float = 0.0


def repair_bio_sequence(sequence: torch.Tensor) -> torch.Tensor:
    repaired = sequence.clone()
    prev = 0
    for idx in range(repaired.numel()):
        label = int(repaired[idx].item())
        if label == 2 and prev == 0:
            repaired[idx] = 1
            label = 1
        prev = label
    return repaired


def repair_bio_predictions(pred: torch.Tensor, label_lens: torch.Tensor | None = None) -> torch.Tensor:
    repaired = pred.clone()
    if repaired.dim() == 1:
        length = repaired.numel() if label_lens is None else min(int(label_lens.item()), repaired.numel())
        repaired[:length] = repair_bio_sequence(repaired[:length])
        return repaired

    if repaired.dim() != 2:
        raise ValueError(f"BIO prediction repair expects a 1D or 2D tensor, got shape={tuple(repaired.shape)}")

    for batch_index in range(repaired.size(0)):
        length = repaired.size(1)
        if label_lens is not None:
            length = min(int(label_lens[batch_index].item()), length)
        repaired[batch_index, :length] = repair_bio_sequence(repaired[batch_index, :length])
    return repaired


def select_frame_from_group(group: list[int], frame_selection: str) -> list[int]:
    if frame_selection == "none":
        return group
    if frame_selection == "first":
        return [group[0]]
    if frame_selection == "last":
        return [group[-1]]
    if frame_selection == "mid":
        return [group[len(group) // 2]]
    raise ValueError(f"Unknown frame selection: {frame_selection}")


def boundary_positions(sequence: torch.Tensor, frame_selection: str) -> list[int]:
    positions = (sequence == 1).nonzero(as_tuple=False).flatten().tolist()
    if not positions:
        return []
    if frame_selection == "none":
        return positions

    selected: list[int] = []
    group = [positions[0]]
    for pos in positions[1:]:
        if pos == group[-1] + 1:
            group.append(pos)
        else:
            selected.extend(select_frame_from_group(group, frame_selection))
            group = [pos]
    selected.extend(select_frame_from_group(group, frame_selection))
    return selected


def paper_segmentation_metrics(
    hits: int | float,
    predicted: int | float,
    reference: int | float,
) -> tuple[float, float, float, float, float]:
    precision = hits / predicted if predicted > 0 else 0.0
    recall = hits / reference if reference > 0 else 0.0
    f_value = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    os = (predicted / reference - 1.0) if reference > 0 else 0.0
    r1 = math.sqrt((1.0 - recall) ** 2 + os**2)
    r2 = (-os + recall - 1.0) / math.sqrt(2.0)
    r_value = 1.0 - (abs(r1) + abs(r2)) / 2.0
    return precision, recall, f_value, os, r_value


def token_boundary_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_lens: torch.Tensor,
    tolerance: int,
    frame_selection: str,
) -> BatchMetrics:
    pred = logits.argmax(dim=-1)
    return prediction_boundary_metrics(pred, labels, label_lens, tolerance, frame_selection)


def prediction_boundary_metrics(
    pred: torch.Tensor,
    labels: torch.Tensor,
    label_lens: torch.Tensor,
    tolerance: int,
    frame_selection: str,
) -> BatchMetrics:
    valid = labels != IGNORE_ID
    metrics = BatchMetrics()
    metrics.len = int(valid.sum().item())
    if valid.any():
        metrics.token_acc = (pred[valid] == labels[valid]).float().mean().item()

    matched = 0
    total_pred = 0
    total_gold = 0
    for b in range(labels.size(0)):
        length = int(label_lens[b].item())
        length = max(0, min(length, labels.size(1), pred.size(1)))
        gold_pos = boundary_positions(labels[b, :length], frame_selection="none")
        pred_pos = boundary_positions(pred[b, :length], frame_selection=frame_selection)
        total_gold += len(gold_pos)
        total_pred += len(pred_pos)
        used: set[int] = set()
        for p in pred_pos:
            candidates = [
                (abs(p - g), idx)
                for idx, g in enumerate(gold_pos)
                if idx not in used and abs(p - g) <= tolerance
            ]
            if candidates:
                _, idx = min(candidates)
                used.add(idx)
                matched += 1

    metrics.boundary_hits = matched
    metrics.boundary_pred = total_pred
    metrics.boundary_ref = total_gold
    precision, recall, f_value, os_value, r_value = paper_segmentation_metrics(matched, total_pred, total_gold)
    metrics.segmentation_precision = precision
    metrics.segmentation_recall = recall
    metrics.segmentation_f_value = f_value
    metrics.segmentation_os = os_value
    metrics.segmentation_r_value = r_value
    return metrics
