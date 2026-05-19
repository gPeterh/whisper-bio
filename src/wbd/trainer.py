"""Epoch-level training and validation loop."""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import IGNORE_ID
from .loss import flatten_for_loss
from .metrics import BatchMetrics, paper_segmentation_metrics, token_boundary_metrics


def autocast_context(args: argparse.Namespace, device: torch.device):
    enabled = args.amp and device.type == "cuda"
    dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    args: argparse.Namespace,
    steps: int,
    desc: str,
    epoch: int,
    global_step: int,
    wandb_run=None,
    step_callback=None,
) -> tuple[BatchMetrics, int]:
    training = optimizer is not None
    model.train(training)
    totals = BatchMetrics()
    count = 0

    iterator = iter(loader)
    max_steps = steps if steps > 0 else None
    pbar = tqdm(total=max_steps, desc=desc, dynamic_ncols=True)
    step = 0
    while max_steps is None or step < max_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            break

        input_features = batch["input_features"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        label_lens = batch["label_lens"].to(device, non_blocking=True)

        with torch.set_grad_enabled(training), autocast_context(args, device):
            logits = model(input_features)
            flat_logits, flat_labels = flatten_for_loss(logits, labels)
            if (flat_labels != IGNORE_ID).any():
                loss = criterion(flat_logits, flat_labels)
            else:
                loss = flat_logits.sum() * 0.0
            if training:
                loss = loss / args.grad_accum_steps

        if training:
            loss.backward()
            grad_norm = None
            if (step + 1) % args.grad_accum_steps == 0:
                if args.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if grad_norm is not None and not torch.isfinite(torch.as_tensor(grad_norm)):
                    raise FloatingPointError(f"Non-finite gradient norm at epoch={epoch}, step={step}.")
                if step_callback is not None:
                    step_callback(epoch, global_step)
        else:
            grad_norm = None

        valid = labels != IGNORE_ID
        with torch.no_grad():
            if training:
                batch_metrics = BatchMetrics()
                if valid.any():
                    pred = logits.argmax(dim=-1)
                    batch_metrics.token_acc = (pred[valid] == labels[valid]).float().mean().item()
            else:
                batch_metrics = token_boundary_metrics(
                    logits.float(),
                    labels,
                    label_lens,
                    args.boundary_tolerance,
                    args.frame_selection,
                )
            batch_metrics.len = int(valid.sum().item())
        real_loss = loss.item() * (args.grad_accum_steps if training else 1)
        if not math.isfinite(real_loss):
            raise FloatingPointError(f"Non-finite loss at epoch={epoch}, step={step}.")
        totals.loss += real_loss
        totals.len += batch_metrics.len
        totals.token_acc += batch_metrics.token_acc
        totals.boundary_hits += batch_metrics.boundary_hits
        totals.boundary_pred += batch_metrics.boundary_pred
        totals.boundary_ref += batch_metrics.boundary_ref
        totals.segmentation_precision += batch_metrics.segmentation_precision
        totals.segmentation_recall += batch_metrics.segmentation_recall
        totals.segmentation_f_value += batch_metrics.segmentation_f_value
        totals.segmentation_os += batch_metrics.segmentation_os
        totals.segmentation_r_value += batch_metrics.segmentation_r_value
        count += 1

        if step == 0 or (args.log_every > 0 and (step + 1) % args.log_every == 0):
            postfix = {
                "loss": f"{totals.loss / count:.4f}",
                "tok_acc": f"{totals.token_acc / count:.3f}",
                "len": int(batch_metrics.len),
            }
            if not training:
                _, _, running_f_value, _, running_r_value = paper_segmentation_metrics(
                    totals.boundary_hits,
                    totals.boundary_pred,
                    totals.boundary_ref,
                )
                postfix.update(f_value=f"{running_f_value:.3f}", r_value=f"{running_r_value:.3f}")
            pbar.set_postfix(**postfix)
            if wandb_run is not None and training:
                import wandb

                lr = optimizer.param_groups[0]["lr"]
                grad_norm_value = float(grad_norm.item()) if torch.is_tensor(grad_norm) else grad_norm
                log_values = {
                    "global_step": global_step,
                    "train/epoch": epoch,
                    "train/lr": lr,
                    "train/grad_norm": grad_norm_value,
                    "train/loss_step": real_loss,
                    "train/token_acc_step": batch_metrics.token_acc,
                    "train/len": batch_metrics.len,
                }
                wandb.log(log_values)
        pbar.update(1)
        step += 1

    pbar.close()

    if training and count > 0 and count % args.grad_accum_steps != 0:
        grad_norm = None
        if args.max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        if step_callback is not None:
            step_callback(epoch, global_step)

    if count == 0:
        return totals, global_step
    for field in ("loss", "len", "token_acc"):
        setattr(totals, field, getattr(totals, field) / count)
    precision, recall, f_value, os_value, r_value = paper_segmentation_metrics(
        totals.boundary_hits,
        totals.boundary_pred,
        totals.boundary_ref,
    )
    totals.segmentation_precision = precision
    totals.segmentation_recall = recall
    totals.segmentation_f_value = f_value
    totals.segmentation_os = os_value
    totals.segmentation_r_value = r_value
    return totals, global_step
