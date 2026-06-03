from src.evaluation.metrics import evaluate
import pandas as pd
from ptflops import get_model_complexity_info
import torchvision.models as models
from thop import profile, clever_format
import tempfile
import os
import torch
from torch.optim import Optimizer
import torch.optim as optim


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_size_mb(model):
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        temp_path = f.name
    try:
        torch.save(model.state_dict(), temp_path)
        size = os.path.getsize(temp_path) / (1024 ** 2)
    finally:
        os.remove(temp_path)
    return size

def compute_sparsity(model):
    total, zeros = 0, 0
    for p in model.parameters():
        total += p.numel()
        zeros += torch.sum(p == 0).item()
    return zeros / total

def compute_flops(model, input_size=(1, 3, 64, 64), device="cpu"):
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(input_size).to(device)

    # Step 1: get raw numbers first
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)

    # Step 2: math on raw numbers
    gflops = (macs * 2) / 1e9

    # Step 3: format for display only
    macs_readable, params_readable = clever_format([macs, params], "%.3f")
    print(f"MACs: {macs_readable} | Params: {params_readable} | GFLOPs: {gflops:.4f}")

    return gflops

def get_lr_scheduler(
    optimizer: Optimizer,
    schedule_type: str,
    total_epochs: int,
    base_lr: float,
    warmup_epochs: int = 5
):
    """
    Returns a LR scheduler based on schedule_type.

    Args:
        optimizer (Optimizer): PyTorch optimizer.
        schedule_type (str): One of ["cosine", "linear", "warmup_constant", ...].
        total_epochs (int): Number of training epochs.
        base_lr (float): Initial learning rate (and sometimes target LR).
        warmup_epochs (int): Used for 'warmup_constant' schedule, or can be extended for other schedules.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: The chosen scheduler.
    """

    if schedule_type == "cosine":
        # CosineAnnealingLR: Gradually decreases LR following a cosine curve.
        # Goes from base_lr down to eta_min over total_epochs.
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=0.0
        )

    elif schedule_type == "linear":
        # Linear decay from base_lr down to 0 over total_epochs
        def linear_decay(epoch):
            # epoch is zero-based; scale goes from 1.0 -> 0.0
            return 1.0 - float(epoch) / float(total_epochs - 1 if total_epochs > 1 else 1)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_decay)

    elif schedule_type == "warmup_constant":
        # 1) Warm-up linearly from 0 -> 1 over warmup_epochs
        # 2) Then remain constant (multiplier=1) after warmup.
        def warmup_then_constant(epoch):
            if epoch < warmup_epochs:
                # linear ramp from 0.0 -> 1.0
                return float(epoch + 1) / float(warmup_epochs)
            else:
                return 1.0

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_then_constant)

    else:
        # Default/fallback: do no scheduling (constant LR).
        print(f"[Warning] Unknown schedule_type='{schedule_type}'. Using constant LR.")
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

    return scheduler
