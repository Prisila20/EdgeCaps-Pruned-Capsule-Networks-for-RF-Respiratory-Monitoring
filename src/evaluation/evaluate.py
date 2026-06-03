import torch
import torch.nn as nn
import torch.nn.functional as F
from .metrics import evaluate
import pandas as pd
from ptflops import get_model_complexity_info
import torchvision.models as models
from thop import profile, clever_format
import tempfile
from src.utils import count_parameters, model_size_mb, compute_flops, compute_sparsity
import numpy as np

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# def model_size_mb(model):
#     with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
#         temp_path = f.name
#     try:
#         torch.save(model.state_dict(), temp_path)
#         size = os.path.getsize(temp_path) / (1024 ** 2)
#     finally:
#         os.remove(temp_path)
#     return size

# def compute_sparsity(model):
#     total, zeros = 0, 0
#     for p in model.parameters():
#         total += p.numel()
#         zeros += torch.sum(p == 0).item()
#     return zeros / total

# def compute_flops(model, input_size=(1, 3, 64, 64), device="cpu"):
#     model = model.to(device)
#     model.eval()

#     dummy_input = torch.randn(input_size).to(device)

#     # Step 1: get raw numbers first
#     macs, params = profile(model, inputs=(dummy_input,), verbose=False)

#     # Step 2: math on raw numbers
#     gflops = (macs * 2) / 1e9

#     # Step 3: format for display only
#     macs_readable, params_readable = clever_format([macs, params], "%.3f")
#     print(f"MACs: {macs_readable} | Params: {params_readable} | GFLOPs: {gflops:.4f}")

#     return gflops


def evaluate_and_log(model, name, test_loader, device,
                     teacher_params, teacher_size, teacher_flops):

    metrics = evaluate(model, test_loader, device)

    params = count_parameters(model)
    size = model_size_mb(model)
    sparsity = compute_sparsity(model)
    flops = compute_flops(model)

    print(f"Teacher  params: {teacher_params}, size: {teacher_size:.3f} MB, flops: {teacher_flops:.6f} GFLOPs")
    print(f"{name} params: {params}, size: {size:.3f} MB, flops: {flops:.6f} GFLOPs \n")

    return {
        "model": name,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "params": params,
        "size_mb": size,
        "flops": flops,
        "sparsity": sparsity,

        # reductions vs teacher
        "param_reduction_%": 100 * (1 - params / teacher_params),
        "size_reduction_%": 100 * (1 - size / teacher_size),
        "flops_reduction_%": 100 * (1 - flops / teacher_flops),
    }
