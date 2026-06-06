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
