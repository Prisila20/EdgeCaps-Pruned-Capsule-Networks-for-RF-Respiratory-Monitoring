import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune



def unstructured_prune_l1(model, prune_ratio=0.3):
    """
    L1 unstructured pruning (zeroing individual weights)
    """
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name="weight", amount=prune_ratio)
    return model
