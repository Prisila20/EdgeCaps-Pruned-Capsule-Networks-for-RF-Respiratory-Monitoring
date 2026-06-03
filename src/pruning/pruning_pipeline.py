import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from sklearn.metrics import precision_score, recall_score, f1_score
import copy
from .structured import structured_prune_l1, compress_conv_layers
from .unstructured import unstructured_prune_l1


def finalize_pruning(model):
    """
    Remove pruning reparametrization to save final model
    """
    for module in model.modules():
        try:
            prune.remove(module, 'weight')
        except:
            pass
    return model

def rebuild_after_pruning(model, image_height=64, image_width=64):
    H = image_height // 4
    W = image_width // 4

    new_channels = model.conv3.out_channels
    new_flattened = new_channels * H * W

    old_hidden = model.fc.out_features

    # rebuild fc1
    model.fc = nn.Linear(new_flattened, old_hidden).to(next(model.parameters()).device)

    return model


def prune_model(model, prune_ratio):
    model = copy.deepcopy(model)

    model = structured_prune_l1(model, prune_ratio)
    model = unstructured_prune_l1(model, prune_ratio)

    model = finalize_pruning(model)
    model = compress_conv_layers(model)
    model = rebuild_after_pruning(model)

    return model

def remove_zero_filters_conv(layer, next_layer=None):
    """
    Removes filters (out_channels) that are entirely zero
    and adjusts next layer accordingly
    """
    with torch.no_grad():
        weight = layer.weight.data
        bias = layer.bias.data if layer.bias is not None else None

        # Find non-zero filters
        keep_mask = torch.sum(torch.abs(weight), dim=(1,2,3)) != 0
        keep_idx = torch.where(keep_mask)[0]

        # Prune current layer
        layer.weight.data = weight[keep_idx,:,:,:]
        if bias is not None:
            layer.bias.data = bias[keep_idx]

        layer.out_channels = len(keep_idx)

        # Adjust NEXT layer input channels
        if next_layer is not None:
            next_layer.weight.data = next_layer.weight.data[:, keep_idx, :, :]
            next_layer.in_channels = len(keep_idx)

    return keep_idx
