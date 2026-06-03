import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

def structured_prune_l1(model, prune_ratio=0.3):
    """
    L1 structured pruning (removes filters)
    """
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name="weight", amount=prune_ratio, n=1, dim=0)
    return model

def compress_conv_layers(model):
    layers = []

    # collect conv layers in order
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            layers.append(module)

    # prune sequentially
    for i in range(len(layers)):
        current = layers[i]
        next_layer = layers[i+1] if i+1 < len(layers) else None

        remove_zero_filters_conv(current, next_layer)

    return model

def remove_zero_neurons_linear(layer):
    with torch.no_grad():
        weight = layer.weight.data

        keep_mask = torch.sum(torch.abs(weight), dim=1) != 0
        keep_idx = torch.where(keep_mask)[0]

        layer.weight.data = weight[keep_idx, :]
        layer.bias.data = layer.bias.data[keep_idx]

        layer.out_features = len(keep_idx)

    return keep_idx


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

