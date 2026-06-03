
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def get_mobilenetv3(num_classes):
    model = models.mobilenet_v3_small(weights=None)

    # Replace classifier
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        num_classes
    )

    return model       


def get_squeezenet(num_classes):
    model = models.squeezenet1_0(weights=None)

    model.classifier[1] = nn.Conv2d(
        512, num_classes, kernel_size=(1, 1)
    )
    model.num_classes = num_classes

    return model


def get_shufflenet(num_classes):
    model = models.shufflenet_v2_x0_5(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model

def get_resnet50(num_classes):
    model = models.resnet50(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model



def get_all_baselines(num_classes=3):
    return {
        "mobilenetv3_small": get_mobilenetv3(num_classes),
        "shufflenet": get_shufflenet(num_classes),
        "squeezenet": get_squeezenet(num_classes),
        "resnet50": get_resnet50(num_classes),
    }