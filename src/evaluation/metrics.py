import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np


def evaluate(model, data_loader, device, average='macro'):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = (np.array(all_preds) == np.array(all_labels)).mean() * 100
    precision = precision_score(all_labels, all_preds, average=average, zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, average=average, zero_division=0) * 100
    f1 = f1_score(all_labels, all_preds, average=average, zero_division=0) * 100

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}