import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from .utils import EarlyStopping, get_lr_scheduler
from src.evaluation.metrics import evaluate
import os

def train_model(model, train_loader, val_loader, device, args, prefix):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = get_lr_scheduler(
        optimizer=optimizer,
        schedule_type="warmup_constant",
        total_epochs=100,
        base_lr=args.lr,
        warmup_epochs=10
    )

    ckpt_path = f"models_2/{prefix}_checkpoint_{args.job_id}.pth"
    
    # Remove any stale checkpoint from a previous run before training
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    early_stopper = EarlyStopping( patience=10, save_path=ckpt_path)
   

    model.train()

    for epoch in range(1, 101):
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)

        # get validation metrics properly
        val_metrics = evaluate(model, val_loader, device)
        val_acc = val_metrics["accuracy"]

        early_stopper.step(val_acc, model)

        print(f"Epoch [{epoch}/100] - Loss: {epoch_loss:.4f}, "
              f"Val Acc: {val_acc:.2f}%, "
              f"Precision: {val_metrics['precision']:.2f}, "
              f"Recall: {val_metrics['recall']:.2f}, "
              f"F1-score: {val_metrics['f1']:.2f}")

        if early_stopper.early_stop:
            # torch.save(model.state_dict(), 'checkpoint.pth') 
            print("Early stopping triggered")
            break

    # model.load_state_dict(torch.load(f"{prefix}_checkpoint.pth"))
    if os.path.exists(ckpt_path) and early_stopper.best is not None:
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))

    return model

