from sklearn.metrics import precision_score, recall_score, f1_score
import math
import torch
from torch.optim import Optimizer
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, save_path='checkpoint.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0
        self.early_stop = False
        self.save_path = save_path

    def step(self, metric, model=None):
        if self.best is None or metric > self.best + self.min_delta:
            self.best = metric
            self.counter = 0
            if model is not None:
                torch.save(model.state_dict(), self.save_path)  # save best model
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


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


def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    ce_loss = F.cross_entropy(student_logits, labels)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    log_student_probs = F.log_softmax(student_logits / temperature, dim=1)
    kd_loss = F.kl_div(log_student_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
    return alpha * kd_loss + (1.0 - alpha) * ce_loss

def train_student_with_distillation(teacher_model, student_model,train_loader, val_loader, optimizer, args, 
                                    device, prefix, epochs=50
):
    teacher_model.eval()

    # early_stopper = EarlyStopping(patience=10 )
    # early_stopper = EarlyStopping(patience=10, save_path=f"{prefix}_checkpoint.pth")
    ckpt_path_kd = f"models_2/{prefix}_checkpoint_{args.job_id}.pth"
   

    early_stopper = EarlyStopping(patience=10, save_path=ckpt_path_kd)

    for epoch in range(1, epochs + 1):
        student_model.train() 

        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                teacher_output = teacher_model(images)

            student_output = student_model(images)

            loss = distillation_loss(
                student_logits=student_output,
                teacher_logits=teacher_output,
                labels=labels,
                temperature=args.temperature,
                alpha=args.alpha
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # validation
        val_metrics = evaluate(student_model, val_loader, device)
        val_acc = val_metrics["accuracy"]

        # early stopping
        early_stopper.step(val_acc, student_model)

        print(f"Epoch [{epoch}/{epochs}] - Loss: {epoch_loss:.4f}, "
              f"Val Acc: {val_acc:.2f}%, "
              f"Precision: {val_metrics['precision']:.2f}, "
              f"Recall: {val_metrics['recall']:.2f}, "
              f"F1-score: {val_metrics['f1']:.2f}")

        if early_stopper.early_stop:
            # torch.save(model.state_dict(), 'checkpoint.pth') 
            print("Early stopping triggered (KD)")
            break

    # load BEST model
    student_model.load_state_dict(torch.load(ckpt_path_kd))

    return student_model

