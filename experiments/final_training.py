import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sample_dataset
import src.models.teacher_capsule_model as teacher
import src.models.student_capsule_model as student
import numpy as np
import yaml
from sklearn.metrics import precision_score, recall_score, f1_score
#from scheduler import get_lr_scheduler
from .utils import EarlyStopping, get_lr_scheduler
import wandb

# Load best parameters from YAML
with open("best_parameters.yaml", "r") as file:
    best_params = yaml.safe_load(file)

teacher_params = best_params["teacher_parameters"]
distill_params = best_params["distill_parameters"]

# Device Configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load Dataset
batch_size = teacher_params["batch_size"]
train_loader, test_loader = sample_dataset.load_dataset("data/EdgeCaps_datasets", batch_size)


# 1. Define Evaluation Function

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


# 2. Train Model Function (Supervised Training)

def train_model(model, train_loader, val_loader, params, device, prefix="teacher"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

    scheduler = get_lr_scheduler(
        optimizer=optimizer,
        schedule_type=params["schedule_type"][prefix],
        total_epochs=params["epochs"],
        base_lr=params["learning_rate"],
        warmup_epochs=params["warmup_epochs"]
    )

    model.train()
    for epoch in range(1, params["epochs"] + 1):
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
        val_acc = evaluate(model, val_loader, device)

        print(f"Epoch [{epoch}/{params['epochs']}] - Loss: {epoch_loss:.4f}, "
              f"Val Acc: {val_acc['accuracy']:.2f}%, "
              f"Precision: {val_acc['precision']:.2f}, "
              f"Recall: {val_acc['recall']:.2f}, "
              f"F1-score: {val_acc['f1']:.2f}")

    return model


# 3. Train Teacher Model
teacher_model = teacher.build_optimized_model(num_classes=3, routings=3).to(device)

print("\nTraining the Teacher model with best parameters...")
teacher_model = train_model(
    teacher_model, train_loader, test_loader, teacher_params, device, prefix="teacher"
)
teacher_model.eval()


# 4. Knowledge Distillation Training

def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    ce_loss = F.cross_entropy(student_logits, labels)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    log_student_probs = F.log_softmax(student_logits / temperature, dim=1)
    kd_loss = F.kl_div(log_student_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
    return alpha * kd_loss + (1.0 - alpha) * ce_loss

def train_student_with_distillation(teacher_model, student_model, train_loader, optimizer, device, params):
    teacher_model.eval()
    student_model.train()

    for epoch in range(1, params["epochs"] + 1):
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
                temperature=params["temperature"],
                alpha=params["alpha"]
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        val_acc = evaluate(student_model, test_loader, device)

        print(f"Epoch [{epoch}/{params['epochs']}] - Loss: {epoch_loss:.4f}, "
              f"Val Acc: {val_acc['accuracy']:.2f}%, "
              f"Precision: {val_acc['precision']:.2f}, "
              f"Recall: {val_acc['recall']:.2f}, "
              f"F1-score: {val_acc['f1']:.2f}")


# 5. Train Student Model with Distillation

student_model = student.build_ultra_small_student(input_channels=3, image_height=64, image_width=64, num_classes=3, routings=3).to(device)
optimizer = optim.Adam(student_model.parameters(), lr=distill_params["learning_rate"])
print(distill_params["learning_rate"])

print("\nTraining the Student model with Knowledge Distillation...")
distilled_model = train_student_with_distillation(teacher_model, student_model, train_loader, optimizer, device, distill_params)



# 6. Final Model Evaluation

print("\nFinal Teacher Model Evaluation:")
teacher_results = evaluate(teacher_model, test_loader, device)
print(teacher_results)

print("\nFinal Distilled Student Model Evaluation:")
student_results = evaluate(student_model, test_loader, device)
print(student_results)


