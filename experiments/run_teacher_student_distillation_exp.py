import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sample_dataset
import os
import src.models.teacher_capsule_model as teacher
import src.models.student_capsule_model as student
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from .utils import EarlyStopping, get_lr_scheduler
import wandb
import timeit
import inference
from datetime import datetime


# 1. Distillation Loss Definition

def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    """
    Compute the knowledge distillation loss.
    """
    ce_loss = F.cross_entropy(student_logits, labels)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    log_student_probs = F.log_softmax(student_logits / temperature, dim=1)

    kd_loss = F.kl_div(
        log_student_probs, 
        teacher_probs, 
        reduction="batchmean"
    ) * (temperature ** 2)

    return alpha * kd_loss + (1.0 - alpha) * ce_loss


# 2. Student Distillation Training Loop

def train_student_with_distillation(
    teacher_model, 
    student_model, 
    train_loader, 
    optimizer, 
    device, 
    temperature=2.0, 
    alpha=0.5
):
    teacher_model.eval()  
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
            temperature=temperature,
            alpha=alpha
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


# 3. Evaluation Function (with Validation Loss)

def evaluate(model, data_loader, device, average='macro'):
    """
    Evaluate the model on a given dataset 
    (accuracy, precision, recall, F1) and compute validation loss.
    """
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0  # Track validation loss
    criterion = nn.CrossEntropyLoss()  # Loss function for validation

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)  # Compute validation loss
            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Average validation loss
    val_loss = running_loss / len(data_loader.dataset)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    correct = (all_preds == all_labels).sum()
    total = all_labels.shape[0]
    accuracy = 100.0 * correct / total

    precision = precision_score(all_labels, all_preds, average=average, zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, average=average, zero_division=0) * 100
    f1 = f1_score(all_labels, all_preds, average=average, zero_division=0) * 100

    return {
        'val_loss': val_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train_model(model, train_loader, val_loader, epochs, device, prefix="teacher"):
    """
    Train a model (teacher or scratch student) with cross-entropy.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=wandb.config['learning_rate'])
    
    print(wandb.config[f'schedule_type_{prefix}'])
    if wandb.config[f'schedule_type_{prefix}'] != "None" :
        
        # Decide which LR scheduler to use based on prefix
        if prefix == "teacher":
            schedule_type = wandb.config['schedule_type_teacher'] 
        elif prefix == "student":
            schedule_type = wandb.config['schedule_type_student'] 
        else:
            schedule_type = wandb.config['schedule_type_distil'] 

        scheduler = get_lr_scheduler(
            optimizer=optimizer,
            schedule_type=schedule_type,
            total_epochs=epochs,
            base_lr=wandb.config['learning_rate'],
            warmup_epochs= wandb.config['warmup_epochs'] * epochs // 100       # 0.01-0.03
        )

    model.train()

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        if wandb.config[f'schedule_type_{prefix}'] != "None" :
            scheduler.step()  # Step LR scheduler after each epoch
        
        epoch_loss = running_loss / len(train_loader.dataset)
        val_metrics = evaluate(model, val_loader, device, average='macro')
        
        print(f"Epoch [{epoch}/{epochs}] - Loss: {epoch_loss:.4f}, "
              f"Val Loss: {val_metrics['val_loss']:.4f}, "
              f"Val Acc: {val_metrics['accuracy']:.2f}%, "
              f"Precision: {val_metrics['precision']:.4f}, "
              f"Recall: {val_metrics['recall']:.4f}, "
              f"F1-score: {val_metrics['f1']:.4f}")

        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({
            # f"{prefix}_epoch": epoch,
            f"{prefix}_train_loss": epoch_loss,
            f"{prefix}_val_loss": val_metrics['val_loss'],
            f"{prefix}_val_accuracy": val_metrics['accuracy']
        })
    
        
    print('\n')
    return model


# 5. The Function Called by W&B Sweeps

def sweep_run():
    """
    This function is called in normal training *and* by W&B sweeps.
    It uses values from `wandb.config` as hyperparameters.
    """
    
    wandb.init()  
    config = wandb.config
    


    # Quick info:
    print(f"--- Starting run with config: {config} ---")

    # Retrieve or set defaults from config
    batch_size = config.get('batch_size', 32)
    epochs = config.get('epochs', 50)
    alpha = config.get('alpha', 0.7)
    temperature = config.get('temperature', 2.0)
    learning_rate = config.get('learning_rate', 1e-4)
    data_path = config.get('data_path', "./data/EdgeCaps_datasets")
    input_channels = config.get('input_channels', 3)
    image_height = config.get('image_height', 64)
    image_width = config.get('image_width', 64)
    num_classes = config.get('num_classes', 3)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, test_loader = sample_dataset.load_dataset(data_path, batch_size)

    # ----- Build Teacher -----
    teacher_model = teacher.build_optimized_model(
        input_channels=input_channels,
        image_height=image_height,
        image_width=image_width,
        num_classes=num_classes,
        routings=3
    ).to(device)

    print('\n Train the Teacher model')
    teacher_model = train_model(
        teacher_model, 
        train_loader, 
        test_loader, 
        epochs, 
        device, 
        prefix="teacher"
    )
    teacher_model.eval()

    # ----- Student from Scratch -----
    student_model_scratch = student.build_ultra_small_student(
        input_channels=input_channels,
        image_height=image_height,
        image_width=image_width,
        num_classes=num_classes,
        routings=3
    ).to(device)

    print('\n Training student from scratch')
    student_model_scratch = train_model(
        student_model_scratch, 
        train_loader, 
        test_loader, 
        epochs, 
        device, 
        prefix="student"
    )

    # ----- Student Distillation -----
    student_model = student.build_ultra_small_student(
        input_channels=input_channels,
        image_height=image_height,
        image_width=image_width,
        num_classes=num_classes,
        routings=3
    ).to(device)

    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)
    
    print('\nStudent distillation loop:')
    for epoch in range(1, epochs + 1):
        train_loss = train_student_with_distillation(
            teacher_model, 
            student_model,
            train_loader, 
            optimizer, 
            device,
            temperature=temperature, 
            alpha=alpha
        )

        val_metrics = evaluate(student_model, test_loader, device, average='macro')
        print(f"Epoch [{epoch}/{epochs}] - Loss: {train_loss:.4f}, "
              f"Val Acc: {val_metrics['accuracy']:.2f}%, "
              f"Precision: {val_metrics['precision']:.4f}, "
              f"Recall: {val_metrics['recall']:.4f}, "
              f"F1-score: {val_metrics['f1']:.4f}")

        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({
            "distill_train_loss": train_loss,
            "distill_val_loss": val_metrics['val_loss'],
            "distill_val_accuracy": val_metrics['accuracy']
        })
       

    # ----- Compare inference speeds -----
    test_data_loader = sample_dataset.test_dataset()  # or test_loader if needed
    inference.compare_model_inference(teacher_model, student_model_scratch, student_model, test_data_loader)

    # Done
    wandb.finish()



# 6. Entry Point

if __name__ == "__main__":
    # If you run "python run_teacher_student_distillation_exp.py" locally, we just call sweep_run().
    # For W&B sweeps, wandb agent will also call this file and run sweep_run() automatically.
    sweep_run()
