import os
import torch
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms, datasets

torch.manual_seed(0) 

def load_dataset(data_path, batch_size=32, val_split=0.2):
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    
    # Create dataset
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    print("Classes found:", dataset.classes)

    # Train/Validation Split
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader

def test_dataset(batch_size=1):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    
    data_path = "images"

    # Create dataset
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    print("Classes found:", dataset.classes)

    # Data loaders
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return test_loader

if __name__ == "__main__":
    data_path = "images"
    batch_size = 16
    
    train_loader, val_loader = load_dataset(data_path, batch_size, val_split=0.2)
    
    # Example loop
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx} -> images.shape: {images.shape}, labels.shape: {labels.shape}")
        # break the loop just to show the first batch
        break
