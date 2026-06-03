import torch
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms, datasets
torch.manual_seed(0)

def test_dataset(data_path, batch_size=1):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])


    # Create dataset
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    print("Classes found:", dataset.classes)

    # Data loaders
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return test_loader

def load_dataset(data_path, batch_size=32, val_split=0.1, test_split=0.1):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=data_path, transform=transform)

    total_size = len(dataset)
    val_size = int(val_split * total_size)
    test_size = int(test_split * total_size)
    train_size = total_size - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

