import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from .capsule_layer import CapsuleLayer

class CapsuleNetworkOptimized(nn.Module):
    def __init__(self, input_channels, image_height, image_width, num_classes, routings=3):
        super(CapsuleNetworkOptimized, self).__init__()

        # Reduced convolutional layers
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16,
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32,
                               kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=16,
                               kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # Calculate new height and width after pooling
        self.height_out = image_height // 4
        self.width_out = image_width // 4

        # Reduced capsule input size
        flattened_size = 16 * self.height_out * self.width_out

        # Reduced number of input capsules
        self.input_num_capsules = 16
        self.input_dim_capsules = self.height_out * self.width_out
        
        hidden_dim = flattened_size
        self.fc1 = nn.Linear(flattened_size, hidden_dim)
        # Smaller fully connected layer
        self.fc2 = nn.Linear(hidden_dim, self.input_num_capsules * self.input_dim_capsules)
        # self.fc = nn.Linear(flattened_size, self.input_num_capsules * self.input_dim_capsules)

        # Capsule layer with reduced dimensions
        self.digitcaps = CapsuleLayer(
            input_num_capsules=self.input_num_capsules,
            input_dim_capsules=self.input_dim_capsules,
            num_capsules=num_classes,
            dim_capsules=8,  # Reduced from 16D to 8D
            routings=routings
        )

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        # x = self.relu(self.conv3(x))
        # x = self.pool(x)
        x = self.relu(self.conv4(x))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        
        x = x.view(x.size(0), self.input_num_capsules, self.input_dim_capsules)

        digitcaps = self.digitcaps(x)
        output = torch.norm(digitcaps, dim=-1)
        return output

def build_optimized_model(input_channels, image_height, image_width, num_classes, routings=3):
    return CapsuleNetworkOptimized(
        input_channels=input_channels,
        image_height=image_height,
        image_width=image_width,
        num_classes=num_classes,
        routings=routings
    )

def get_model_info(model, model_name="model", save_path="./"):
    total_params = sum(p.numel() for p in model.parameters())
    model_file = os.path.join(save_path, f"{model_name}.pth")
    torch.save(model.state_dict(), model_file)
    model_size = os.path.getsize(model_file) / (1024 * 1024)

    print(f"**Model Info for {model_name}:**")
    print(f"   - Total Parameters: {total_params:,}")
    print(f"   - Model Size: {model_size:.2f} MB\n")

    return {"total_params": total_params, "model_size_MB": model_size}

# ---- Test optimized model ----
if __name__ == "__main__":
    model_opt = build_optimized_model(input_channels=3, image_height=64, image_width=64, num_classes=3)
    model_info_opt = get_model_info(model_opt, model_name="Optimized_CapsuleNet", save_path="./")
