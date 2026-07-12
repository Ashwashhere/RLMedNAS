import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from medmnist import OrganAMNIST
from environment.child_network import ChildNet

def get_data_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    train_dataset = OrganAMNIST(split='train', transform=transform, download=True)
    val_dataset = OrganAMNIST(split='val', transform=transform, download=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, val_loader

def train_and_evaluate_child(f_size, k_size, d_val, train_loader, val_loader, device, epochs=5):
    model = ChildNet(f_size, k_size, d_val).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training Pipeline
    for _ in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.view(-1).long().to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            
    # Validation Pipeline
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.view(-1).long().to(device)
            _, predicted = model(inputs).max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = correct / total if total > 0 else 0
    param_count = sum(p.numel() for p in model.parameters())
    
    # Custom efficiency-penalized reward function
    reward = accuracy - (1e-6 * param_count)
    return reward, accuracy, param_count