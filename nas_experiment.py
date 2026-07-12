# Library Imports
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from medmnist import OrganAMNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Experiment Configuration 
EPISODES = 100              # Number of models per experiment phase
CHILD_EPOCHS = 5            # Training epochs per individual model
MASK_FILTER_IDX = 3         # Index 3 = 128 filters (The restricted action)
SAVE_NAME = "nas_full_comparison.csv"

# Device Configuration
# Metal Performance Shaders (MPS) for Mac GPUs else CPU fallback
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
cpu_device = torch.device("cpu")
print(f"Running on: {device}")

# Search Space Definition
SEARCH_SPACE = {
    # Filter sizes in number of channels
    'filters': [16, 32, 64, 128],
    # Kernel sizes
    'kernel_size': [3, 5],
    # Dropout rates
    'dropout': [0.1, 0.25, 0.5]
}

# Data Loading
# Data Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Data Loaders
# Train Set
train_dataset = OrganAMNIST(split='train', transform=transform, download=True)
# Validation Set
val_dataset = OrganAMNIST(split='val', transform=transform, download=True)
# Data Loaders
# Train loader
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
# Validation loader
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, drop_last=True)

# Model Definitions
# Controller Network
class Controller(nn.Module):
    def __init__(self, input_size=1, hidden_size=100):
        super(Controller, self).__init__()
        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size)
        # Output Heads for each hyperparameter
        self.f_head = nn.Linear(hidden_size, len(SEARCH_SPACE['filters']))
        self.k_head = nn.Linear(hidden_size, len(SEARCH_SPACE['kernel_size']))
        self.d_head = nn.Linear(hidden_size, len(SEARCH_SPACE['dropout']))

    # Forward with optional action masking
    def forward(self, apply_mask=False):
        x = torch.zeros(1, 1, 1).to(cpu_device)
        out, _ = self.lstm(x)
        out = out.squeeze(0)
        f_logits, k_logits, d_logits = self.f_head(out), self.k_head(out), self.d_head(out)
        
        if apply_mask:
            f_logits.data[0, MASK_FILTER_IDX] = -1e10  # Apply Action Masking
        return f_logits, k_logits, d_logits

# Child Network
class ChildNet(nn.Module):
    def __init__(self, f_idx, k_idx, d_idx):
        super(ChildNet, self).__init__()
        # Convolutional Layers
        f, k, d = SEARCH_SPACE['filters'][f_idx], SEARCH_SPACE['kernel_size'][k_idx], SEARCH_SPACE['dropout'][d_idx]
        self.conv = nn.Conv2d(1, f, kernel_size=k, padding=k//2)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(d)
        self.fc = nn.Linear(f * 14 * 14, 11)

    # Forward pass
    def forward(self, x):
        x = self.dropout(self.pool(torch.relu(self.conv(x))))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Training and Evaluation Function
def train_child(f_idx, k_idx, d_idx, epochs=5):
    # Instantiate Child Model
    model = ChildNet(f_idx, k_idx, d_idx).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    for _ in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.view(-1).long().to(device)
            optimizer.zero_grad(); loss = criterion(model(inputs), targets); loss.backward(); optimizer.step()
    model.eval()

    # Evaluation
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.view(-1).long().to(device)
            _, predicted = model(inputs).max(1); total += targets.size(0); correct += predicted.eq(targets).sum().item()

    # Calculate Reward, Accuracy, and Parameter Count
    acc = correct / total if total > 0 else 0
    params = sum(p.numel() for p in model.parameters())
    reward = acc - (1e-6 * params)
    return reward, acc, params

# Experiment Execution
full_results = []

# Phase 1: Random Search Baseline
print("\n>>> Phase 1: Random Search Baseline")
# Random Search Loop
for i in range(EPISODES):
    f_idx = random.randint(0, len(SEARCH_SPACE['filters'])-1)
    k_idx = random.randint(0, len(SEARCH_SPACE['kernel_size'])-1)
    d_idx = random.randint(0, len(SEARCH_SPACE['dropout'])-1)
    reward, acc, p_count = train_child(f_idx, k_idx, d_idx, epochs=CHILD_EPOCHS)
    full_results.append({'Method': 'Random', 'Episode': i, 'Filters': SEARCH_SPACE['filters'][f_idx], 'Kernel': SEARCH_SPACE['kernel_size'][k_idx], 'Dropout': SEARCH_SPACE['dropout'][d_idx], 'Accuracy': acc, 'Reward': reward, 'Params': p_count})
    print(f"RS {i}: Reward {reward:.4f}")

# Phase 2: RL Search (No Masking)
print("\n>>> Phase 2: Baseline RL Search")
# RL Search Loop
controller = Controller().to(cpu_device)
optimizer_c = optim.Adam(controller.parameters(), lr=0.01)
# RL Search Loop
for ep in range(EPISODES):
    f_l, k_l, d_l = controller(apply_mask=False)
    d_f, d_k, d_d = Categorical(logits=f_l), Categorical(logits=k_l), Categorical(logits=d_l)
    a_f, a_k, a_d = d_f.sample(), d_k.sample(), d_d.sample()
    reward, acc, p_count = train_child(a_f.item(), a_k.item(), a_d.item(), epochs=CHILD_EPOCHS)
    
    optimizer_c.zero_grad()
    loss = -(d_f.log_prob(a_f) + d_k.log_prob(a_k) + d_d.log_prob(a_d)) * torch.tensor(reward).to(cpu_device)
    loss.backward(); optimizer_c.step()
    
    full_results.append({'Method': 'RL Baseline', 'Episode': ep, 'Filters': SEARCH_SPACE['filters'][a_f.item()], 'Kernel': SEARCH_SPACE['kernel_size'][a_k.item()], 'Dropout': SEARCH_SPACE['dropout'][a_d.item()], 'Accuracy': acc, 'Reward': reward, 'Params': p_count})
    print(f"RL Base {ep}: Reward {reward:.4f}")

# Phase 3: RL Search (With Action Masking)
print("\n>>> Phase 3: Masked RL Search")
# RL Search Loop with Action Masking
controller_masked = Controller().to(cpu_device)
optimizer_cm = optim.Adam(controller_masked.parameters(), lr=0.01)
# RL Search Loop with Action Masking
for ep in range(EPISODES):
    f_l, k_l, d_l = controller_masked(apply_mask=True)
    d_f, d_k, d_d = Categorical(logits=f_l), Categorical(logits=k_l), Categorical(logits=d_l)
    a_f, a_k, a_d = d_f.sample(), d_k.sample(), d_d.sample()
    reward, acc, p_count = train_child(a_f.item(), a_k.item(), a_d.item(), epochs=CHILD_EPOCHS)
    
    optimizer_cm.zero_grad()
    loss = -(d_f.log_prob(a_f) + d_k.log_prob(a_k) + d_d.log_prob(a_d)) * torch.tensor(reward).to(cpu_device)
    loss.backward(); optimizer_cm.step()
    
    # Append Results
    full_results.append({'Method': 'RL Masked', 'Episode': ep, 'Filters': SEARCH_SPACE['filters'][a_f.item()], 'Kernel': SEARCH_SPACE['kernel_size'][a_k.item()], 'Dropout': SEARCH_SPACE['dropout'][a_d.item()], 'Accuracy': acc, 'Reward': reward, 'Params': p_count})
    print(f"RL Masked {ep}: Reward {reward:.4f}")

# Save Results and Generate Visuals
df = pd.DataFrame(full_results)
df.to_csv(SAVE_NAME, index=False)

# Visualisation and Analysis
plt.figure(figsize=(12, 6))
for method in ['Random', 'RL Baseline', 'RL Masked']:
    subset = df[df['Method'] == method]
    plt.plot(subset['Episode'], subset['Reward'].rolling(window=5).mean(), label=f'{method} (Rolling Mean)')

plt.title("NAS Comparison: Random vs RL Baseline vs Masked RL")
plt.xlabel("Episode"); plt.ylabel("Reward"); plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig("nas_comparison_plot.png")
plt.show()