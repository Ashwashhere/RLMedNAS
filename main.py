import os
import random
import torch
import numpy as np
import pandas as pd
from torch.distributions import Categorical

# Modular Imports
from controllers.rnn_agent import Controller
from environment.evaluator import get_data_loaders, train_and_evaluate_child
from utils.plotting import generate_nas_comparison_plot

# Hyperparameter Setup
EPISODES = 100
CHILD_EPOCHS = 5
MASK_FILTER_IDX = 3
SAVE_CSV = "nas_full_comparison.csv"
SAVE_PLOT = "nas_comparison_plot.png"

SEARCH_SPACE = {
    'filters': [16, 32, 64, 128],
    'kernel_size': [3, 5],
    'dropout': [0.1, 0.25, 0.5]
}

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
cpu_device = torch.device("cpu")
print(f"Execution Target Device: {device}")

def main():
    train_loader, val_loader = get_data_loaders(batch_size=128)
    full_results = []

    # --- PHASE 1: RANDOM SEARCH ---
    print("\n>>> Running Phase 1: Random Search Baseline...")
    for i in range(EPISODES):
        f_idx = random.randint(0, len(SEARCH_SPACE['filters'])-1)
        k_idx = random.randint(0, len(SEARCH_SPACE['kernel_size'])-1)
        d_idx = random.randint(0, len(SEARCH_SPACE['dropout'])-1)
        
        reward, acc, p_count = train_and_evaluate_child(
            SEARCH_SPACE['filters'][f_idx], SEARCH_SPACE['kernel_size'][k_idx], SEARCH_SPACE['dropout'][d_idx],
            train_loader, val_loader, device, epochs=CHILD_EPOCHS
        )
        full_results.append({
            'Method': 'Random', 'Episode': i, 
            'Filters': SEARCH_SPACE['filters'][f_idx], 'Kernel': SEARCH_SPACE['kernel_size'][k_idx], 
            'Dropout': SEARCH_SPACE['dropout'][d_idx], 'Accuracy': acc, 'Reward': reward, 'Params': p_count
        })

    # --- PHASE 2: BASELINE RL (UNCONSTRAINED) ---
    print("\n>>> Running Phase 2: Baseline RL Search...")
    controller = Controller(len(SEARCH_SPACE['filters']), len(SEARCH_SPACE['kernel_size']), len(SEARCH_SPACE['dropout']))
    optimizer_c = torch.optim.Adam(controller.parameters(), lr=0.01)
    
    for ep in range(EPISODES):
        f_logits, k_logits, d_logits = controller(apply_mask=False)
        dist_f, dist_k, dist_d = Categorical(logits=f_logits), Categorical(logits=k_logits), Categorical(logits=d_logits)
        act_f, act_k, act_d = dist_f.sample(), dist_k.sample(), dist_d.sample()
        
        reward, acc, p_count = train_and_evaluate_child(
            SEARCH_SPACE['filters'][act_f.item()], SEARCH_SPACE['kernel_size'][act_k.item()], SEARCH_SPACE['dropout'][act_d.item()],
            train_loader, val_loader, device, epochs=CHILD_EPOCHS
        )
        
        optimizer_c.zero_grad()
        loss = -(dist_f.log_prob(act_f) + dist_k.log_prob(act_k) + dist_d.log_prob(act_d)) * torch.tensor(reward).to(cpu_device)
        loss.backward()
        optimizer_c.step()
        
        full_results.append({
            'Method': 'RL Baseline', 'Episode': ep, 
            'Filters': SEARCH_SPACE['filters'][act_f.item()], 'Kernel': SEARCH_SPACE['kernel_size'][act_k.item()], 
            'Dropout': SEARCH_SPACE['dropout'][act_d.item()], 'Accuracy': acc, 'Reward': reward, 'Params': p_count
        })

    # --- PHASE 3: MASKED RL (CONSTRAINED SEARCH SPACE) ---
    print("\n>>> Running Phase 3: Masked RL Search...")
    controller_masked = Controller(len(SEARCH_SPACE['filters']), len(SEARCH_SPACE['kernel_size']), len(SEARCH_SPACE['dropout']))
    optimizer_cm = torch.optim.Adam(controller_masked.parameters(), lr=0.01)
    
    for ep in range(EPISODES):
        f_logits, k_logits, d_logits = controller_masked(apply_mask=True, mask_filter_idx=MASK_FILTER_IDX)
        dist_f, dist_k, dist_d = Categorical(logits=f_logits), Categorical(logits=k_logits), Categorical(logits=d_logits)
        act_f, act_k, act_d = dist_f.sample(), dist_k.sample(), dist_d.sample()
        
        reward, acc, p_count = train_and_evaluate_child(
            SEARCH_SPACE['filters'][act_f.item()], SEARCH_SPACE['kernel_size'][act_k.item()], SEARCH_SPACE['dropout'][act_d.item()],
            train_loader, val_loader, device, epochs=CHILD_EPOCHS
        )
        
        optimizer_cm.zero_grad()
        loss = -(dist_f.log_prob(act_f) + dist_k.log_prob(act_k) + dist_d.log_prob(act_d)) * torch.tensor(reward).to(cpu_device)
        loss.backward()
        optimizer_cm.step()
        
        full_results.append({
            'Method': 'RL Masked', 'Episode': ep, 
            'Filters': SEARCH_SPACE['filters'][act_f.item()], 'Kernel': SEARCH_SPACE['kernel_size'][act_k.item()], 
            'Dropout': SEARCH_SPACE['dropout'][act_d.item()], 'Accuracy': acc, 'Reward': reward, 'Params': p_count
        })

    # Save outputs and run visualization tracking code
    df = pd.DataFrame(full_results)
    df.to_csv(SAVE_CSV, index=False)
    generate_nas_comparison_plot(SAVE_CSV, SAVE_PLOT)

if __name__ == "__main__":
    main()