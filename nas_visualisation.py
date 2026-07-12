import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Setup 
FILE_PATH = "nas_full_comparison.csv"
SAVE_PREFIX = "nas_report_"

# Set visual style for academic standards
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

def generate_report_visuals():
    try:
        df = pd.read_csv(FILE_PATH)
    except FileNotFoundError:
        print(f"Error: {FILE_PATH} not found. Please run the experiment script first.")
        return

    print("Generating report visuals...")

    # Performance Comparison (Learning Curves)
    plt.figure(figsize=(12, 6))
    for method in df['Method'].unique():
        subset = df[df['Method'] == method]
        # Use a rolling mean to show trends clearly
        rolling_mean = subset['Reward'].rolling(window=10, min_periods=1).mean()
        plt.plot(subset['Episode'], rolling_mean, label=f'{method} (Rolling Avg)', linewidth=2)
    
    plt.title("Search Performance: Multi-Objective Reward vs. Iteration")
    plt.xlabel("Episode (Architecture Iteration)")
    plt.ylabel("Reward (Accuracy - Penalty)")
    plt.legend(loc='lower right')
    plt.savefig(f"{SAVE_PREFIX}learning_curves.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Efficiency Frontier (Accuracy vs. Complexity) 
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df, x='Params', y='Accuracy', hue='Method', style='Method', s=100, alpha=0.7)
    
    # Highlight the "Ideal" zone (Top-Left)
    plt.title("Efficiency Frontier: Model Accuracy vs. Parameter Count")
    plt.xlabel("Total Parameters")
    plt.ylabel("Validation Accuracy")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"{SAVE_PREFIX}efficiency_frontier.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Hyperparameter Preference Analysis 
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Filter Choice Distribution
    sns.countplot(data=df, x='Filters', hue='Method', ax=axes[0])
    axes[0].set_title("Filter Size Distribution")
    
    # Kernel Choice Distribution
    sns.countplot(data=df, x='Kernel', hue='Method', ax=axes[1])
    axes[1].set_title("Kernel Size Distribution")
    
    # Dropout Choice Distribution
    sns.countplot(data=df, x='Dropout', hue='Method', ax=axes[2])
    axes[2].set_title("Dropout Rate Distribution")
    
    plt.suptitle("Knowledge Acquisition: Hyperparameter Selection by Agent Type", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{SAVE_PREFIX}preferences.png", dpi=300)
    plt.close()

    # Statistical Summary Tables 
    # Metric Summary
    stats = df.groupby('Method').agg({
        'Accuracy': ['mean', 'max', 'std'],
        'Params': ['mean', 'min', 'std'],
        'Reward': ['mean', 'max']
    }).round(4)
    
    print("\n--- Summary Statistics Table ---")
    print(stats)
    stats.to_csv(f"{SAVE_PREFIX}summary_stats.csv")

    # Best Architecture per Method
    best_models = df.sort_values('Reward', ascending=False).drop_duplicates('Method')
    print("\n--- Best Performing Architectures ---")
    print(best_models[['Method', 'Filters', 'Kernel', 'Dropout', 'Accuracy', 'Params']])
    best_models.to_csv(f"{SAVE_PREFIX}best_architectures.csv", index=False)

    print(f"\nAll files saved with prefix '{SAVE_PREFIX}'")

if __name__ == "__main__":
    generate_report_visuals()