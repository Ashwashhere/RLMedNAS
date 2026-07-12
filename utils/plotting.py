import pandas as pd
import matplotlib.pyplot as plt

def generate_nas_comparison_plot(csv_path, save_path):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(12, 6))
    
    for method in ['Random', 'RL Baseline', 'RL Masked']:
        subset = df[df['Method'] == method]
        if not subset.empty:
            plt.plot(
                subset['Episode'], 
                subset['Reward'].rolling(window=5).mean(), 
                label=f'{method} (Rolling Mean)'
            )

    plt.title("NAS Comparison: Random vs RL Baseline vs Masked RL")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()
    print(f"Convergence plot successfully generated at: {save_path}")