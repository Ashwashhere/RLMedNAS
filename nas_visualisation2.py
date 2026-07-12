import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the experimental results
df = pd.read_csv("nas_full_comparison.csv")

# Set the professional aesthetic
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))

# Define colors and markers for clarity
palette = {'Random': '#4c72b0', 'RL Baseline': '#dd8452', 'RL Masked': '#55a868'}
markers = {'Random': 'o', 'RL Baseline': 'X', 'RL Masked': 's'}

# Create the scatter plot
sns.scatterplot(
    data=df, 
    x='Params', 
    y='Accuracy', 
    hue='Method', 
    style='Method',
    palette=palette,
    markers=markers,
    alpha=0.7,
    s=100
)

# Labeling and Formatting
plt.title("Efficiency Frontier: Model Accuracy vs. Parameter Count", fontsize=14, pad=15)
plt.xlabel("Total Trainable Parameters (Model Complexity)", fontsize=12)
plt.ylabel("Validation Accuracy (OrganAMNIST)", fontsize=12)
plt.legend(title="Search Strategy", frameon=True)

# Add a text annotation for the 'Champion' zone
plt.annotate('Optimal Efficient Zone\n(High Accuracy, Low Params)', 
             xy=(45000, 0.965), xytext=(100000, 0.945),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8))

plt.tight_layout()
plt.savefig("nas_efficiency_frontier.png", dpi=300)
plt.show()