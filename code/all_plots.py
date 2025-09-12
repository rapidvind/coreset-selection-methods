import pandas as pd
import matplotlib.pyplot as plt
import os

# Folder to save plots
save_path = r"C:\Users\Aravind Dora\Desktop\DataScieneThesis_Coreset_methods\plots"
os.makedirs(save_path, exist_ok=True)  # Create folder if it doesn't exist

# Manually inputting table data
data = {
    "Dataset": ["MNIST", "MNIST", "MNIST", "MNIST", 
                "CIFAR-10", "CIFAR-10", "CIFAR-10", "CIFAR-10"],
    "Method": ["Full Dataset", "Random Sampling", "k-Center Greedy", "Herding",
               "Full Dataset", "Random Sampling", "k-Center Greedy", "Herding"],
    "Subset Size": [60000, 6000, 6000, 6000, 50000, 12500, 12500, 12500],
    "Epochs": [5]*8,
    "Avg Epoch Time (s)": [42, 3.7, 3.75, 3.6, 30.3, 7.21, 7.442, 7.38],
    "Final Accuracy (%)": [98.78, 97.65, 95.51, 97.75, 72.51, 62.25, 58.06, 59.99],
    "Total Training Time (s)": [220, 18.5, 19, 18, 150, 60, 3700, 207]
}

# Define DataFrame
df = pd.DataFrame(data)

# Separate datasets
mnist_df = df[df["Dataset"] == "MNIST"]
cifar_df = df[df["Dataset"] == "CIFAR-10"]

# 1. Accuracy comparison (bar chart)
plt.figure(figsize=(8,5))
for dataset, subdf in df.groupby("Dataset"):
    plt.bar(subdf["Method"] + " (" + dataset + ")", subdf["Final Accuracy (%)"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Final Accuracy (%)")
plt.title("Final Accuracy by Method (MNIST vs CIFAR-10)")
plt.tight_layout()
plt.savefig(os.path.join(save_path, "accuracy_comparison.png"))
plt.show()

# 2. Training time comparison (bar chart)
plt.figure(figsize=(8,5))
for dataset, subdf in df.groupby("Dataset"):
    plt.bar(subdf["Method"] + " (" + dataset + ")", subdf["Total Training Time (s)"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Total Training Time (s)")
plt.title("Training Time by Method (MNIST vs CIFAR-10)")
plt.tight_layout()
plt.savefig(os.path.join(save_path, "training_time_comparison.png"))
plt.show()

# 3. Accuracy vs Training Time scatter plot
plt.figure(figsize=(7,5))
for dataset, subdf in df.groupby("Dataset"):
    plt.scatter(subdf["Total Training Time (s)"], subdf["Final Accuracy (%)"], label=dataset)
    for i, row in subdf.iterrows():
        plt.text(row["Total Training Time (s)"], row["Final Accuracy (%)"], row["Method"], fontsize=8)
plt.xlabel("Total Training Time (s)")
plt.ylabel("Final Accuracy (%)")
plt.title("Accuracy vs Training Time Trade-off")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_path, "accuracy_vs_training_time.png"))
plt.show()

# 4. Side-by-side bar chart for Accuracy
fig, ax = plt.subplots(1, 2, figsize=(12,5), sharey=True)
ax[0].bar(mnist_df["Method"], mnist_df["Final Accuracy (%)"], color="skyblue")
ax[0].set_title("MNIST Accuracy")
ax[0].tick_params(axis="x", rotation=45)
ax[1].bar(cifar_df["Method"], cifar_df["Final Accuracy (%)"], color="salmon")
ax[1].set_title("CIFAR-10 Accuracy")
ax[1].tick_params(axis="x", rotation=45)
plt.suptitle("Final Accuracy Comparison Across Methods")
plt.tight_layout()
plt.savefig(os.path.join(save_path, "side_by_side_accuracy.png"))
plt.show()
