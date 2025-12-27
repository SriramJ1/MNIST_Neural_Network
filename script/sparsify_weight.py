
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)


# ---------- Load dataset ----------
print("Reading MNIST CSV data...")
data = pd.read_csv("./data/mnist_train.csv")

X = data.iloc[:, 1:].values.astype(np.float32)  # pixels
y = data.iloc[:, 0].values                      # labels

# normalize to [0,1]
X /= 255.0


# ---------- Load original dense weights & biases ----------
W1 = np.loadtxt("data/model/weights_hidden.csv", delimiter=",")   # shape: h x n
W2 = np.loadtxt("data/model/weights_output.csv", delimiter=",")   # shape: m x h
b1 = np.loadtxt("data/model/biases_hidden.csv", delimiter=",")    # shape: h
b2 = np.loadtxt("data/model/biases_output.csv", delimiter=",")    # shape: m

print(f"W1 shape: {W1.shape}, W2 shape: {W2.shape}")


# ---------- Activations ----------
def tanh(x):
    return np.tanh(x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------- Forward pass using given W1/W2 ----------
def forward(X_batch: np.ndarray, W1_curr: np.ndarray, W2_curr: np.ndarray) -> np.ndarray:
    """
    Same architecture as dense_nn.py, but takes W1/W2 as arguments.

    X_batch: [N, n]
    W1_curr: [h, n]
    W2_curr: [m, h]
    """
    H = tanh(X_batch @ W1_curr.T + b1)       # [N, h]
    Z = sigmoid(H @ W2_curr.T + b2)          # [N, m]
    predY = np.argmax(Z, axis=1)
    return predY


# ---------- Helper: magnitude-based pruning ----------
def prune_to_sparsity(W: np.ndarray, target_sparsity: float):
    """
    Prune weights in W by magnitude to reach approximately target_sparsity.

    target_sparsity: fraction in [0,1], e.g., 0.8 for 80%

    Returns:
        W_pruned: pruned weight matrix (same shape as W)
        actual_sparsity: achieved sparsity (fraction of zeros)
    """
    W_flat = np.abs(W).ravel()
    total = W_flat.size
    k = int(target_sparsity * total)

    if k <= 0:
        # nothing to prune
        W_pruned = W.copy()
        actual_sparsity = 0.0
        return W_pruned, actual_sparsity

    # threshold = k-th smallest magnitude
    # np.partition is O(n) and efficient
    thresh = np.partition(W_flat, k - 1)[k - 1]

    # strictly greater than threshold to avoid over-pruning when many equal values
    mask = np.abs(W) > thresh
    W_pruned = W * mask

    nnz = np.count_nonzero(W_pruned)
    actual_sparsity = 1.0 - nnz / W_pruned.size

    return W_pruned, actual_sparsity


# ---------- Baseline dense accuracy ----------
print("Computing baseline dense accuracy...")
dense_pred = forward(X, W1, W2)
dense_acc = np.mean(dense_pred == y)
print(f"Baseline dense accuracy: {dense_acc * 100:.2f}%\n")


# ---------- Prune and evaluate for each sparsity level ----------
os.makedirs("data/model", exist_ok=True)
os.makedirs("logs", exist_ok=True)

results = []

for sparsity_pct in range(50, 100, 5):  # 50, 55, ..., 95
    target = sparsity_pct / 100.0

    print(f"=== Target sparsity: {sparsity_pct}% ===")

    # Prune W1 and W2 separately to target sparsity
    W1_pruned, sp1 = prune_to_sparsity(W1, target)
    W2_pruned, sp2 = prune_to_sparsity(W2, target)

    # Evaluate accuracy with pruned weights
    pred = forward(X, W1_pruned, W2_pruned)
    acc = np.mean(pred == y)

    print(
        f"  W1 sparsity: {sp1 * 100:.2f}%, "
        f"W2 sparsity: {sp2 * 100:.2f}%, "
        f"accuracy: {acc * 100:.2f}%"
    )

    # Save dense CSVs for C++ (as required)
    out_W1 = f"data/model/{sparsity_pct}_W1.csv"
    out_W2 = f"data/model/{sparsity_pct}_W2.csv"
    np.savetxt(out_W1, W1_pruned, delimiter=",")
    np.savetxt(out_W2, W2_pruned, delimiter=",")
    print(f"  Saved: {out_W1}, {out_W2}")

    results.append((sparsity_pct, sp1, sp2, acc))

# Optionally save a small log for plotting accuracy vs sparsity later
log_path = "logs/sparsity_accuracy_python.csv"
with open(log_path, "w") as f:
    f.write("target_sparsity_pct,W1_sparsity,W2_sparsity,accuracy\n")
    for (pct, sp1, sp2, acc) in results:
        f.write(f"{pct},{sp1},{sp2},{acc}\n")

print(f"\nSaved accuracy log to {log_path}")

# ========== NEW: Plotting Functions ==========

def plot_sparsity_accuracy_tradeoff(results, dense_acc, output_dir="plots"):
    """
    Plot accuracy vs sparsity tradeoff curve showing degradation.
    
    Args:
        results: list of (sparsity_pct, sp1, sp2, acc) tuples
        dense_acc: baseline dense accuracy for comparison
        output_dir: output directory for plots
    """
    sparsity_levels = [r[0] for r in results]
    accuracies = [r[3] * 100 for r in results]  # Convert to percentage
    
    plt.figure(figsize=(12, 7))
    
    # Plot accuracy vs sparsity
    plt.plot(sparsity_levels, accuracies, marker="o", linewidth=2.5, markersize=10, 
            color="#2E86AB", label="Sparse NN Accuracy")
    
    # Plot baseline dense accuracy as horizontal line
    plt.axhline(y=dense_acc * 100, color="#D62828", linestyle="--", linewidth=2, 
               label=f"Dense NN Baseline ({dense_acc*100:.2f}%)")
    
    # Formatting
    plt.xlabel("Sparsity Level (%)", fontsize=13, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=13, fontweight="bold")
    plt.title("Neural Network Accuracy vs Weight Sparsity", fontsize=15, fontweight="bold")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(fontsize=12, loc="lower left")
    plt.ylim([0, 105])
    plt.xticks(sparsity_levels, fontsize=11)
    plt.yticks(fontsize=11)
    
    # Add value labels on each point
    for x, y in zip(sparsity_levels, accuracies):
        plt.text(x, y + 1.5, f"{y:.1f}%", ha="center", va="bottom", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sparsity_accuracy_tradeoff.png"), dpi=200)
    plt.close()
    print(f"Saved {os.path.join(output_dir, 'sparsity_accuracy_tradeoff.png')}")


def plot_sparsity_metrics(results, output_dir="plots"):
    """
    Plot W1/W2 sparsities vs target sparsity showing actual achieved sparsity.
    """
    sparsity_levels = [r[0] for r in results]
    w1_sparsities = [r[1] * 100 for r in results]  # Convert to percentage
    w2_sparsities = [r[2] * 100 for r in results]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(sparsity_levels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, w1_sparsities, width, label="W1 (Hidden Layer)", 
                   color="#06A77D", alpha=0.8)
    bars2 = ax.bar(x + width/2, w2_sparsities, width, label="W2 (Output Layer)", 
                   color="#F18F01", alpha=0.8)
    
    # Plot target sparsity as reference line
    ax.plot(x, sparsity_levels, marker="x", linestyle="--", linewidth=2, 
           color="#D62828", markersize=10, label="Target Sparsity", zorder=5)
    
    # Formatting
    ax.set_xlabel("Pruning Configuration", fontsize=13, fontweight="bold")
    ax.set_ylabel("Sparsity Level (%)", fontsize=13, fontweight="bold")
    ax.set_title("Achieved Weight Sparsity Levels by Layer", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}%" for s in sparsity_levels], fontsize=11)
    ax.set_ylim([0, 105])
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f"{height:.1f}%", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f"{height:.1f}%", ha="center", va="bottom", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "achieved_sparsity_by_layer.png"), dpi=200)
    plt.close()
    print(f"Saved {os.path.join(output_dir, 'achieved_sparsity_by_layer.png')}")


def plot_accuracy_degradation(results, dense_acc, output_dir="plots"):
    """
    Plot accuracy degradation percentage (relative to dense baseline).
    """
    sparsity_levels = [r[0] for r in results]
    accuracies = [r[3] for r in results]
    degradation = [(dense_acc - acc) / dense_acc * 100 for acc in accuracies]
    
    plt.figure(figsize=(12, 7))
    
    # Use color gradient to show severity
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sparsity_levels)))
    bars = plt.bar(sparsity_levels, degradation, color=colors, edgecolor="black", linewidth=1.5, width=3)
    
    # Formatting
    plt.xlabel("Sparsity Level (%)", fontsize=13, fontweight="bold")
    plt.ylabel("Accuracy Degradation (%)", fontsize=13, fontweight="bold")
    plt.title("Accuracy Loss Relative to Dense Baseline", fontsize=15, fontweight="bold")
    plt.grid(True, alpha=0.3, axis="y", linestyle="--")
    plt.xticks(sparsity_levels, fontsize=11)
    plt.yticks(fontsize=11)
    
    # Add value labels on bars
    for bar, deg in zip(bars, degradation):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f"{deg:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_degradation.png"), dpi=200)
    plt.close()
    print(f"Saved {os.path.join(output_dir, 'accuracy_degradation.png')}")


def plot_sparsity_accuracy_combined(results, dense_acc, output_dir="plots"):
    """
    Plot both sparsity levels and accuracy on same figure with dual y-axes.
    """
    sparsity_levels = [r[0] for r in results]
    w1_sparsities = [r[1] * 100 for r in results]
    accuracies = [r[3] * 100 for r in results]
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Left y-axis: Accuracy
    color1 = "#2E86AB"
    ax1.plot(sparsity_levels, accuracies, marker="o", linewidth=2.5, markersize=10,
            color=color1, label="Accuracy", zorder=3)
    ax1.axhline(y=dense_acc * 100, color=color1, linestyle="--", linewidth=2, alpha=0.5)
    ax1.set_xlabel("Target Sparsity Level (%)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", fontsize=13, fontweight="bold", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim([0, 105])
    
    # Right y-axis: Sparsity
    ax2 = ax1.twinx()
    color2 = "#D62828"
    ax2.plot(sparsity_levels, w1_sparsities, marker="s", linewidth=2.5, markersize=10,
            color=color2, label="W1 Achieved Sparsity", zorder=3)
    ax2.plot(sparsity_levels, sparsity_levels, marker="x", linewidth=2, linestyle=":",
            color=color2, alpha=0.5, label="Target Sparsity", zorder=2)
    ax2.set_ylabel("Sparsity Level (%)", fontsize=13, fontweight="bold", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim([0, 105])
    
    # Title and grid
    plt.title("Accuracy and Sparsity vs Target Sparsity Level", fontsize=15, fontweight="bold")
    ax1.grid(True, alpha=0.3, linestyle="--")
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sparsity_accuracy_combined.png"), dpi=200)
    plt.close()
    print(f"Saved {os.path.join(output_dir, 'sparsity_accuracy_combined.png')}")


# Generate all plots
print("\n" + "="*50)
print("Generating sparsity analysis plots...")
print("="*50)

plot_sparsity_accuracy_tradeoff(results, dense_acc)
plot_sparsity_metrics(results)
plot_accuracy_degradation(results, dense_acc)
plot_sparsity_accuracy_combined(results, dense_acc)

print("\nAll sparsity analysis plots generated successfully!")

