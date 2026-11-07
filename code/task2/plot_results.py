import matplotlib.pyplot as plt

def plot_metrics(results, ranks=[32, 64, 128], output_prefix="task2"):
    layers = list(range(1, 13))
    
    # 1. Plot UUAS vs Layer
    plt.figure(figsize=(8, 5))
    for r in ranks:
        plt.plot(layers, results[r]["uuas"], marker="o", label=f"Rank k={r}")
    plt.title("UUAS vs. BERT Layer")
    plt.xlabel("Layer l")
    plt.ylabel("UUAS")
    plt.xticks(layers)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.savefig(f"{output_prefix}_uuas_vs_layer.png", dpi=300)
    plt.close()

    # 2. Plot Depth Correlation vs Layer
    plt.figure(figsize=(8, 5))
    for r in ranks:
        plt.plot(layers, results[r]["depth_spearman"], marker="s", label=f"Rank k={r}")
    plt.title("Depth-Probe Spearman Correlation vs. BERT Layer")
    plt.xlabel("Layer l")
    plt.ylabel("Spearman Correlation")
    plt.xticks(layers)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.savefig(f"{output_prefix}_depth_vs_layer.png", dpi=300)
    plt.close()