import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# LOAD MATRICES
# ---------------------------
acc_matrix = pd.read_csv("accuracy_matrix.csv", index_col=0)
sil_matrix = pd.read_csv("silhouette_matrix.csv", index_col=0)

# convertir en float (au cas où strings / NaN)
acc_matrix = acc_matrix.astype(float)
sil_matrix = sil_matrix.astype(float)

# ---------------------------
# FUNCTION HEATMAP
# ---------------------------
def plot_heatmap(matrix, title, cmap="viridis", center=None, filename=None):
    plt.figure(figsize=(8, 6))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=center,
        square=True,
        linewidths=0.5,
        cbar=True
    )

    plt.title(title)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300)

    plt.show()

# ---------------------------
# PLOT
# ---------------------------
plot_heatmap(
    acc_matrix,
    title="Accuracy Matrix (Marker vs Marker)",
    cmap="coolwarm",
    filename="accuracy_heatmap.png"
)

plot_heatmap(
    sil_matrix,
    title="Silhouette Matrix (Marker vs Marker)",
    cmap="viridis",
    center=0,
    filename="silhouette_heatmap.png"
)