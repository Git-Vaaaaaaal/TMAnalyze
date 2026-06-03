import pandas as pd
import numpy as np
import os
from itertools import combinations
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# ---------------------------
# CONFIG
# ---------------------------
MARKERS = ["BCL2", "BCL6", "CD10", "HE", "MUM1", "MYC"]

LABELS_CSV = "csv/id_label_patient_complete.csv"  # optionnel

FILENAME_COL = "wsi_name"
LABEL_COL = "status"
GROUP_COL = "group"

ENCODER = "virchow2"


# ---------------------------
# PATH BUILDER
# ---------------------------
def get_tiles_dir(marker, encoder):
    return f"data_224/{encoder}/{marker}/features_{encoder}"


# ---------------------------
# LOAD DATA
# ---------------------------
def load_slide_features(tiles_dir):
    """Charge tous les CSV patch-level d'un dossier et retourne un DataFrame
    slide-level (une ligne par slide = mean pooling des patches).

    Format CSV attendu : col 0 = x, col 1 = y, col 2+ = features.
    Le nom du fichier (sans .csv) est utilisé comme patient_id.
    """
    records = []
    for csv_path in sorted(Path(tiles_dir).glob("*.csv")):
        patient_id = csv_path.stem
        df = pd.read_csv(csv_path)
        feats = df.iloc[:, 2:].values.astype(np.float32)  # ignore x, y
        mean_feat = feats.mean(axis=0)
        records.append({FILENAME_COL: patient_id, **{f"f{i}": v for i, v in enumerate(mean_feat)}})
    return pd.DataFrame(records)


def load_data_dual(tiles_dir_A, tiles_dir_B, filename_col, label_col=None, labels_csv=None):
    df_A = load_slide_features(tiles_dir_A)
    df_B = load_slide_features(tiles_dir_B)

    df_A[GROUP_COL] = "A"
    df_B[GROUP_COL] = "B"

    df = pd.concat([df_A, df_B], ignore_index=True)

    if labels_csv is not None and label_col is not None:
        labels_df = pd.read_csv(labels_csv)
        labels_df = labels_df[[filename_col, label_col]]
        labels_df[filename_col] = labels_df[filename_col].astype(str)
        df[filename_col] = df[filename_col].astype(str)
        df = df.merge(labels_df, on=filename_col, how="left")

    drop_cols = [filename_col]
    if label_col and label_col in df.columns:
        drop_cols.append(label_col)

    X = df.drop(columns=drop_cols + [GROUP_COL])
    X = X.select_dtypes(include=["number"]).values

    y = df[GROUP_COL].values  # classification entre marqueurs

    return X, y

# ---------------------------
# CLASSIFICATION
# ---------------------------
def classification_test(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)

# ---------------------------
# CLUSTERING
# ---------------------------
def clustering_test(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return silhouette_score(X_scaled, y_encoded)

# ---------------------------
# VISUALIZATION
# ---------------------------
def compute_projections(X, y):
    """Retourne (X_pca, X_tsne, y_encoded) pour une paire de marqueurs."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_pca  = PCA(n_components=2).fit_transform(X_scaled)
    X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_scaled)

    return X_pca, X_tsne, y_encoded


def plot_all(viz_data, encoder):
    """Crée une seule figure avec toutes les PCA et t-SNE côte à côte."""
    n = len(viz_data)
    fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n))
    fig.suptitle(f"Embeddings — {encoder}", fontsize=14, fontweight="bold")

    for row, (m1, m2, X_pca, X_tsne, y_enc) in enumerate(viz_data):
        ax_pca, ax_tsne = axes[row] if n > 1 else axes

        sc1 = ax_pca.scatter(X_pca[:, 0],  X_pca[:, 1],  c=y_enc, cmap="tab10", s=20)
        ax_pca.set_title(f"PCA  {m1} vs {m2}", fontsize=9)
        ax_pca.axis("off")
        plt.colorbar(sc1, ax=ax_pca, fraction=0.03)

        sc2 = ax_tsne.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_enc, cmap="tab10", s=20)
        ax_tsne.set_title(f"t-SNE  {m1} vs {m2}", fontsize=9)
        ax_tsne.axis("off")
        plt.colorbar(sc2, ax=ax_tsne, fraction=0.03)

    plt.tight_layout()
    os.makedirs(os.path.join("figure", encoder), exist_ok=True)
    save_path = os.path.join("figure", encoder, "all_projections.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure sauvegardée → {save_path}")

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":

    acc_matrix = pd.DataFrame(index=MARKERS, columns=MARKERS, dtype=float)
    sil_matrix = pd.DataFrame(index=MARKERS, columns=MARKERS, dtype=float)
    viz_data   = []

    for m1, m2 in combinations(MARKERS, 2):
        print(f"\n=== {m1} vs {m2} ===")

        path_A = get_tiles_dir(m1, ENCODER)
        path_B = get_tiles_dir(m2, ENCODER)

        if not os.path.isdir(path_A) or not os.path.isdir(path_B):
            print("Missing folder, skipping...")
            continue

        X, y = load_data_dual(
            path_A,
            path_B,
            FILENAME_COL,
            LABEL_COL,
            LABELS_CSV
        )

        print(f"Dataset shape: {X.shape}")

        acc = classification_test(X, y)
        sil = clustering_test(X, y)

        print(f"Accuracy: {acc:.4f}")
        print(f"Silhouette: {sil:.4f}")

        acc_matrix.loc[m1, m2] = acc
        acc_matrix.loc[m2, m1] = acc

        sil_matrix.loc[m1, m2] = sil
        sil_matrix.loc[m2, m1] = sil

        X_pca, X_tsne, y_enc = compute_projections(X, y)
        viz_data.append((m1, m2, X_pca, X_tsne, y_enc))

    # diagonale
    np.fill_diagonal(acc_matrix.values, 1.0)
    np.fill_diagonal(sil_matrix.values, 0.0)

    print("\n=== Accuracy Matrix ===")
    print(acc_matrix)

    print("\n=== Silhouette Matrix ===")
    print(sil_matrix)

    acc_matrix.to_csv("accuracy_matrix.csv")
    sil_matrix.to_csv("silhouette_matrix.csv")

    if viz_data:
        plot_all(viz_data, ENCODER)