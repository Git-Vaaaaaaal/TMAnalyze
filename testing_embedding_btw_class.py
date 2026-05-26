import pandas as pd
import numpy as np
import os
from itertools import combinations

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

# ---------------------------
# PATH BUILDER
# ---------------------------
def get_feature_path(marker):
    return f"data/prism/{marker}/slide_features_prism/prism_encoder.csv"

# ---------------------------
# LOAD DATA
# ---------------------------
def load_data_dual(features_csv_A, features_csv_B, filename_col, label_col=None, labels_csv=None):
    df_A = pd.read_csv(features_csv_A)
    df_B = pd.read_csv(features_csv_B)

    df_A[GROUP_COL] = "A"
    df_B[GROUP_COL] = "B"

    df = pd.concat([df_A, df_B], ignore_index=True)

    if labels_csv is not None and label_col is not None:
        labels_df = pd.read_csv(labels_csv)
        labels_df = labels_df[[filename_col, label_col]]
        df = df.merge(labels_df, on=filename_col, how="left")

    drop_cols = [filename_col]
    if label_col and label_col in df.columns:
        drop_cols.append(label_col)

    X = df.drop(columns=drop_cols + [GROUP_COL])
    X = X.select_dtypes(include=['number']).values

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
# VISUALIZATION (optionnel)
# ---------------------------
def visualize(X, y, marker_a, marker_b):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    os.makedirs("figure", exist_ok=True)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure()
    plt.title(f"PCA {marker_a} vs {marker_b}")
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded, cmap="tab10")
    plt.colorbar()
    plt.savefig(f"figure/pca_{marker_a}_{marker_b}.png")
    plt.close()

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    plt.figure()
    plt.title(f"t-SNE {marker_a} vs {marker_b}")
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_encoded, cmap="tab10")
    plt.colorbar()
    plt.savefig(f"figure/tsne_{marker_a}_{marker_b}.png")
    plt.close()

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":

    acc_matrix = pd.DataFrame(index=MARKERS, columns=MARKERS, dtype=float)
    sil_matrix = pd.DataFrame(index=MARKERS, columns=MARKERS, dtype=float)

    for m1, m2 in combinations(MARKERS, 2):
        print(f"\n=== {m1} vs {m2} ===")

        path_A = get_feature_path(m1)
        path_B = get_feature_path(m2)

        if not os.path.exists(path_A) or not os.path.exists(path_B):
            print("Missing file, skipping...")
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

        # décommente si tu veux les figures
        visualize(X, y, m1, m2)

    # diagonale
    np.fill_diagonal(acc_matrix.values, 1.0)
    np.fill_diagonal(sil_matrix.values, 0.0)

    print("\n=== Accuracy Matrix ===")
    print(acc_matrix)

    print("\n=== Silhouette Matrix ===")
    print(sil_matrix)

    acc_matrix.to_csv("accuracy_matrix.csv")
    sil_matrix.to_csv("silhouette_matrix.csv")