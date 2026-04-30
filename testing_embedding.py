import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# ---------------------------
# CONFIG
# ---------------------------
FEATURES_CSV = "features.csv"
LABELS_CSV = "labels.csv"  # mettre None si déjà dans features

FILENAME_COL = "filename"
LABEL_COL = "label"

# ---------------------------
# LOAD DATA
# ---------------------------
def load_data(features_csv, labels_csv=None):
    df = pd.read_csv(features_csv)

    if labels_csv is not None:
        labels_df = pd.read_csv(labels_csv)

        # merge sur filename
        df = df.merge(labels_df, on=FILENAME_COL)

    # extraction
    y = df[LABEL_COL].values
    X = df.drop(columns=[FILENAME_COL, LABEL_COL]).values

    return X, y

# ---------------------------
# CLASSIFICATION TEST
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
    acc = accuracy_score(y_test, y_pred)

    print(f"[Classification] Accuracy: {acc:.4f}")

# ---------------------------
# CLUSTERING QUALITY
# ---------------------------
def clustering_test(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    score = silhouette_score(X_scaled, y)
    print(f"[Clustering] Silhouette Score: {score:.4f}")

# ---------------------------
# VISUALIZATION
# ---------------------------
def visualize(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure()
    plt.title("PCA")
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="tab10")
    plt.colorbar()
    plt.show()

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    plt.figure()
    plt.title("t-SNE")
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="tab10")
    plt.colorbar()
    plt.show()

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    X, y = load_data(FEATURES_CSV, LABELS_CSV)

    print(f"Dataset shape: {X.shape}")

    classification_test(X, y)
    clustering_test(X, y)
    visualize(X, y)
