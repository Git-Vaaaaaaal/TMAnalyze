import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.utils import resample


marker_list  = ["BCL2", "BCL6", "CD10", "HE", "MUM1", "MYC"]
encoder_list = ["prism", "feather"]
algo_list    = ["knn", "svm", "random_forest"]

ENCODER_CFG = {
    "prism":   dict(slide_subdir="slide_features_prism",   slide_csv="prism_encoder.csv"),
    "titan":   dict(slide_subdir="slide_features_titan",   slide_csv="titan_encoder.csv"),
    "feather": dict(slide_subdir="slide_features_feather", slide_csv="feather_encoder.csv"),
}

dataframe_id = os.path.join("csv", "multi_label_patient_id.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def binarize_column(df, column: str, group_0: list, group_1: list) -> pd.DataFrame:
    """
    Regroupe les valeurs d'une colonne en 0 / 1.

    Args:
        df         : DataFrame contenant les données
        column     : colonne à transformer
        group_0    : liste des valeurs à mapper → 0
        group_1    : liste des valeurs à mapper → 1
    """

    if column not in df.columns:
        print(f"Colonne '{column}' introuvable. Colonnes disponibles :")
        print("  " + "\n  ".join(df.columns.tolist()))
        return df

    mapping = {str(v): 0 for v in group_0}
    mapping.update({str(v): 1 for v in group_1})

    src = df[column].astype(str)
    unknown = set(src.dropna().unique()) - set(mapping.keys()) - {"nan"}
    if unknown:
        print(f"[ATTENTION] Valeurs non mappées (seront NaN) : {unknown}")

    result = src.map(mapping)

    df[column] = result.astype("Int64")

    n_0       = (df[column] == 0).sum()
    n_1       = (df[column] == 1).sum()
    n_missing = df[column].isna().sum()

    print(f"\n{'─' * 45}")
    print(f"{'─' * 45}")
    print(f"  Groupe 0  ({group_0}) : {n_0} patients")
    print(f"  Groupe 1  ({group_1}) : {n_1} patients")
    print(f"  Non mappés / NaN      : {n_missing}")
    print(f"{'─' * 45}\n")
    return df


def cleaning_csv(df_path, marker, element="RIPI Risk Group"):
    df = pd.read_csv(df_path)
    df = df[df["stain"] == marker]
    df = df[["patient_id", element]].rename(columns={element: "Status"})
    df = binarize_column(df, "Status", group_0=[0,1], group_1=[2])
    df = df.dropna(subset=["Status"])
    df["Status"] = (df["Status"] > 0).astype(int)
    return df


def load_dataset(features_csv, labels_df):
    """Merge features + labels on patient_id, return X, y, ids."""
    df_feat = pd.read_csv(features_csv).rename(columns={"wsi_name": "patient_id"})
    df = df_feat.merge(labels_df, on="patient_id", how="inner")
    feat_cols = [c for c in df.columns if c.startswith("slide_feat_")]
    X = df[feat_cols].values.astype(np.float32)
    y = df["Status"].values.astype(int)
    return X, y, df["patient_id"].values


def balance_training_data(X_train, y_train, seed=42):
    """Oversample minority class so every class has the same count."""
    classes, counts = np.unique(y_train, return_counts=True)
    max_count = counts.max()
    X_parts, y_parts = [], []
    for cls in classes:
        idx = np.where(y_train == cls)[0]
        Xc, yc = X_train[idx], y_train[idx]
        if len(Xc) < max_count:
            Xc, yc = resample(Xc, yc, n_samples=max_count, random_state=seed, replace=True)
        X_parts.append(Xc)
        y_parts.append(yc)
    X_bal = np.vstack(X_parts)
    y_bal = np.concatenate(y_parts)
    rng  = np.random.default_rng(seed)
    perm = rng.permutation(len(y_bal))
    return X_bal[perm], y_bal[perm]


def build_model(algo):
    if algo == "knn":
        return KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    elif algo == "svm":
        return SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
    elif algo == "random_forest":
        return RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)


def save_confusion_matrix(y_true, y_pred, encoder, marker, algo, save_path):
    cm   = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{encoder}  |  {marker}  |  {algo}", fontsize=11, fontweight="bold")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix → {save_path}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

os.makedirs("output_classical", exist_ok=True)
log_path = "output_logs_classical.txt"

for marker in marker_list:
    for encoder in encoder_list:
        enc          = ENCODER_CFG[encoder]
        features_csv = os.path.join("export", "export", encoder, marker, enc["slide_subdir"], enc["slide_csv"])

        if not os.path.exists(features_csv):
            print(f"[SKIP] Fichier manquant : {features_csv}")
            continue

        labels_df = cleaning_csv(dataframe_id, marker)

        try:
            X, y, patient_ids = load_dataset(features_csv, labels_df)
        except Exception as e:
            print(f"[SKIP] {marker}/{encoder} — erreur chargement : {e}")
            continue

        if len(np.unique(y)) < 2:
            print(f"[SKIP] {marker}/{encoder} — une seule classe présente")
            continue

        print(f"\n{'='*62}")
        print(f"  Encoder : {encoder}   Marker : {marker}")
        print(f"  Patients total : {len(y)}   cls0={( y==0).sum()}   cls1={(y==1).sum()}")
        print(f"{'='*62}")

        # Stratified split 80/20
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(sss.split(X, y))
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        ids_val        = patient_ids[val_idx]

        # Balance training set (oversample minority)
        X_train_bal, y_train_bal = balance_training_data(X_train, y_train)
        print(f"  Train avant balance : cls0={(y_train==0).sum()}  cls1={(y_train==1).sum()}")
        print(f"  Train après balance : cls0={(y_train_bal==0).sum()}  cls1={(y_train_bal==1).sum()}")
        print(f"  Val                 : cls0={(y_val==0).sum()}  cls1={(y_val==1).sum()}")

        # Normalisation (fit sur train uniquement)
        scaler       = StandardScaler()
        X_train_sc   = scaler.fit_transform(X_train_bal)
        X_val_sc     = scaler.transform(X_val)

        for algo in algo_list:
            print(f"\n  --- {algo.upper()} ---")
            model = build_model(algo)
            model.fit(X_train_sc, y_train_bal)

            y_pred = model.predict(X_val_sc)
            y_prob = model.predict_proba(X_val_sc)[:, 1]

            try:
                auc = roc_auc_score(y_val, y_prob)
                ap  = average_precision_score(y_val, y_prob)
            except ValueError:
                auc = ap = float("nan")
            acc = (y_pred == y_val).mean()

            # Prediction log
            print(f"  acc={acc:.3f}  AUC={auc:.3f}  AP={ap:.3f}")
            print(f"  {'Patient':<15} {'Vrai':>5} {'Pred':>5} {'Prob':>8}")
            print(f"  {'-'*38}")
            for pid, true, pred, prob in zip(ids_val, y_val, y_pred, y_prob):
                flag = "  ← erreur" if true != pred else ""
                print(f"  {str(pid):<15} {true:>5} {pred:>5} {prob:>8.3f}{flag}")

            print()
            print(classification_report(y_val, y_pred, target_names=["cls0", "cls1"], zero_division=0))

            # Confusion matrix
            save_path = os.path.join("output_classical", f"{marker}_{encoder}_{algo}.png")
            save_confusion_matrix(y_val, y_pred, encoder, marker, algo, save_path)

            # Global log file
            with open(log_path, "a") as f:
                f.write(f"{marker},{encoder},{algo},acc={acc:.4f},auc={auc:.4f},ap={ap:.4f}\n")
