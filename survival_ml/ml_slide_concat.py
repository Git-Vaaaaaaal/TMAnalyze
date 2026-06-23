import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import kaplan_meier_estimator
from sklearn.utils import resample


marker_list  = ["BCL2", "BCL6", "CD10", "HE", "MUM1", "MYC"]
encoder_list = ["prism", "feather"]

element_event = "status"

ENCODER_CFG = {
    "prism":   dict(slide_subdir="slide_features_prism",   slide_csv="prism_encoder.csv"),
    "titan":   dict(slide_subdir="slide_features_titan",   slide_csv="titan_encoder.csv"),
    "feather": dict(slide_subdir="slide_features_feather", slide_csv="feather_encoder.csv"),
}

dataframe_id = os.path.join("csv", "multi_label_patient_id.csv")



def cleaning_csv(df_path, marker, element_time, element_event, os_bool):
    df = pd.read_csv(df_path)
    df = df[df["stain"] == marker]
    df = df[["patient_id", element_time, element_event]]
    df = df.dropna(subset=[element_time, element_event])
    df[element_event] = df[element_event].astype(int)   # 1 = événement, 0 = censuré
    df[element_time]  = df[element_time].astype(float)
    if os_bool:
        mask = df[element_time] > 5.0
        df.loc[mask, element_event] = 0
        df.loc[mask, element_event] = 5.0
    return df


def load_dataset(features_csv, labels_df, element_time, element_event):
    """Merge features + labels on patient_id, return X, y (structured), ids."""
    df_feat = pd.read_csv(features_csv).rename(columns={"wsi_name": "patient_id"})
    df = df_feat.merge(labels_df, on="patient_id", how="inner")
    feat_cols = [c for c in df.columns if c.startswith("slide_feat_")]
    X = df[feat_cols].values.astype(np.float32)
    y = np.array(
        list(zip(df[element_event].astype(bool), df[element_time].astype(float))),
        dtype=[("event", bool), ("time", float)],
    )
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

def rsf_concordance_score(estimator, X, y):
    prediction = estimator.predict(X)
    result = concordance_index_censored(y['event'], y['time'], prediction)
    return result[0]  # retourne le C-index


# ---------------------------------------------------------------------------
# Variante : concaténation de tous les marqueurs
# ---------------------------------------------------------------------------

CONCAT_LABELS_CSV = "/home/imvia/Bureau/valentin/TMAnalyze/common_patients_labels.csv"

def load_concat_dataset(encoder, labels_csv, element_time, element_event, markers, os_bool):
    """Concatène les features slide de tous les marqueurs pour chaque patient commun."""
    df_labels = pd.read_csv(labels_csv)

    # Une ligne par patient (dédupliquée sur old_patient_id)
    labels_unique = (
        df_labels.drop_duplicates(subset=["old_patient_id"])
        [["old_patient_id", element_time, element_event]]
        .dropna(subset=[element_time, element_event])
        .copy()
    )
    labels_unique[element_event] = labels_unique[element_event].astype(int)
    labels_unique[element_time]  = labels_unique[element_time].astype(float)
    if os_bool:
        mask = labels_unique[element_time] > 5.0
        labels_unique.loc[mask, element_event] = 1
        labels_unique.loc[mask, element_time]  = 5.0

    enc = ENCODER_CFG[encoder]
    merged = None
    for marker in markers:
        feat_csv = os.path.join("data_224", encoder, marker, enc["slide_subdir"], enc["slide_csv"])
        if not os.path.exists(feat_csv):
            print(f"  [concat] features manquantes pour {marker}, ignoré")
            continue
        df_feat = pd.read_csv(feat_csv).rename(columns={"wsi_name": "patient_id"})
        # Lier patient_id (stain) → old_patient_id via common_patients_labels
        map_df = df_labels[df_labels["stain"] == marker][["patient_id", "old_patient_id"]].drop_duplicates()
        df_feat = df_feat.merge(map_df, on="patient_id", how="inner")
        feat_cols = [c for c in df_feat.columns if c.startswith("slide_feat_")]
        df_feat = df_feat[["old_patient_id"] + feat_cols].rename(
            columns={c: f"{marker}_{c}" for c in feat_cols}
        )
        merged = df_feat if merged is None else merged.merge(df_feat, on="old_patient_id", how="inner")

    if merged is None or merged.empty:
        raise ValueError("Aucune feature disponible pour la concaténation")

    merged = merged.merge(labels_unique, on="old_patient_id", how="inner")
    feat_cols = [c for c in merged.columns if "slide_feat_" in c]
    X = merged[feat_cols].values.astype(np.float32)
    y = np.array(
        list(zip(merged[element_event].astype(bool), merged[element_time].astype(float))),
        dtype=[("event", bool), ("time", float)],
    )
    return X, y, merged["old_patient_id"].values


results_concat = []

for encoder in encoder_list:
    km_data = {}

    for element_time in ["OS", "PFS"]:
        os_bool = (element_time == "OS")

        try:
            X, y, patient_ids = load_concat_dataset(
                encoder, CONCAT_LABELS_CSV, element_time, element_event, marker_list, os_bool
            )
        except Exception as e:
            print(f"[SKIP concat] {encoder}/{element_time} — {e}")
            continue

        X_train, X_val, y_train, y_val, _, _ = train_test_split(
            X, y, patient_ids, test_size=0.2, random_state=42
        )
        print(f"  [concat {encoder} {element_time}] Train={len(y_train)}  Val={len(y_val)}")

        scaler     = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc   = scaler.transform(X_val)

        model = RandomSurvivalForest(n_estimators=200, random_state=42, n_jobs=-1)
        clf   = GridSearchCV(model, parameters, cv=5, scoring=rsf_concordance_score)
        clf.fit(X_train_sc, y_train)

        c_index = clf.score(X_val_sc, y_val)
        print(f"  [concat {encoder} {element_time}] C-index : {c_index:.4f}")
        with open(log_path, "a") as f:
            f.write(f"CONCAT {element_time},{encoder},c_index={c_index:.4f}\n")

        results_concat.append({
            "element_time": element_time, "marker": "ALL", "encoder": encoder,
            "c_index": round(c_index, 4),
            "n_train": len(y_train), "n_val": len(y_val),
            "events_train": int(y_train["event"].sum()),
            "events_val":   int(y_val["event"].sum()),
            **{f"best_{k}": v for k, v in clf.best_params_.items()},
        })

        km_times, km_surv  = kaplan_meier_estimator(y_val["event"], y_val["time"])
        censored_times     = y_val["time"][~y_val["event"].astype(bool)]
        km_data[element_time] = {
            "times": km_times, "surv": km_surv,
            "censored_times": censored_times, "c_index": c_index,
        }

    if km_data:
        fig, ax = plt.subplots(figsize=(8, 5))
        for et, data in km_data.items():
            color = COLORS[et]
            ax.step(data["times"], data["surv"], where="post",
                    label=f"KM {et}  (C={data['c_index']:.3f})", color=color)
            for ct in data["censored_times"]:
                idx     = max(np.searchsorted(data["times"], ct, side="right") - 1, 0)
                surv_ct = data["surv"][idx]
                ax.plot(ct, surv_ct, "+", color=color, markersize=7, markeredgewidth=1.5)
        ax.set_ylabel("Probabilité de survie")
        ax.set_xlabel("Temps (années)")
        ax.set_title(f"{encoder} | ALL markers (concat)")
        ax.legend()
        ax.grid(True)
        fig.savefig(f"out_rfs/ALL_{encoder}_km.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

if results_concat:
    pd.DataFrame(results_concat).to_csv("out_rfs/results_concat.csv", index=False)
    print("\nRésultats concat sauvegardés → out_rfs/results_concat.csv")

