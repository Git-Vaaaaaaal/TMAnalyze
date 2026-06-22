import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
# Main loop
# ---------------------------------------------------------------------------

os.makedirs("output_classical", exist_ok=True)
os.makedirs("out_rfs", exist_ok=True)
log_path = "logs_rf_survival.txt"
results  = []

parameters_df = pd.read_csv("out_rfs/results.csv")

COLORS = {"OS": "steelblue", "PFS": "tomato"}

for marker in marker_list:
    for encoder in encoder_list:
        enc          = ENCODER_CFG[encoder]
        features_csv = os.path.join("data_224", encoder, marker, enc["slide_subdir"], enc["slide_csv"])

        df_marker = parameters_df[(parameters_df["marker"] == marker) & (parameters_df["encoder"] == encoder)]

        parameters = {
            'n_estimators': df_marker["best_n_estimators"].iloc[0],
            'max_depth': int(df_marker["best_max_depth"].iloc[0]) if not pd.isna(df_marker["best_max_depth"].iloc[0]) else None,
            'min_samples_split': df_marker["best_min_samples_split"].iloc[0],
            'min_samples_leaf': df_marker["best_min_samples_leaf"].iloc[0],
            'max_features': df_marker["best_max_features"].iloc[0] if not pd.isna(df_marker["best_max_features"].iloc[0]) else None,
        }

        km_data = {}  # stocke les courbes KM pour OS et PFS

        for element_time in ["OS", "PFS"]:
            os_bool  = (element_time == "OS")
            labels_df = cleaning_csv(dataframe_id, marker, element_time, element_event, os_bool)

            try:
                X, y, patient_ids = load_dataset(features_csv, labels_df, element_time, element_event)
            except Exception as e:
                print(f"[SKIP] {marker}/{encoder}/{element_time} — erreur chargement : {e}")
                continue

            X_train, X_val, y_train, y_val, _, _ = train_test_split(
                X, y, patient_ids, test_size=0.2, random_state=42
            )

            print(f"  [{element_time}] Train : {len(y_train)} patients  |  Val : {len(y_val)} patients")

            scaler     = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_val_sc   = scaler.transform(X_val)

            pca        = PCA(n_components=0.95, random_state=42)
            X_train_sc = pca.fit_transform(X_train_sc)
            X_val_sc   = pca.transform(X_val_sc)
            n_comp     = pca.n_components_
            var_exp    = pca.explained_variance_ratio_.sum()
            print(f"  [{element_time}] PCA : {X_train.shape[1]} → {n_comp} composantes ({var_exp*100:.1f}% variance)")

            model = RandomSurvivalForest(n_estimators=parameters["n_estimators"], max_depth=parameters["max_depth"], min_samples_split=parameters["min_samples_split"], min_samples_leaf=parameters["min_samples_leaf"], max_features=parameters["max_features"], random_state=42, n_jobs=-1)
            pca.fit(X_train_sc, y_train)

            c_index = pca.score(X_val_sc, y_val)
            with open(log_path, "a") as f:
                f.write(f"{element_time}, {marker},{encoder},c_index={c_index:.4f}\n")

            results.append({
                "element_time": element_time, "marker": marker, "encoder": encoder,
                "c_index":      round(c_index, 4),
                "n_train":      len(y_train), "n_val": len(y_val),
                "events_train": int(y_train["event"].sum()),
                "events_val":   int(y_val["event"].sum()),
            })

            km_times, km_surv = kaplan_meier_estimator(y_val["event"], y_val["time"])
            censored_times    = y_val["time"][~y_val["event"].astype(bool)]
            km_data[element_time] = {
                "times": km_times, "surv": km_surv,
                "censored_times": censored_times, "c_index": c_index,
            }

        # --- Graphe combiné OS + PFS ---
        if km_data:
            fig, ax = plt.subplots(figsize=(8, 5))
            for et, data in km_data.items():
                color = COLORS[et]
                ax.step(data["times"], data["surv"], where="post",
                        label=f"KM {et}  (C={data['c_index']:.3f})", color=color)
                # Croix pour les patients censurés
                for ct in data["censored_times"]:
                    idx     = max(np.searchsorted(data["times"], ct, side="right") - 1, 0)
                    surv_ct = data["surv"][idx]
                    ax.plot(ct, surv_ct, "+", color=color, markersize=7, markeredgewidth=1.5)
            ax.set_ylabel("Probabilité de survie")
            ax.set_xlabel("Temps (années)")
            ax.set_title(f"{encoder} | {marker}")
            ax.legend()
            ax.grid(True)
            fig.savefig(f"out_rfs/{marker}_{encoder}_km.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

csv_path = "out_rfs_pca/results.csv"
pd.DataFrame(results).to_csv(csv_path, index=False)
print(f"\nRésultats sauvegardés → {csv_path}")

