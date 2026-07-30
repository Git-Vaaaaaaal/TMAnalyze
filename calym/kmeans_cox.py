import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

N_CLUSTERS = 2

encoder_list = ["prism", "feather"]
ENCODER_CFG = {
    "prism":   dict(slide_subdir="slide_features_prism",   slide_csv="prism_encoder.csv"),
    "feather": dict(slide_subdir="slide_features_feather", slide_csv="feather_encoder.csv"),
}

dataframe_id = os.path.join("csv_calym", "ia2hl_clinical.csv")
os.makedirs("output_calym", exist_ok=True)
os.makedirs("output_calym/kmeans", exist_ok=True)
cox_results = []

for encoder in encoder_list:
    enc = ENCODER_CFG[encoder]
    candidate = os.path.join("embedding", encoder, enc["slide_subdir"], enc["slide_csv"])
    df_feat = pd.read_csv(candidate).rename(columns={"wsi_name": "patient_id"})
    feat_cols = [c for c in df_feat.columns if c != "patient_id"]
    X = df_feat[feat_cols].values

    # --- KMeans k=2 fixe ---
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    sil_avg     = silhouette_score(X, labels)
    sil_samples = silhouette_samples(X, labels)
    print(f"[{encoder}]Silhouette (k=2) : {sil_avg:.4f}")

    df_clusters  = pd.DataFrame({"patient_id": df_feat["patient_id"], "cluster": labels})
    df_clinical  = pd.read_csv(dataframe_id)
    df_clin_mark = df_clinical

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{encoder} | —  KMeans k=2  (silhouette={sil_avg:.3f})")

    # --- Silhouette plot (indépendant de OS/PFS) ---
    ax_sil = axes[2]
    y_lower = 10
    for c, color in enumerate(["steelblue", "tomato"]):
        c_vals  = np.sort(sil_samples[labels == c])
        y_upper = y_lower + len(c_vals)
        ax_sil.barh(range(y_lower, y_upper), c_vals, height=1.0, color=color)
        ax_sil.text(-0.05, (y_lower + y_upper) / 2, str(c))
        y_lower = y_upper + 10
    ax_sil.axvline(x=sil_avg, color="black", linestyle="--", label=f"moy={sil_avg:.3f}")
    ax_sil.set_xlabel("Silhouette")
    ax_sil.set_title("Silhouette k=2")
    ax_sil.legend()

    # --- Cox + KM pour OS et PFS ---
    for ax_km, element_time in zip(axes[:2], ["OS", "PFS"]):
        df_clin_t = df_clin_mark[["patient_id", element_time, "status", "Age"]].dropna().copy()
        df_clin_t[element_time] = df_clin_t[element_time].astype(float)
        df_clin_t["status"]     = df_clin_t["status"].astype(int)
        mask = df_clin_t[element_time] > 5.0
        df_clin_t.loc[mask, "status"]     = 0
        df_clin_t.loc[mask, element_time] = 5.0

        df_merged = df_clusters.merge(df_clin_t, on="patient_id", how="inner")
        if len(df_merged) < 10:
            print(f"  [{element_time}] Pas assez de données ({len(df_merged)} patients), skip")
            ax_km.set_title(f"{element_time} — données insuffisantes")
            continue

        g0 = df_merged[df_merged["cluster"] == 0]
        g1 = df_merged[df_merged["cluster"] == 1]

        # Log-rank test
        lr = logrank_test(
            g0[element_time], g1[element_time],
            event_observed_A=g0["status"], event_observed_B=g1["status"],
        )

        # Cox PH (cluster binaire comme covariable)
        cph = CoxPHFitter()
        cph.fit(df_merged[["cluster", element_time, "status", "Age"]],
                duration_col=element_time, event_col="status")
        row  = cph.summary.loc["cluster"]
        hr   = row["exp(coef)"]
        ci_l = row["exp(coef) lower 95%"]
        ci_u = row["exp(coef) upper 95%"]
        p    = row["p"]

        print(f"  [{element_time}] n={len(df_merged)}  "
                f"log-rank p={lr.p_value:.4f}  "
                f"Cox HR={hr:.3f} [{ci_l:.3f}-{ci_u:.3f}] p={p:.4f}")

        cox_results.append({
            "encoder": encoder, "time": element_time,
            "n": len(df_merged),
            "logrank_p": round(lr.p_value, 4),
            "cox_HR":    round(hr,  3),
            "cox_CI_low":  round(ci_l, 3),
            "cox_CI_high": round(ci_u, 3),
            "cox_p": round(p, 4),
            "silhouette": round(sil_avg, 4),
        })

        # Courbes KM
        for c, color in enumerate(["steelblue", "tomato"]):
            g = df_merged[df_merged["cluster"] == c]
            kmf = KaplanMeierFitter()
            kmf.fit(g[element_time], event_observed=g["status"],
                    label=f"Cluster {c} (n={len(g)})")
            kmf.plot_survival_function(ax=ax_km, color=color)

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax_km.set_title(
            f"{element_time}  log-rank p={lr.p_value:.3f}\n"
            f"Cox HR={hr:.2f} [{ci_l:.2f}-{ci_u:.2f}] p={p:.3f} {sig}"
        )
        ax_km.set_xlabel("Temps (années)")
        ax_km.set_ylabel("Survie")
        ax_km.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join("kmeans", f"{encoder}_cox_age.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

df_cox = pd.DataFrame(cox_results)
df_cox.to_csv("output_calym/kmeans/cox_results_age.csv", index=False)
print("\nRésultats Cox → kmeans/cox_results_age.csv")
print(df_cox.to_string(index=False))
