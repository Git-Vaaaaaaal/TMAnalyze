import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.compare import compare_survival
from sksurv.nonparametric import kaplan_meier_estimator
from scipy.stats import chi2 as chi2_dist


def cox_wald(x, y_surv, coef):
    """Wald test et IC 95 % pour un unique covariable Cox.

    Calcule l'information de Fisher observée à β̂ pour obtenir SE(β̂),
    puis : z = β̂ / SE, p = chi2(z², df=1), IC = exp(β̂ ± 1.96·SE).
    """
    times, events = y_surv["time"], y_surv["event"]
    info = 0.0
    for t in np.unique(times[events]):
        mask = times >= t
        xr   = x[mask].astype(float)
        w    = np.exp(coef * xr)
        w   /= w.sum()
        info += (w * xr ** 2).sum() - (w * xr).sum() ** 2
    se    = 1.0 / np.sqrt(max(info, 1e-12))
    pval  = chi2_dist.sf((coef / se) ** 2, df=1)
    ci_lo = np.exp(coef - 1.96 * se)
    ci_hi = np.exp(coef + 1.96 * se)
    return float(pval), float(ci_lo), float(ci_hi)

CLINICAL_CSV = "csv/multi_label_patient_id.csv"
ENDPOINTS    = [("OS", "status"), ("PFS", "status")]
OUTPUT_DIR   = "out_kmeans_cox"

marker_list  = ["BCL2", "BCL6", "CD10", "HE", "MUM1", "MYC"]
encoder_list = ["prism", "feather"]
ENCODER_CFG  = {
    "prism":   dict(slide_subdir="slide_features_prism",   slide_csv="prism_encoder.csv"),
    "feather": dict(slide_subdir="slide_features_feather", slide_csv="feather_encoder.csv"),
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

df_clinical = pd.read_csv(CLINICAL_CSV)
df_clinical["patient_id"] = df_clinical["patient_id"].astype(str).apply(
    lambda x: os.path.splitext(x)[0]
)
time_cols  = [t for t, _ in ENDPOINTS]
event_cols = list({e for _, e in ENDPOINTS})
df_clinical = df_clinical[["patient_id", "stain"] + time_cols + event_cols].copy()
for col in time_cols:
    df_clinical[col] = pd.to_numeric(df_clinical[col], errors="coerce")
for col in event_cols:
    df_clinical[col] = pd.to_numeric(df_clinical[col], errors="coerce").astype("Int64")

results = []

for encoder in encoder_list:
    enc = ENCODER_CFG[encoder]
    for marker in marker_list:
        candidate = os.path.join("data_224_reborn", encoder, marker, enc["slide_subdir"], enc["slide_csv"])
        if not os.path.isfile(candidate):
            print(f"[SKIP] {candidate} introuvable")
            continue

        df_feat = pd.read_csv(candidate)
        df_feat["wsi_name"] = df_feat["wsi_name"].astype(str).apply(
            lambda x: os.path.splitext(x)[0]
        )
        X = df_feat.drop(columns=["wsi_name"]).values

        # KMeans k=2 — une seule fois par encoder/marker
        km2 = KMeans(n_clusters=2, random_state=42, n_init=10)
        df_feat["cluster"] = km2.fit_predict(X)

        df_clin_marker = df_clinical[df_clinical["stain"] == marker].drop(columns=["stain"])
        df_base = (
            df_feat[["wsi_name", "cluster"]]
            .rename(columns={"wsi_name": "patient_id"})
            .merge(df_clin_marker, on="patient_id", how="inner")
        )

        for time_col, event_col in ENDPOINTS:
            df_ep = df_base.dropna(subset=[time_col, event_col]).copy()
            df_ep[time_col]  = df_ep[time_col].astype(float)
            df_ep[event_col] = df_ep[event_col].astype(int)
            df_ep["cluster"] = df_ep["cluster"].astype(int)

            # Cap à 5 ans
            mask = df_ep[time_col] > 5.0
            df_ep.loc[mask, event_col] = 0
            df_ep.loc[mask, time_col]  = 5.0

            n_total = len(df_ep)
            if n_total < 10 or df_ep["cluster"].nunique() < 2:
                print(f"[SKIP] {encoder}/{marker}/{time_col} — pas assez de patients ({n_total})")
                continue

            # ── Structured array sksurv ───────────────────────────────────
            y_surv = np.array(
                [(bool(e), float(t)) for e, t in zip(df_ep[event_col], df_ep[time_col])],
                dtype=[("event", bool), ("time", float)],
            )
            groups = df_ep["cluster"].values

            # ── Cox : HR + Wald test ──────────────────────────────────────
            cox = CoxPHSurvivalAnalysis(ties="efron")
            cox.fit(df_ep[["cluster"]].values, y_surv)
            coef = cox.coef_[0]
            hr   = float(np.exp(coef))
            p_cox, ci_lo, ci_hi = cox_wald(groups.astype(float), y_surv, coef)

            # ── Log-rank ──────────────────────────────────────────────────
            chisq, p_logrank = compare_survival(y_surv, groups)[:2]

            n0 = (groups == 0).sum()
            n1 = (groups == 1).sum()
            print(
                f"[{encoder}][{marker}][{time_col}]  N={n_total} (C0={n0}, C1={n1})\n"
                f"  Cox      : HR={hr:.3f} (95% CI {ci_lo:.3f}–{ci_hi:.3f})  p={p_cox:.4f}\n"
                f"  Log-rank : chi2={chisq:.3f}  p={p_logrank:.4f}"
            )
            results.append(dict(
                encoder=encoder, marker=marker, endpoint=time_col,
                n=n_total, n0=n0, n1=n1,
                HR=round(hr, 4), CI_lo=round(ci_lo, 4), CI_hi=round(ci_hi, 4),
                p_cox=round(p_cox, 4), chi2_logrank=round(chisq, 4), p_logrank=round(p_logrank, 4),
            ))

            # ── Kaplan-Meier ──────────────────────────────────────────────
            COLORS_KM = {0: "steelblue", 1: "tomato"}
            fig, ax = plt.subplots(figsize=(8, 6))

            for cl in [0, 1]:
                sub = df_ep[df_ep["cluster"] == cl]
                y_cl = np.array(
                    [(bool(e), float(t)) for e, t in zip(sub[event_col], sub[time_col])],
                    dtype=[("event", bool), ("time", float)],
                )
                times, surv = kaplan_meier_estimator(y_cl["event"], y_cl["time"])
                ax.step(times, surv, where="post", color=COLORS_KM[cl], linewidth=2,
                        label=f"Cluster {cl}  (N={len(sub)})")
                censored = sub[sub[event_col] == 0][time_col].values
                for ct in censored:
                    idx = max(np.searchsorted(times, ct, side="right") - 1, 0)
                    ax.plot(ct, surv[idx], "|", color=COLORS_KM[cl],
                            markersize=9, markeredgewidth=1.5)

            ax.set_title(
                f"{encoder} | {marker} | {time_col}  (N={n_total})\n"
                f"Cox: HR={hr:.3f} (95% CI {ci_lo:.3f}–{ci_hi:.3f})  p={p_cox:.4f}\n"
                f"Log-rank: p={p_logrank:.4f}",
                fontsize=10,
            )
            ax.set_xlabel(f"{time_col} (years)", fontsize=11)
            ax.set_ylabel("Survival probability", fontsize=11)
            ax.set_ylim(0, 1)
            ax.legend(fontsize=10, loc="upper right")
            ax.grid(alpha=0.3)

            plt.tight_layout()
            outpath = os.path.join(OUTPUT_DIR, f"{encoder}_{marker}_{time_col}_kmeans_cox.png")
            fig.savefig(outpath, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  → {outpath}")

# ── Tableau récap ──────────────────────────────────────────────────────────
if results:
    df_results = pd.DataFrame(results)
    out_csv = os.path.join(OUTPUT_DIR, "kmeans_cox_summary.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"\nRécap → {out_csv}")
    print(df_results.to_string(index=False))
