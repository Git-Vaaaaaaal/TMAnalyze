import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    integrated_brier_score,
    cumulative_dynamic_auc,
)
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv

# ── Constants (Ishwaran 2011 calibration, n=200, p=2560) ─────────────────────
N_PATIENTS        = 200
P_FEATURES        = 2560
N_TREES           = 1000
MAX_FEATURES      = round(P_FEATURES ** 0.75)  # ≈ 362 = p^(3/4)
MIN_SAMPLES_LEAF  = 3                           # nodesize≈1 en R (proxy Python)
MIN_SAMPLES_SPLIT = 6
SPLITTER          = "random"                    # nsplit=10 log-rank aléatoire (non exposé dans sksurv)
MAX_DEPTH         = None                        # arbres non élagués
LAMBDA_P          = round(P_FEATURES ** 0.8)   # ≈ 481 pour pondération MD-W
N_COMPONENTS_PCA  = 80
RANDOM_STATE      = 42
# ─────────────────────────────────────────────────────────────────────────────


def compute_cox_weights(
    X_train: np.ndarray,
    y_train: np.ndarray,
    p: int,
    lambda_p: float,
) -> np.ndarray:
    """Compute MD-W Cox univariate p-value weights (Ishwaran 2011) for p features."""
    print(f"[RSF] Calcul des poids Cox univariés ({p} features)...")
    cox_pvals = np.ones(p)
    for j in range(p):
        try:
            col = X_train[:, j]
            if col.std() < 1e-10:
                raise ValueError("constant")
            df_cox = pd.DataFrame({
                "feat":  col,
                "time":  y_train["time"],
                "event": y_train["event"].astype(int),
            })
            cph = CoxPHFitter()
            cph.fit(df_cox, duration_col="time", event_col="event", show_progress=False)
            cox_pvals[j] = float(cph.summary["p"].values[0])
        except Exception:
            cox_pvals[j] = 1.0
        if (j + 1) % 512 == 0:
            print(f"[RSF]   {j + 1}/{p} features traitées")
    cox_weights = 1.0 / np.maximum(cox_pvals, 1.0 / lambda_p)
    cox_weights /= cox_weights.sum()
    return cox_weights


def build_preprocessor(n_components: int) -> Pipeline:
    """Build an unfitted Pipeline: SimpleImputer → StandardScaler → PCA."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
        ("pca",     PCA(n_components=n_components, random_state=RANDOM_STATE)),
    ])


def build_rsf(n_pca_components: int) -> RandomSurvivalForest:
    """Instantiate RandomSurvivalForest with Ishwaran-calibrated hyperparameters."""
    return RandomSurvivalForest(
        n_estimators      = N_TREES,
        max_features      = min(MAX_FEATURES, n_pca_components),
        min_samples_leaf  = MIN_SAMPLES_LEAF,
        min_samples_split = MIN_SAMPLES_SPLIT,
        max_depth         = MAX_DEPTH,
        random_state      = RANDOM_STATE,
        n_jobs            = -1,
    )


def evaluate(
    rsf: RandomSurvivalForest,
    preprocessor: Pipeline,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Compute C-index train/test, IBS and mean dynamic AUC."""
    X_train_pca = preprocessor.transform(X_train)
    X_test_pca  = preprocessor.transform(X_test)

    c_train = float(rsf.score(X_train_pca, y_train))
    c_test  = float(rsf.score(X_test_pca,  y_test))

    ibs      = float("nan")
    mean_auc = float("nan")

    # Eval times : percentiles des événements training, bornés aux deux distributions
    event_times = y_train["time"][y_train["event"].astype(bool)]
    eval_times  = np.array([])
    if len(event_times) >= 2:
        t_max = min(y_train["time"].max(), y_test["time"].max())
        raw   = np.percentile(event_times, np.linspace(10, 90, 20))
        eval_times = np.unique(raw[
            (raw > y_train["time"].min()) & (raw < t_max)
        ])

    if len(eval_times) >= 2:
        try:
            surv_fns   = rsf.predict_survival_function(X_test_pca)
            surv_array = np.row_stack([fn(eval_times) for fn in surv_fns])
            ibs = float(integrated_brier_score(y_train, y_test, surv_array, eval_times))
        except Exception as exc:
            print(f"[RSF] IBS non calculable : {exc}")

        try:
            risk_test       = rsf.predict(X_test_pca)
            _, mean_auc_val = cumulative_dynamic_auc(y_train, y_test, risk_test, eval_times)
            mean_auc        = float(mean_auc_val)
        except Exception as exc:
            print(f"[RSF] AUC dynamique non calculable : {exc}")

    return {"c_train": c_train, "c_test": c_test, "ibs": ibs, "mean_auc": mean_auc}


def plot_results(
    rsf: RandomSurvivalForest,
    preprocessor: Pipeline,
    X_test_pca: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    risk_test: np.ndarray,
) -> None:
    """Generate and save KM-by-tertile, PCA variance and feature-importance plots."""

    # 1. Courbes de survie par tertile de risque ─────────────────────────────
    tertiles = np.percentile(risk_test, [33.3, 66.7])
    groups   = np.digitize(risk_test, tertiles)
    labels   = ["Faible", "Intermédiaire", "Élevé"]
    colors   = ["#2ca02c", "#ff7f0e", "#d62728"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for g, (label, color) in enumerate(zip(labels, colors)):
        mask = groups == g
        if mask.sum() == 0:
            continue
        y_grp = y_test[mask]
        times, surv = kaplan_meier_estimator(y_grp["event"].astype(bool), y_grp["time"])
        ax.step(times, surv, where="post", label=f"{label} (N={mask.sum()})",
                color=color, linewidth=2)
    ax.set_title("Survie Kaplan-Meier par tertile de risque RSF")
    ax.set_xlabel("Temps")
    ax.set_ylabel("Probabilité de survie")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.savefig("survival_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Variance cumulée expliquée par PCA ───────────────────────────────────
    pca    = preprocessor.named_steps["pca"]
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(cumvar) + 1), cumvar, color="#1f77b4", linewidth=2)
    ax.axhline(95, color="red", linestyle="--", linewidth=1.2, label="95 % variance")
    idx_95 = int(np.searchsorted(cumvar, 95))
    if idx_95 < len(cumvar):
        ax.axvline(idx_95 + 1, color="red", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_title("Variance cumulée expliquée par les composantes PCA")
    ax.set_xlabel("Nombre de composantes")
    ax.set_ylabel("Variance expliquée cumulée (%)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.savefig("pca_variance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Top-20 composantes PCA — feature_importances_ RSF ───────────────────
    importances = rsf.feature_importances_
    top_n   = min(20, len(importances))
    top_idx = np.argsort(importances)[-top_n:]
    top_imp = importances[top_idx]
    top_lbl = [f"PC{i + 1}" for i in top_idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_lbl, top_imp, color="#1f77b4")
    ax.set_title(f"Top {top_n} composantes PCA — importances RSF")
    ax.set_xlabel("Importance (MDI)")
    ax.grid(True, alpha=0.3, axis="x")
    fig.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("[RSF] Plots sauvegardés : pca_variance.png, survival_curves.png, feature_importance.png")


def run_pipeline(
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
) -> tuple:
    """Orchestrate split → Cox weights → PCA preprocessing → RSF → evaluation → plots."""

    # 1. Structured survival target
    y = Surv.from_arrays(event=event.astype(bool), time=time.astype(float))

    # 2. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"[RSF] Split — train: {len(y_train)}, test: {len(y_test)}")
    print(f"[RSF] Événements — train: {y_train['event'].sum()}, test: {y_test['event'].sum()}")

    # 3. Cox MD-W univariate weights (sur données train uniquement)
    p           = X_train.shape[1]
    cox_weights = compute_cox_weights(X_train, y_train, p, LAMBDA_P)

    # 4. Pondération des features
    X_train_w = X_train * cox_weights
    X_test_w  = X_test  * cox_weights

    # 5. Prétraitement : Imputer → Scaler → PCA
    preprocessor = build_preprocessor(N_COMPONENTS_PCA)
    X_train_pca  = preprocessor.fit_transform(X_train_w)
    X_test_pca   = preprocessor.transform(X_test_w)
    n_comp       = preprocessor.named_steps["pca"].n_components_
    var_exp      = preprocessor.named_steps["pca"].explained_variance_ratio_.sum() * 100
    print(f"[RSF] PCA : {p} → {n_comp} composantes ({var_exp:.1f} % variance)")

    # 6. Entraînement RSF
    rsf = build_rsf(n_comp)
    print(f"[RSF] Entraînement RSF — {N_TREES} arbres, max_features={rsf.max_features}...")
    rsf.fit(X_train_pca, y_train)

    # 7. Évaluation (evaluate transforme X_train_w / X_test_w en interne)
    metrics = evaluate(rsf, preprocessor, X_train_w, X_test_w, y_train, y_test)

    # 8. Scores de risque pour les plots
    risk_test = rsf.predict(X_test_pca)

    # 9. Graphiques
    plot_results(rsf, preprocessor, X_test_pca, y_train, y_test, risk_test)

    # 10. Récapitulatif
    def _fmt(v: float) -> str:
        return f"{v:.4f}" if not np.isnan(v) else "N/A"

    print(f"[RSF] C-index train : {_fmt(metrics['c_train'])}")
    print(f"[RSF] C-index test  : {_fmt(metrics['c_test'])}")
    print(f"[RSF] IBS           : {_fmt(metrics['ibs'])}")
    print(f"[RSF] AUC dynamique : {_fmt(metrics['mean_auc'])}")

    return rsf, preprocessor, metrics


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng   = np.random.default_rng(RANDOM_STATE)
    X     = rng.standard_normal((N_PATIENTS, P_FEATURES))
    time  = rng.exponential(scale=2.0, size=N_PATIENTS)
    event = (rng.uniform(size=N_PATIENTS) > 0.3).astype(int)  # ~70 % événements

    rsf, preprocessor, metrics = run_pipeline(X, time, event)
