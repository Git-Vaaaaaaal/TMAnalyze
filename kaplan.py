import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival
from scipy.stats import chi2


# ── Configuration ──────────────────────────────────────────────────────────
CSV_PATH      = r"csv/clinical_data_cleaned.csv"
TIME_COL      = "OS"           # colonne durée (float, en années)
EVENT_COL     = "Follow-up Status"       # colonne événement (0=censuré, 1=événement)
GROUP_COL     = None        # colonne de stratification (None = courbe globale)
INVERT_EVENT  = False        # True = inverser 0↔1 dans EVENT_COL
OUTPUT_DIR    = "out_kaplan"
# ───────────────────────────────────────────────────────────────────────────


def load_survival_data(csv_path, time_col, event_col, group_col=None, invert_event=False):
    df = pd.read_csv(csv_path)
    required = [time_col, event_col] + ([group_col] if group_col else [])
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Colonne '{col}' introuvable. Disponibles : {df.columns.tolist()}")

    df = df[required].dropna()
    df[event_col] = df[event_col].astype(int)
    if invert_event:
        df[event_col] = 1 - df[event_col]
    df[time_col]  = df[time_col].astype(float)
    df = df[df[time_col] >= 0]
    return df


def plot_kaplan_meier(df, time_col, event_col, group_col=None, output_path="km.png"):
    fig, ax = plt.subplots(figsize=(9, 6))

    PALETTE = [
        "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
        "#9467bd", "#8c564b", "#e377c2", "#17becf",
    ]

    if group_col is None:
        # Courbe globale unique
        groups = {"Tous les patients": df}
    else:
        groups = {name: grp for name, grp in df.groupby(group_col)}

    pval_str = ""
    if group_col and len(groups) >= 2:
        y_all = np.array(
            list(zip(df[event_col].astype(bool), df[time_col])),
            dtype=[("event", bool), ("time", float)],
        )
        group_labels = df[group_col].values
        try:
            chi2_stat, pval = compare_survival(y_all, group_labels)[:2]
            pval_str = f"\nLog-rank p = {pval:.4f}"
        except Exception:
            pval_str = ""

    for i, (label, grp) in enumerate(groups.items()):
        color = PALETTE[i % len(PALETTE)]
        n     = len(grp)
        events = grp[event_col].sum()

        y = np.array(
            list(zip(grp[event_col].astype(bool), grp[time_col])),
            dtype=[("event", bool), ("time", float)],
        )
        times, surv = kaplan_meier_estimator(y["event"], y["time"])

        ax.step(times, surv, where="post", color=color, linewidth=2,
                label=f"{label}  (N={n}, events={events})")

        # marques de censure
        censored = grp[grp[event_col] == 0][time_col].values
        for ct in censored:
            idx = max(np.searchsorted(times, ct, side="right") - 1, 0)
            ax.plot(ct, surv[idx], "|", color=color, markersize=9, markeredgewidth=1.5)

    title = f"Courbe de Kaplan-Meier — {time_col}{pval_str}"
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Temps (années)", fontsize=11)
    ax.set_ylabel("Probabilité de survie", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Courbe sauvegardée → {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = load_survival_data(CSV_PATH, TIME_COL, EVENT_COL, GROUP_COL, invert_event=INVERT_EVENT)
print(f"N patients : {len(df)}  |  événements : {df[EVENT_COL].sum()}")

# Courbe globale (sans stratification)
plot_kaplan_meier(
    df, TIME_COL, EVENT_COL,
    group_col=None,
    output_path=os.path.join(OUTPUT_DIR, f"km_{TIME_COL}_global.png"),
)

# Courbe stratifiée par GROUP_COL
if GROUP_COL:
    plot_kaplan_meier(
        df, TIME_COL, EVENT_COL,
        group_col=GROUP_COL,
        output_path=os.path.join(OUTPUT_DIR, f"km_{TIME_COL}_by_{GROUP_COL}.png"),
    )
