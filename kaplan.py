import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival


# ── Configuration ──────────────────────────────────────────────────────────
CSV_PATH      = r"csv/clinical_data_cleaned.csv"
OS_TIME_COL   = "OS"
OS_EVENT_COL  = "Follow-up Status"
PFS_TIME_COL  = "PFS"
PFS_EVENT_COL = "Follow-up Status"   # adapter si colonne différente pour PFS
GROUP_COL     = None        # colonne de stratification (None = courbe globale)
INVERT_EVENT  = False
OUTPUT_DIR    = "out_kaplan"
CAP_YEARS     = 5.0         # patients OS > 5 ans → censurés à 5 ans
# ───────────────────────────────────────────────────────────────────────────

COLORS = {"OS": "steelblue", "PFS": "tomato"}


def apply_cap(df, time_col, event_col, cap=5.0):
    df = df.copy()
    mask = df[time_col] > cap
    df.loc[mask, event_col] = 0
    df.loc[mask, time_col]  = cap
    return df


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
    df[time_col] = df[time_col].astype(float)
    df = df[df[time_col] >= 0]
    return df


def plot_combined_km(df_os, df_pfs, os_time, os_event, pfs_time, pfs_event,
                     population_label="Global", output_path="km.png"):
    fig, ax = plt.subplots(figsize=(9, 6))

    for endpoint, df, time_col, event_col in [
        ("OS",  df_os,  os_time,  os_event),
        ("PFS", df_pfs, pfs_time, pfs_event),
    ]:
        color  = COLORS[endpoint]
        n      = len(df)
        events = int(df[event_col].sum())

        y = np.array(
            list(zip(df[event_col].astype(bool), df[time_col])),
            dtype=[("event", bool), ("time", float)],
        )
        times, surv = kaplan_meier_estimator(y["event"], y["time"])

        ax.step(times, surv, where="post", color=color, linewidth=2,
                label=f"{endpoint}  (N={n}, events={events})")

        censored = df[df[event_col] == 0][time_col].values
        for ct in censored:
            idx = max(np.searchsorted(times, ct, side="right") - 1, 0)
            ax.plot(ct, surv[idx], "|", color=color, markersize=9, markeredgewidth=1.5)

    ax.set_title(f"Kaplan-Meier OS + PFS — {population_label}", fontsize=13)
    ax.set_xlabel("Temps (années)", fontsize=11)
    ax.set_ylabel("Probabilité de survie", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Courbe sauvegardée → {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Chargement + cap 5 ans pour l'OS
df_os  = load_survival_data(CSV_PATH, OS_TIME_COL, OS_EVENT_COL, GROUP_COL, INVERT_EVENT)
df_os  = apply_cap(df_os, OS_TIME_COL, OS_EVENT_COL, cap=CAP_YEARS)

# Chargement PFS (pas de cap)
df_pfs = load_survival_data(CSV_PATH, PFS_TIME_COL, PFS_EVENT_COL, GROUP_COL, INVERT_EVENT)

print(f"OS  — N={len(df_os)}  | événements : {df_os[OS_EVENT_COL].sum()}")
print(f"PFS — N={len(df_pfs)} | événements : {df_pfs[PFS_EVENT_COL].sum()}")

# Graphe global
plot_combined_km(
    df_os, df_pfs, OS_TIME_COL, OS_EVENT_COL, PFS_TIME_COL, PFS_EVENT_COL,
    population_label="Global",
    output_path=os.path.join(OUTPUT_DIR, "km_combined_global.png"),
)

# Graphes par groupe si GROUP_COL défini
if GROUP_COL:
    for group_val in sorted(df_os[GROUP_COL].dropna().unique()):
        sub_os  = df_os[df_os[GROUP_COL] == group_val]
        sub_pfs = df_pfs[df_pfs[GROUP_COL] == group_val]
        plot_combined_km(
            sub_os, sub_pfs, OS_TIME_COL, OS_EVENT_COL, PFS_TIME_COL, PFS_EVENT_COL,
            population_label=str(group_val),
            output_path=os.path.join(OUTPUT_DIR, f"km_combined_{GROUP_COL}_{group_val}.png"),
        )
