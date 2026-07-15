import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator


# ── Configuration ──────────────────────────────────────────────────────────
CSV_PATH    = r"csv/IA2HL.csv"
CSV_DLBCL   = r"csv/multiple_patient_id.csv"
EVENT_COL   = "status"
EVENT_COL_PFS = "PFS\ncensoring"
TIME_COL_PFS   = "PFS\n(years)"
EVENT_COL_OS  = "OS\ncensoring"
TIME_COL_OS = "Overall\nsurvival\n(years)"
OUTPUT_DIR  = "out_kaplan"
CAP_YEARS   = 5.0
# ───────────────────────────────────────────────────────────────────────────

COLORS = {"OS": "steelblue", "PFS": "tomato"}


def load_marker_data(df, time_col, event_col, cap=5.0):
    df = df.dropna(subset=[time_col, event_col])
    if df[event_col].dtype == object:
        df[event_col] = df[event_col].map({"Yes": 0, "No": 1})
    df[time_col]  = df[time_col].astype(str).str.replace(",", ".", regex=False)
    df[event_col] = df[event_col].astype(int)
    df[time_col]  = df[time_col].astype(float)
    df = df[df[time_col] >= 0]
    if cap is not None:
        mask = df[time_col] > cap
        df.loc[mask, event_col] = 0
        df.loc[mask, time_col]  = cap
    return df


def plot_single_km(df, time_col, event_col, endpoint, output_path):
    color  = COLORS[endpoint]
    n      = len(df)
    events = int(df[event_col].sum())

    y = np.array(
        list(zip(df[event_col].astype(bool), df[time_col])),
        dtype=[("event", bool), ("time", float)],
    )
    times, surv = kaplan_meier_estimator(y["event"], y["time"])

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.step(times, surv, where="post", color=color, linewidth=2,
            label=f"{endpoint}  (N={n}, events={events})")

    censored = df[df[event_col] == 0][time_col].values
    for ct in censored:
        idx = max(np.searchsorted(times, ct, side="right") - 1, 0)
        ax.plot(ct, surv[idx], "|", color=color, markersize=9, markeredgewidth=1.5)

    ax.set_title(f"Kaplan-Meier {endpoint} - Population globale", fontsize=13)
    ax.set_xlabel("Time (years)", fontsize=11)
    ax.set_ylabel("Survival Probability", fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
df_all = pd.read_csv(CSV_PATH, sep=";")

df_os  = load_marker_data(df_all.copy(), TIME_COL_OS, EVENT_COL_OS,  cap=CAP_YEARS)
df_pfs = load_marker_data(df_all.copy(), TIME_COL_PFS, EVENT_COL_PFS, cap=CAP_YEARS)

plot_single_km(df_os, TIME_COL_OS,  EVENT_COL_OS, "OS",  os.path.join(OUTPUT_DIR, "km_OS_ia2hl.png"))
plot_single_km(df_pfs, TIME_COL_PFS, EVENT_COL_PFS, "PFS", os.path.join(OUTPUT_DIR, "km_PFS_ia2hl.png"))

print(f"\nTerminé. Graphes dans {OUTPUT_DIR}/")


