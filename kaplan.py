import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator


# ── Configuration ──────────────────────────────────────────────────────────
CSV_PATH    = r"csv/multi_label_patient_id.csv"
EVENT_COL   = "status"
MARKER_LIST = ["BCL2", "BCL6", "CD10", "HE", "MUM1", "MYC"]
OUTPUT_DIR  = "out_kaplan"
CAP_YEARS   = 5.0
# ───────────────────────────────────────────────────────────────────────────

COLORS = {"OS": "steelblue", "PFS": "tomato"}


def load_marker_data(df_all, marker, time_col, event_col, cap=None):
    df = df_all[df_all["stain"] == marker][["patient_id", time_col, event_col]].dropna()
    df = df.copy()
    df[event_col] = df[event_col].astype(int)
    df[time_col]  = df[time_col].astype(float)
    df = df[df[time_col] >= 0]
    if cap is not None:
        mask = df[time_col] > cap
        df.loc[mask, event_col] = 0
        df.loc[mask, time_col]  = cap
    return df


def plot_combined_km(df_os, df_pfs, os_time, pfs_time, event_col,
                     marker, output_path):
    fig, ax = plt.subplots(figsize=(9, 6))

    for endpoint, df, time_col in [
        ("OS",  df_os,  os_time),
        ("PFS", df_pfs, pfs_time),
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

    ax.set_title(f"Kaplan-Meier OS + PFS — {marker}", fontsize=13)
    ax.set_xlabel("Temps (années)", fontsize=11)
    ax.set_ylabel("Probabilité de survie", fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
df_all = pd.read_csv(CSV_PATH)

for marker in MARKER_LIST:
    df_os  = load_marker_data(df_all, marker, "OS",  EVENT_COL, cap=CAP_YEARS)
    df_pfs = load_marker_data(df_all, marker, "PFS", EVENT_COL, cap=CAP_YEARS)

    print(f"{marker} — OS: N={len(df_os)}, events={df_os[EVENT_COL].sum()} "
          f"| PFS: N={len(df_pfs)}, events={df_pfs[EVENT_COL].sum()}")

    plot_combined_km(
        df_os, df_pfs, "OS", "PFS", EVENT_COL,
        marker=marker,
        output_path=os.path.join(OUTPUT_DIR, f"km_{marker}.png"),
    )

print(f"\nTerminé. Graphes dans {OUTPUT_DIR}/")
