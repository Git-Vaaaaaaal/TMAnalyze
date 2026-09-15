import matplotlib.pyplot as plt
import pandas as pd


CSV_PATH = "csv/multi_label_patient_id.csv"
MARKERS  = ["BCL2", "BCL6", "CD10", "HE", "MUM1", "MYC"]

# Put categories in the order you want them to appear in the plot
CATEGORIES = [
    {
        "name":        "IPI Score",
        "column":      "IPI Score",
        "group_low":   [0, 1],
        "group_high":  [2, 3, 4, 5],
        "label_low":   "(0 - 1)",
        "label_high":  "(2 - 5)",
        "color_low":   "#4C72B0",
        "color_high":  "#DD8452",
    },
    {
        "name":        "IPI Risk Group",
        "column":      "IPI Risk Group (4 Class)",
        "group_low":   [0.0, 1.0],
        "group_high":  [2.0, 3.0],
        "label_low":   "(0.0 - 1.0)",
        "label_high":  "(2.0 - 5.0)",
        "color_low":   "#17099B",
        "color_high":  "#DDDB52",
    },
]

BAR_WIDTH  = 0.6
GROUP_GAP  = 0.35
Y_LINE     = -0.20
Y_LABEL    = -0.24  
FIGSIZE    = (8, 5)
TITLE      = "Groupes IPI Score et IPI Risk Group par marqueur"


def draw_group_delimiter(ax, start, end, label):
    """Trait horizontal + petits traits verticaux + nom du marqueur sous un groupe de barres."""
    mid = (start + end) / 2
    ax.text(mid, Y_LABEL, label, transform=ax.get_xaxis_transform(),
             ha="center", va="top", fontsize=11, fontweight="bold", color="gray")

    arrow = dict(arrowstyle="-", color="gray", lw=1.2)
    ax.annotate("", xy=(end + GROUP_GAP, Y_LINE), xytext=(start - GROUP_GAP, Y_LINE),
                xycoords=("data", "axes fraction"), textcoords=("data", "axes fraction"),
                arrowprops=arrow, annotation_clip=False)

    for xpos in (start - GROUP_GAP, end + GROUP_GAP):
        ax.annotate("", xy=(xpos, Y_LINE - 0.02), xytext=(xpos, Y_LINE + 0.02),
                    xycoords=("data", "axes fraction"), textcoords=("data", "axes fraction"),
                    arrowprops=arrow, annotation_clip=False)


def plot_marker_group(ax, df_marker, start, show_legend):
    """Trace une barre empilée bas/haut par catégorie, à partir de la position `start`."""
    for j, cat in enumerate(CATEGORIES):
        x      = start + j
        n_low  = len(df_marker[df_marker[cat["column"]].isin(cat["group_low"])])
        n_high = len(df_marker[df_marker[cat["column"]].isin(cat["group_high"])])

        ax.bar(x, n_low,  BAR_WIDTH, color=cat["color_low"],
               label=cat["label_low"] if show_legend else None)
        ax.bar(x, n_high, BAR_WIDTH, bottom=n_low, color=cat["color_high"],
               label=cat["label_high"] if show_legend else None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df_pop = pd.read_csv(CSV_PATH)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(bottom=0.22)  # espace pour les labels sous l'axe

    x_ticks, x_labels = [], []

    for i, marker in enumerate(MARKERS):
        df_marker = df_pop[df_pop["stain"] == marker]
        start     = i * len(CATEGORIES)

        plot_marker_group(ax, df_marker, start, show_legend=(i == 0))
        draw_group_delimiter(ax, start, start + len(CATEGORIES) - 1, marker)

        x_ticks.extend(range(start, start + len(CATEGORIES)))
        x_labels.extend(cat["name"] for cat in CATEGORIES)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_title(TITLE)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
