import matplotlib.pyplot as plt
import pandas as pd


df_pop = pd.read_csv("csv/multi_label_patient_id.csv")

ipi_low = [0, 1]
ipi_high = [2, 3, 4, 5]

ipi_r_low = [0.0, 1.0]
ipi_r_high = [2.0, 3.0]

categories = ["IPI Score", "IPI Risk Group"]
groups     = [("BCL2", 0, 2), ("BCL6", 2, 4), ("CD10", 4, 6), ("HE", 6, 8), ("MUM1", 8, 10), ("MYC", 10, 12)]

fig, ax = plt.subplots(figsize=(8, 5))
fig.subplots_adjust(bottom=0.22)   # ← espace pour les labels
width = 0.6
y_line  = -0.20
y_label = -0.24


x_ticks, x_labels = [], []

for i, (marker, start, end) in enumerate(groups):
    df_marker = df_pop[df_pop["stain"] == marker]
    nb_ipi_low   = len(df_marker[df_marker["IPI Score"].isin(ipi_low)])
    nb_ipi_high  = len(df_marker[df_marker["IPI Score"].isin(ipi_high)])
    nb_ripi_low  = len(df_marker[df_marker["IPI Risk Group (4 Class)"].isin(ipi_r_low)])
    nb_ripi_high = len(df_marker[df_marker["IPI Risk Group (4 Class)"].isin(ipi_r_high)])

    label_low  = '(0 - 1)' if i == 0 else None
    label_high = '(2 - 5)' if i == 0 else None

    label_low_ripi  = '(0.0 - 1.0)' if i == 0 else None
    label_high_ripi = '(2.0 - 5.0)' if i == 0 else None

    # Barre IPI Score à la position `start`
    ax.bar(start,     nb_ipi_low,   width, label=label_low,  color='#4C72B0')
    ax.bar(start,     nb_ipi_high,  width, bottom=nb_ipi_low,  label=label_high, color='#DD8452')

    # Barre IPI Risk Group à la position `start + 1`
    ax.bar(start + 1, nb_ripi_low,  width, label=label_low_ripi,  color="#17099B")
    ax.bar(start + 1, nb_ripi_high, width, bottom=nb_ripi_low, label=label_high_ripi, color="#DDDB52")

    x_ticks.extend([start, start + 1])
    x_labels.extend(["IPI Score", "IPI Risk Group"])

    #Délimitation marker
    mid = (start + end - 1) / 2
    ax.text(mid, y_label, marker,
            transform=ax.get_xaxis_transform(),
            ha='center', va='top',
            fontsize=11, fontweight='bold', color='gray')

    # trait horizontal sous le groupe
    ax.annotate('', xy=(end - 1 + 0.35, y_line),
                xytext=(start - 0.35, y_line),
                xycoords=('data', 'axes fraction'),
                textcoords=('data', 'axes fraction'),
                arrowprops=dict(arrowstyle='-',
                                color='gray', lw=1.2),
                annotation_clip=False)

    # petits traits verticaux aux extrémités
    for xpos in [start - 0.35, end - 1 + 0.35]:
        ax.annotate('', xy=(xpos, y_line - 0.02),
                    xytext=(xpos, y_line + 0.02),
                    xycoords=('data', 'axes fraction'),
                    textcoords=('data', 'axes fraction'),
                    arrowprops=dict(arrowstyle='-',
                                    color='gray', lw=1.2),
                    annotation_clip=False)

ax.set_xticks(x_ticks)
ax.title.set_text("Groupes IPI Score et IPI Risk Group par marqueur")
ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
ax.legend()
plt.show()

