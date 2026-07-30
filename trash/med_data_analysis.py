import pandas as pd


def describe_continuous(csv_path: str, column: str, sep: str = ",") -> None:
    df = pd.read_csv(csv_path, sep=sep)

    if column not in df.columns:
        print(f"Colonne '{column}' introuvable. Colonnes disponibles :")
        print("  " + "\n  ".join(df.columns.tolist()))
        return

    series = pd.to_numeric(df[column], errors="coerce")
    n_invalid = series.isna().sum()
    series = series.dropna()

    q1  = series.quantile(0.25)
    q3  = series.quantile(0.75)
    iqr = q3 - q1

    stats = {
        "N valides"  : len(series),
        "N manquants": n_invalid,
        "Minimum"    : series.min(),
        "Q1 (25%)"   : q1,
        "Médiane"    : series.median(),
        "Moyenne"    : series.mean(),
        "Q3 (75%)"   : q3,
        "Maximum"    : series.max(),
        "IQR"        : iqr,
        "Écart-type" : series.std(),
    }

    col_width = max(len(k) for k in stats) + 2
    print(f"\n{'─' * 40}")
    print(f"  Colonne : {column}")
    print(f"{'─' * 40}")
    for label, value in stats.items():
        if isinstance(value, float):
            print(f"  {label:<{col_width}} {value:.4f}")
        else:
            print(f"  {label:<{col_width}} {value}")
    print(f"{'─' * 40}\n")

def describe_categorical(csv_path: str, column: str, sep: str = ",") -> None:
    df = pd.read_csv(csv_path, sep=sep)

    if column not in df.columns:
        print(f"Colonne '{column}' introuvable. Colonnes disponibles :")
        print("  " + "\n  ".join(df.columns.tolist()))
        return

    series = df[column].astype(str).replace("nan", pd.NA).dropna()
    n_total   = len(series)
    n_missing = len(df) - n_total

    counts = series.value_counts().sort_index()

    col_width = max(len(str(k)) for k in counts.index) + 2

    print(f"\n{'─' * 45}")
    print(f"  Colonne : {column}  (N={n_total}, manquants={n_missing})")
    print(f"{'─' * 45}")
    print(f"  {'Classe':<{col_width}} {'N':>6}  {'%':>7}")
    print(f"  {'─'*(col_width + 17)}")
    for label, count in counts.items():
        pct = count / n_total * 100
        print(f"  {str(label):<{col_width}} {count:>6}  {pct:>6.1f}%")
    print(f"{'─' * 45}\n")





# ── Exemples d'utilisation ──────────────────────────────────────────────

csv_path = r"csv_calym/merged_calym.csv"

#column_name = "OS"
#describe_continuous(csv_path, column_name)

column_name_cat = "IPI score"
describe_categorical(csv_path, column_name_cat)
