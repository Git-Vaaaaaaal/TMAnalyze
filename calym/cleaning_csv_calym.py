import pandas as pd
import os
import numpy as np


def binarize_column(df, column: str, group_0: list, group_1: list) -> pd.DataFrame:
    """
    Regroupe les valeurs d'une colonne en 0 / 1.

    Args:
        df         : DataFrame contenant les données
        column     : colonne à transformer
        group_0    : liste des valeurs à mapper → 0
        group_1    : liste des valeurs à mapper → 1
    """

    if column not in df.columns:
        print(f"Colonne '{column}' introuvable. Colonnes disponibles :")
        print("  " + "\n  ".join(df.columns.tolist()))
        return df

    mapping = {str(v): 0 for v in group_0}
    mapping.update({str(v): 1 for v in group_1})

    def _norm(s):
        try:
            return str(int(float(s)))
        except (ValueError, TypeError):
            return s

    src = df[column].astype(str).apply(_norm)
    unknown = set(src.dropna().unique()) - set(mapping.keys()) - {"nan"}
    if unknown:
        print(f"[ATTENTION] Valeurs hors group_0/group_1 {unknown} -> lignes supprimées")

    result = src.map(mapping)

    df = df[~(result.isna() & (src != "nan"))]
    result = result.loc[df.index]

    df[column] = result.astype("Int64")

    n_0       = (df[column] == 0).sum()
    n_1       = (df[column] == 1).sum()
    n_missing = df[column].isna().sum()

    print(f"\n{'─' * 45}")
    print(f"{'─' * 45}")
    print(f"  Groupe 0  ({group_0}) : {n_0} patients")
    print(f"  Groupe 1  ({group_1}) : {n_1} patients")
    print(f"  Non mappés / NaN      : {n_missing}")
    print(f"{'─' * 45}\n")
    return df



def cleaning_csv(df_path, encoder, element, group_0, group_1):
    df_label = pd.read_csv(df_path)
    keep_cols = [c for c in ["patient_id", "old_patient_id", element] if c in df_label.columns]
    df_label = df_label[keep_cols].rename(columns={element: "Status"})
    df_label["Status"] = df_label["Status"].replace("", np.nan)
    df_label = df_label.dropna(subset=["Status"])
    df_label = binarize_column(df_label, "Status", group_0=group_0, group_1=group_1)
    df_label = df_label.dropna(subset=["Status"])
    df_label["Status"] = pd.factorize(df_label["Status"])[0]
    os.makedirs("temporary_mil_csv", exist_ok=True)
    out_csv_marker = os.path.join("temporary_mil_csv", f"{encoder}_{element}.csv")
    df_label.to_csv(out_csv_marker, index=False)
    print(f"  [{element}] {len(df_label)} patients avec valeur valide")
    return out_csv_marker, df_label

