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

    src = df[column].astype(str)
    unknown = set(src.dropna().unique()) - set(mapping.keys()) - {"nan"}
    if unknown:
        print(f"[ATTENTION] Valeurs non mappées (seront NaN) : {unknown}")

    result = src.map(mapping)

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



def cleaning_csv(df_path, marker, encoder, element):
    df_label = pd.read_csv(df_path)
    df_label = df_label[df_label["stain"] == marker]
    df_label = df_label[["patient_id", element]].rename(columns={element: "Status"})
    df_label["Status"] = df_label["Status"].replace("", np.nan)
    df_label = df_label.dropna(subset=["Status"])
    df_label = binarize_column(df_label, "Status", group_0=[0.0], group_1=[1.0, 2.0, 3.0])
    df_label = df_label[df_label["Status"].astype(str).str.strip() != ""]
    df_label["Status"] = pd.factorize(df_label["Status"])[0]
    out_csv_marker = os.path.join("csv", f"{marker}_{encoder}.csv")
    df_label.to_csv(out_csv_marker, index=False)
    print(f"  [{element}] {len(df_label)} patients avec valeur valide")
    return out_csv_marker

