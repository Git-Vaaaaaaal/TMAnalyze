import pandas as pd

input_path = r"csv/id_label_patient_complete.csv"
label_path = r"csv/clinical_data_cleaned.csv"

list_colonne = ["Age", "ECOG PS", "LDH", "EN", "Stage", "IPI Score", "IPI Risk Group (4 Class)", "RIPI Risk Group",
                "OS", "PFS"]

def merge_labels(
    ids_csv    : str,
    labels_csv : str,
    on         : str       = "old_patient_id",
    id_cols    : list[str] = None,
    label_cols : list[str] = list_colonne,
    how        : str       = "left",
) -> pd.DataFrame:
    """
    Fusionne les colonnes de labels_csv dans le dataframe ids_csv.

    Args:
        ids_csv    : chemin vers le CSV contenant les identifiants.
        labels_csv : chemin vers le CSV contenant les labels.
        on         : colonne clé de jointure (doit exister dans les deux CSV).
        id_cols    : colonnes à conserver depuis ids_csv (None = toutes).
        label_cols : colonnes à récupérer depuis labels_csv (None = toutes sauf doublons).
        how        : type de jointure pandas — 'left' garde tous les ids.

    Returns:
        DataFrame fusionné.
    """
    df_ids    = pd.read_csv(ids_csv)
    df_labels = pd.read_csv(labels_csv)

    if id_cols is not None:
        keep   = [on] + [c for c in id_cols if c != on]
        df_ids = df_ids[keep]

    if label_cols is not None:
        df_labels = df_labels[[on] + [c for c in label_cols if c != on]]
    else:
        cols_to_add = [on] + [c for c in df_labels.columns if c != on and c not in df_ids.columns]
        df_labels   = df_labels[cols_to_add]

    return df_ids.merge(df_labels, on=on, how=how)


if __name__ == "__main__":
    result = merge_labels(
        ids_csv    = input_path,
        labels_csv = label_path,
    )
    print(result.head())
    result.to_csv("csv/multi_label_patient_id.csv", index=False)
