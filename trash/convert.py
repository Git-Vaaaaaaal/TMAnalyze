import pandas as pd 



def convert_float_to_int(df, column):
    """
    Convertit une colonne de float à int, en gérant les NaN.

    Args:
        df     : DataFrame contenant les données
        column : nom de la colonne à convertir
    """
    if column not in df.columns:
        print(f"Colonne '{column}' introuvable. Colonnes disponibles :")
        print("  " + "\n  ".join(df.columns.tolist()))
        return df

    df[column] = df[column].astype("Int64")
    return df


df = pd.read_csv("csv/multi_label_patient_id.csv")
status_list = ["IPI Score", "IPI Risk Group (4 Class)", "ECOG PS", "LDH", "Stage"] #["IPI Score", "IPI Risk Group (4 Class)", "ECOG PS", "LDH", "Stage"]

for status in status_list:
    df = convert_float_to_int(df, status)

df.to_csv("csv/multi_label_patient_id.csv", index=False)