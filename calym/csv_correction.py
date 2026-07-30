import pandas as pd 


df_csv = pd.read_csv("csv_calym/merged_calym.csv", sep=";")

df_csv = df_csv.dropna(subset=["Identifiant échantillon"], how="any")
df_csv = df_csv.drop(columns=["COLLECTION - Nom de la collection",
                              "Randomisation number", "Arm"])
df_csv = df_csv.rename(columns={"PATIENT DANS L'ÉTUDE - Numéro d'inclusion": "old_patient_id",
                                "Identifiant échantillon": "patient_id"})


df_csv.to_csv("csv_calym/ia2hl_clinical.csv", index=False)
