import pandas as pd


path = r"csv/id_label_patient_complete.csv"


df_patient = pd.read_csv(path)


stain_list = ["BCL2", "BCL6", "CD10", "MUM1", "MYC", "HE"]

for stain in stain_list :
    df_stain = df_patient[df_patient["stain"]==stain]
    nb_patient = len(df_stain[df_stain["status"] == 0])
    guy = len(df_stain) - nb_patient
    print(f"marqueur : {stain} --> 0 : {nb_patient}/{len(df_stain)}")
    print(f"marqueur : {stain} --> 1 : {guy}/{len(df_stain)}")