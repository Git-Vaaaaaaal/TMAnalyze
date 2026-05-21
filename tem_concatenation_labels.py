import pandas as pd

path = r"csv\multi_label_patient_id.csv"
df_label = pd.read_csv(path)

marker_list = ["BCL2", "BCL6", "CD10", "HE", "MUM1", "MYC"]

sets_per_marker = []
for marker in marker_list:
    ids = set(str(id) for id in df_label[df_label["stain"] == marker]["old_patient_id"].unique())
    sets_per_marker.append(ids)

common_patients = set.intersection(*sets_per_marker)

print(f"{len(common_patients)} patients présents dans tous les stains")

df_label["old_patient_id"] = df_label["old_patient_id"].astype(str)
df_out = df_label[df_label["old_patient_id"].isin(common_patients)].reset_index(drop=True)

df_out.to_csv(r"csv\common_patients_labels.csv", index=False)
print(f"Saved {len(df_out)} rows → csv/common_patients_labels.csv")
