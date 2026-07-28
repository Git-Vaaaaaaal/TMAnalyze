import pandas as pd 


csv_path = r"C:\Users\valbo\Desktop\archive\IA2HL\Export_AHL.csv"  # Replace with your CSV file path
df_csv = pd.read_csv(csv_path, sep=";", encoding="cp1252")  # Export Windows/French -> cp1252, pas utf-8

df_csv["Identifiant échantillon"] = df_csv["Identifiant échantillon"].str.replace("/", "-")
df_csv.to_csv(r"C:\Users\valbo\Desktop\archive\IA2HL\Export_AHL.csv", sep=";", index=False, encoding="utf-8")  # Save the modified DataFrame back to CSV with UTF-8 encoding