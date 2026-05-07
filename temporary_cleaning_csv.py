import os
import pandas as pd


marker_list = ["MUM1"]

    

def merge_csv_folder(input_dir: str, output_path: str) -> str:
    """
    Lit tous les CSV d'un dossier et les concatène en un seul fichier.

    Args:
        input_dir (str): Dossier contenant les fichiers CSV
        output_path (str): Chemin du CSV de sortie

    Returns:
        str: Chemin du fichier sauvegardé
    """
    dfs = []

    for file in os.listdir(input_dir):
        if file.endswith(".csv"):
            full_path = os.path.join(input_dir, file)
            df = pd.read_csv(full_path)
            dfs.append(df)

    if not dfs:
        raise ValueError("Aucun fichier CSV trouvé dans le dossier.")

    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv(output_path, index=False)

    return output_path



for marker in marker_list :
    path = os.path.join("rebuilt.prism", "prism", marker, "job_dir", "20.0x_64px_0px_overlap", "slide_features_prism")
    out = os.path.join("rebuilt.prism", "prism", marker, f"slide_encoder_{marker}.csv")
    merge_csv_folder(path, out)
    print(f"{marker} done")