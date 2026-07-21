import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.csv_status import binarize_column


status_dict = {
    "IPI Score": {
        "group_0": [0, 1, 2],
        "group_1": [3, 4, 5],
    },
    "IPI Risk Group (4 Class)": {
        "group_0": [0],
        "group_1": [1, 2, 3],
    },
    "ECOG PS": {
        "group_0": [0],
        "group_1": [1, 2, 3],
    },
    "Stage": {
        "group_0": [1, 2],
        "group_1": [3, 4],
    },
}


def convert_dlbcl(input_path, output_path):
    df = pd.read_csv(input_path)

    for col, cfg in status_dict.items():
        if col not in df.columns:
            print(f"[SKIP] Colonne absente : {col}")
            continue

        if "pre_map" in cfg:
            df[col] = df[col].map(cfg["pre_map"])

        df = binarize_column(df, col, group_0=cfg["group_0"], group_1=cfg["group_1"])

    df.to_csv(output_path, index=False)
    print(f"\nCSV binarisé → {output_path}")
    return df


if __name__ == "__main__":
    convert_dlbcl(
        input_path="/home/imvia/Bureau/valentin/r_studio/IA2HL.csv",
        output_path="/home/imvia/Bureau/valentin/r_studio/DLBCL_binarized.csv",
    )
