"""Fusionne Export_AHL.csv et IA2HL_clinical_data.csv sur le numero de patient."""

from pathlib import Path

import pandas as pd

CSV_DIR = Path(__file__).resolve().parent.parent / "csv_calym"

EXPORT_AHL_PATH = CSV_DIR / "Export_AHL.csv"
IA2HL_PATH = CSV_DIR / "IA2HL_clinical_data.csv"
OUTPUT_PATH = CSV_DIR / "merged_calym.csv"

EXPORT_AHL_KEY = "PATIENT DANS L'ÉTUDE - Numéro d'inclusion"
IA2HL_KEY = "Randomisation\nnumber"


def load_export_ahl() -> pd.DataFrame:
    df = pd.read_csv(EXPORT_AHL_PATH, sep=";", encoding="utf-8")
    df[EXPORT_AHL_KEY] = df[EXPORT_AHL_KEY].astype(str).str.strip()
    return df


def load_ia2hl() -> pd.DataFrame:
    df = pd.read_csv(IA2HL_PATH, sep=";", encoding="utf-8")
    df[IA2HL_KEY] = df[IA2HL_KEY].astype(str).str.strip()
    return df


def merge_csv() -> pd.DataFrame:
    export_ahl = load_export_ahl()
    ia2hl = load_ia2hl()

    merged = export_ahl.merge(
        ia2hl,
        left_on=EXPORT_AHL_KEY,
        right_on=IA2HL_KEY,
        how="outer",
        indicator=True,
    )

    print(merged["_merge"].value_counts())

    return merged


if __name__ == "__main__":
    merged = merge_csv()
    merged.to_csv(OUTPUT_PATH, sep=";", index=False, encoding="utf-8")
    print(f"Fichier fusionne ecrit dans {OUTPUT_PATH}")
