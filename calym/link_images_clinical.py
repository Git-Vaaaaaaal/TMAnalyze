"""Associe chaque image WSI a sa ligne de donnees cliniques via le patient_id."""

import os
import re

import pandas as pd

IMAGES_DIR = "/data/TMAnalyze/img"
IMAGE_EXTENSION = ".tiff"

CLINICAL_CSV_PATH = "csv_calym/merged_calym.csv"
CLINICAL_ID_COLUMN = "patient_id"

OUTPUT_PATH = "csv_calym/ia2hl_clinical.csv"


def normalize_patient_id(patient_id: str) -> str:
    """"xx_yyy_extra" -> "xx_yyy" ; numero sur 2 chiffres -> complete par un 0 ("xx_yy" -> "xx_0yy")."""
    prefix, number = re.split(r"[_-]", str(patient_id))[:2]
    return f"{prefix}-{number.zfill(3)}"



def list_image_patients() -> pd.DataFrame:
    filenames = [f for f in os.listdir(IMAGES_DIR) if f.endswith(IMAGE_EXTENSION)]
    df = pd.DataFrame({"filename": filenames})
    df["patient_id"] = df["filename"].str.removesuffix(IMAGE_EXTENSION)
    df["patient_id_norm"] = df["patient_id"].apply(normalize_patient_id)
    return df


def load_clinical_data() -> pd.DataFrame:
    df = pd.read_csv(CLINICAL_CSV_PATH)
    df["patient_id_norm"] = df[CLINICAL_ID_COLUMN].apply(normalize_patient_id)
    return df


def merge_images_clinical() -> pd.DataFrame:
    images = list_image_patients()
    clinical = load_clinical_data()

    merged = images.merge(
        clinical, on="patient_id_norm", how="left", suffixes=("", "_clinical"), indicator=True
    )

    print(merged["_merge"].value_counts())
    merged = merged.drop(columns=["_merge"])

    return merged


if __name__ == "__main__":
    merged = merge_images_clinical()
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"Fichier ecrit dans {OUTPUT_PATH}")
