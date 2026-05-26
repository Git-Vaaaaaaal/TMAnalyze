"""
EmbeddingBagDataset — dataset MIL multi-images à partir de dictionnaires.

Format CSV attendu (sans en-tête) :
    col 0  : x  (coordonnée)
    col 1  : y  (coordonnée)
    col 2+ : features

Chaque patient = une liste de chemins CSV (un par image).
Chaque CSV → un tenseur indépendant (N_i, D).
Les images entre elles n'ont pas de corrélation (pas d'alignement sur x,y).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class EmbeddingBagDataset(Dataset):
    """Dataset MIL multi-images.

    Args:
        bag_dict:   {patient_id: [chemin_csv_img1, ..., chemin_csv_imgK]}
        label_dict: {patient_id: label_entier}
        use_coords: Si True, les colonnes x et y (cols 0-1) sont incluses
                    dans les features. Par défaut False.

    Seuls les patients présents dans les deux dictionnaires sont conservés.
    """

    def __init__(
        self,
        bag_dict:   dict[str, list[str | Path]],
        label_dict: dict[str, int],
        use_coords: bool = False,
    ) -> None:
        patient_ids = sorted(set(bag_dict.keys()) & set(label_dict.keys()))
        if not patient_ids:
            raise ValueError("Aucun patient commun entre bag_dict et label_dict.")

        self.patient_ids: list[str]        = patient_ids
        self.csv_paths:   list[list[Path]] = [
            [Path(p) for p in bag_dict[pid]] for pid in patient_ids
        ]
        self.labels:      list[int]        = [label_dict[pid] for pid in patient_ids]
        self.use_coords:  bool             = use_coords

    # ------------------------------------------------------------------
    # Chargement interne
    # ------------------------------------------------------------------

    def _load_csv(self, path: Path) -> torch.Tensor:
        """Charge un fichier CSV → tenseur (N, D)."""
        df = pd.read_csv(path, dtype=np.float32)
        if df.shape[1] < 3:
            raise ValueError(
                f"{path} : au moins 3 colonnes attendues (x, y, feat…), "
                f"trouvé {df.shape[1]}."
            )
        arr = df.values if self.use_coords else df.iloc[:, 2:].values
        return torch.from_numpy(arr.astype(np.float32))  # (N, D)

    # ------------------------------------------------------------------
    # Interface Dataset
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Retourne (liste_de_tenseurs, label).

        Returns:
            images: Liste de tenseurs (N_i, D), un par fichier CSV.
            label:  Tensor scalaire dtype torch.long.
        """
        images = [self._load_csv(p) for p in self.csv_paths[idx]]
        label  = torch.tensor(self.labels[idx], dtype=torch.long)
        return images, label


# ======================================================================
# Collate function
# ======================================================================


def collate_bags(
    batch: list[tuple[list[torch.Tensor], torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function pour des bags multi-images de taille variable.

    Chaque bag est une liste de K tenseurs (N_i, D). La fonction construit :
      - un tenseur paddé ``(B, K_max, N_max, D)``
      - un masque         ``(B, K_max, N_max)``

    Args:
        batch: Liste de ``(images, label)`` renvoyée par EmbeddingBagDataset.

    Returns:
        bags:   Tensor ``(B, K_max, N_max, D)`` — zero-paddé.
        labels: Tensor ``(B,)``                  — dtype torch.long.
        mask:   Tensor ``(B, K_max, N_max)``      — uint8 : 1=réel, 0=padding.
    """
    images_list, labels = zip(*batch)

    B     = len(images_list)
    K_max = max(len(imgs) for imgs in images_list)
    N_max = max(t.size(0) for imgs in images_list for t in imgs)
    D     = images_list[0][0].size(1)

    bags = torch.zeros(B, K_max, N_max, D, dtype=torch.float32)
    mask = torch.zeros(B, K_max, N_max,    dtype=torch.uint8)

    for b, imgs in enumerate(images_list):
        for k, t in enumerate(imgs):
            n = t.size(0)
            bags[b, k, :n] = t
            mask[b, k, :n] = 1

    return bags, torch.stack(labels), mask
