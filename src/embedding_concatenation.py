"""
EmbeddingBagDataset — dataset MIL à partir de dossiers d'embeddings CSV.

Format CSV attendu (sans en-tête) :
    col 0  : x  (coordonnée)
    col 1  : y  (coordonnée)
    col 2+ : features

Un dossier = un bag. Un bag peut contenir un ou plusieurs CSV ;
toutes les instances sont concaténées en un seul tensor (N, D).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class EmbeddingBagDataset(Dataset):
    """Dataset MIL chargeant des bags depuis des dossiers d'embeddings CSV.

    Args:
        bag_folders: Un chemin de dossier par bag.
        labels:      Label entier correspondant à chaque bag.
        use_coords:  Si True, les colonnes x et y sont conservées comme
                     deux premières dimensions de feature (défaut : False).
    """

    def __init__(
        self,
        bag_folders: list[str | Path],
        labels: list[int],
        use_coords: bool = False,
    ) -> None:
        if len(bag_folders) != len(labels):
            raise ValueError(
                f"bag_folders ({len(bag_folders)}) et labels ({len(labels)}) "
                "doivent avoir la même longueur."
            )
        if not bag_folders:
            raise ValueError("bag_folders ne peut pas être vide.")

        self.bag_folders: list[Path] = [Path(f) for f in bag_folders]
        self.labels: list[int] = list(labels)
        self.use_coords: bool = use_coords

        missing = [str(f) for f in self.bag_folders if not f.is_dir()]
        if missing:
            raise FileNotFoundError(
                f"Dossier(s) introuvable(s) : {', '.join(missing)}"
            )

    # ------------------------------------------------------------------
    # Chargement interne
    # ------------------------------------------------------------------

    def _load_bag(self, folder: Path) -> torch.Tensor:
        """Charge et concatène tous les CSV du dossier → tensor (N, D)."""
        csv_files = sorted(folder.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"Aucun fichier CSV dans : {folder}")

        parts: list[np.ndarray] = []
        for path in csv_files:
            df = pd.read_csv(path, header=None, dtype=np.float32)
            if df.shape[1] < 3:
                raise ValueError(
                    f"{path} : au moins 3 colonnes attendues (x, y, feat…), "
                    f"trouvé {df.shape[1]}."
                )
            arr = df.values if self.use_coords else df.iloc[:, 2:].values
            parts.append(arr)

        return torch.from_numpy(np.concatenate(parts, axis=0))  # (N, D)

    # ------------------------------------------------------------------
    # Interface Dataset
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.bag_folders)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retourne (bag, label).

        Returns:
            bag:   Tensor de shape (N, D) — N instances, D features.
            label: Tensor scalaire de type torch.long.
        """
        bag = self._load_bag(self.bag_folders[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return bag, label


# ======================================================================
# Collate function
# ======================================================================


def collate_bags(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function pour des bags de taille variable.

    Zero-pad les bags à la taille maximale du batch et produit un masque
    booléen compatible avec ``torchmil.models.ABMIL(forward(X, mask))``.

    Args:
        batch: Liste de ``(bag, label)`` renvoyée par EmbeddingBagDataset.

    Returns:
        bags:   Tensor de shape ``(B, max_N, D)``  — instances paddées.
        labels: Tensor de shape ``(B,)``            — dtype torch.long.
        mask:   Tensor de shape ``(B, max_N)``      — uint8 : 1 = réel, 0 = padding.
    """
    bags, labels = zip(*batch)

    max_n: int = max(b.size(0) for b in bags)
    feat_dim: int = bags[0].size(1)
    batch_size: int = len(bags)

    padded = bags[0].new_zeros(batch_size, max_n, feat_dim)
    mask = torch.zeros(batch_size, max_n, dtype=torch.uint8)

    for i, bag in enumerate(bags):
        n = bag.size(0)
        padded[i, :n] = bag
        mask[i, :n] = 1

    return padded, torch.stack(labels), mask
