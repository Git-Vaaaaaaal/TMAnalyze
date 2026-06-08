import os
import copy
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensordict import TensorDict
from torchmil.utils import build_adj, normalize_adj, add_self_loops


class CSVMILDataset(torch.utils.data.Dataset):
    """
    Dataset MIL à partir de fichiers CSV, compatible avec l'écosystème TorchMIL.

    Sources :
        - slide_features_csv : un seul CSV global (patient_id | feat_0 | feat_1 | ...)
        - tiles_dir          : un dossier avec un CSV par patient (x | y | feat_0 | feat_1 | ...)
        - labels_csv         : un CSV global (patient_id | label)

    bag_keys possibles :
        - "X"       : features tuiles (tiles_dir)
        - "X_slide" : features slides (slide_features_csv)
        - "Y"       : label patient (labels_csv)
        - "coords"  : coordonnées x, y (tiles_dir)
        - "adj"     : matrice d'adjacence construite depuis les coords
    """

    def __init__(
        self,
        slide_features_csv: str = None,
        tiles_dir: str = None,
        labels_csv: str = None,
        bag_keys: list = ["X", "X_slide", "Y", "coords"],
        patient_ids: list = None,
        slide_id_col: str = "patient_id",
        dist_thr: float = 1.5,
        adj_with_dist: bool = False,
        norm_adj: bool = True,
        load_at_init: bool = False,
        verbose: bool = True,
    ):
        super().__init__()

        self.slide_features_csv = slide_features_csv
        self.tiles_dir = tiles_dir
        self.labels_csv = labels_csv
        self.bag_keys = bag_keys
        self.slide_id_col = slide_id_col
        self.dist_thr = dist_thr
        self.adj_with_dist = adj_with_dist
        self.norm_adj = norm_adj
        self.load_at_init = load_at_init
        self.verbose = verbose

        # Vérifications
        if "X" in bag_keys and tiles_dir is None:
            raise ValueError("tiles_dir doit être fourni si 'X' est dans bag_keys")
        if "coords" in bag_keys and tiles_dir is None:
            raise ValueError("tiles_dir doit être fourni si 'coords' est dans bag_keys")
        if "adj" in bag_keys and tiles_dir is None:
            raise ValueError("tiles_dir doit être fourni si 'adj' est dans bag_keys")
        if "X_slide" in bag_keys and slide_features_csv is None:
            raise ValueError("slide_features_csv doit être fourni si 'X_slide' est dans bag_keys")
        if "Y" in bag_keys and labels_csv is None:
            raise ValueError("labels_csv doit être fourni si 'Y' est dans bag_keys")

        # Chargement des sources globales — index converti en str pour éviter
        # les bugs int/str identiques au fix de TridentWSIDataset
        self.df_slides = None
        if slide_features_csv is not None:
            self.df_slides = pd.read_csv(slide_features_csv)
            self.df_slides[slide_id_col] = self.df_slides[slide_id_col].astype(str).apply(
                lambda x: os.path.splitext(x)[0]
            )
            self.df_slides = self.df_slides.rename(columns={slide_id_col: "patient_id"}).set_index("patient_id")

        self.df_labels = None
        if labels_csv is not None:
            self.df_labels = pd.read_csv(labels_csv)
            self.df_labels["patient_id"] = self.df_labels["patient_id"].astype(str).apply(
                lambda x: os.path.splitext(x)[0]
            )
            self.df_labels = self.df_labels.set_index("patient_id")

        # Détermination de la liste des patients
        if patient_ids is not None:
            self.patient_ids = sorted(
                str(os.path.splitext(pid)[0]) for pid in patient_ids
            )
        elif self.df_labels is not None:
            self.patient_ids = sorted(self.df_labels.index.tolist())
        elif self.df_slides is not None:
            self.patient_ids = sorted(self.df_slides.index.tolist())
        elif tiles_dir is not None:
            self.patient_ids = sorted(
                os.path.splitext(f)[0]
                for f in os.listdir(tiles_dir)
                if f.endswith(".csv")
            )
        else:
            raise ValueError("Impossible de déterminer la liste des patients.")

        # Filtrage : ne garder que les patients dont le CSV de tiles existe réellement
        if tiles_dir is not None and "X" in bag_keys:
            before = len(self.patient_ids)
            self.patient_ids = [
                pid for pid in self.patient_ids
                if os.path.isfile(os.path.join(tiles_dir, f"{pid}.csv"))
            ]
            removed = before - len(self.patient_ids)
            if removed > 0 and verbose:
                print(f"[CSVMILDataset] {removed} patient(s) ignores (CSV tiles manquant sur {before})")

        # Chargement optionnel à l'init (même comportement que ProcessedMILDataset)
        self.loaded_bags = {}
        if self.load_at_init:
            pbar = tqdm(
                self.patient_ids,
                desc="Chargement des bags",
                unit="bag",
                disable=not self.verbose,
            )
            for pid in pbar:
                self.loaded_bags[pid] = self._build_bag(pid)

    # ------------------------------------------------------------------
    # Méthodes de chargement (même découpage que ProcessedMILDataset)
    # ------------------------------------------------------------------

    def _load_tiles(self, patient_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Charge le CSV tuiles → (coords, features)."""
        path = os.path.join(self.tiles_dir, f"{patient_id}.csv")
        df = pd.read_csv(path)
        coords = df[["x", "y"]].values.astype(np.float32)
        features = df.drop(columns=["x", "y"]).values.astype(np.float32)
        return coords, features

    def _load_slide_features(self, patient_id: str) -> np.ndarray:
        """Charge les features slide depuis le CSV global."""
        try:
            return self.df_slides.loc[patient_id].values.astype(np.float32)
        except KeyError:
            raise ValueError(f"Patient '{patient_id}' introuvable dans slide_features_csv.")

    def _load_label(self, patient_id: str) -> np.ndarray:
        """Charge le label depuis le CSV global."""
        try:
            return self.df_labels.loc[patient_id].values.astype(np.float32)
        except KeyError:
            raise ValueError(f"Patient '{patient_id}' introuvable dans labels_csv.")

    # ------------------------------------------------------------------
    # Construction de la matrice d'adjacence (identique à ProcessedMILDataset)
    # ------------------------------------------------------------------

    def _build_adj(self, coords: np.ndarray, features: np.ndarray) -> torch.Tensor:
        bag_size = coords.shape[0]
        X = features if self.adj_with_dist else None
        edge_index, edge_weight = build_adj(coords, X, dist_thr=self.dist_thr)
        norm_edge_weight = normalize_adj(edge_index, edge_weight, n_nodes=bag_size)
        if bag_size == 1:
            edge_index, norm_edge_weight = add_self_loops(edge_index, norm_edge_weight, bag_size)
        edge_val = norm_edge_weight if self.norm_adj else edge_weight
        return torch.sparse_coo_tensor(
            edge_index, edge_val, (bag_size, bag_size)
        ).coalesce()

    # ------------------------------------------------------------------
    # Construction du bag (même découpage que ProcessedMILDataset)
    # ------------------------------------------------------------------

    def _build_bag(self, patient_id: str) -> dict:
        bag = {}

        # Chargement des tuiles une seule fois si plusieurs clés en ont besoin
        coords, tile_features = None, None
        if any(k in self.bag_keys for k in ["X", "coords", "adj"]):
            coords, tile_features = self._load_tiles(patient_id)

        if "X" in self.bag_keys:
            bag["X"] = torch.from_numpy(tile_features)

        if "coords" in self.bag_keys:
            bag["coords"] = torch.from_numpy(coords)

        if "adj" in self.bag_keys:
            bag["adj"] = self._build_adj(coords, tile_features)

        if "X_slide" in self.bag_keys:
            bag["X_slide"] = torch.from_numpy(self._load_slide_features(patient_id))

        if "Y" in self.bag_keys:
            bag["Y"] = torch.from_numpy(self._load_label(patient_id))

        return bag

    # ------------------------------------------------------------------
    # Interface Dataset (identique à ProcessedMILDataset)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, index: int) -> TensorDict:
        patient_id = self.patient_ids[index]
        if patient_id not in self.loaded_bags:
            self.loaded_bags[patient_id] = self._build_bag(patient_id)
        bag = self.loaded_bags[patient_id]
        return TensorDict({k: bag[k] for k in self.bag_keys if k in bag})

    def get_bag_labels(self) -> list:
        return [self._load_label(pid) for pid in self.patient_ids]

    def get_bag_names(self) -> list:
        return self.patient_ids

    def subset(self, indices: list) -> "CSVMILDataset":
        new_dataset = copy.deepcopy(self)
        new_dataset.patient_ids = [self.patient_ids[i] for i in indices]
        new_dataset.loaded_bags = {
            k: v for k, v in new_dataset.loaded_bags.items()
            if k in new_dataset.patient_ids
        }
        return new_dataset
    


