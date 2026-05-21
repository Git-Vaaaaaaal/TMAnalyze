"""
train_concatenation.py — Entraînement ABMIL sur des embeddings chargés depuis des dossiers CSV.

Structure attendue :
    <DATA_DIR>/
        <patient_id>/        ← un dossier par bag
            patch_0.csv
            patch_1.csv      ← optionnel, plusieurs CSV concaténés
    <LABELS_CSV>             ← colonnes : <ID_COL>, <LABEL_COL>
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchmil.models import ABMIL

from embedding_concatenation import EmbeddingBagDataset, collate_bags


# ======================================================================
# Données
# ======================================================================


def load_bag_data(
    data_dir: str | Path,
    labels_csv: str | Path,
    id_col: str,
    label_col: str,
) -> tuple[list[Path], list[int]]:
    """Charge la liste des dossiers de bags et leurs labels depuis un CSV.

    Args:
        data_dir:   Répertoire racine contenant un sous-dossier par patient.
        labels_csv: CSV avec au minimum les colonnes ``id_col`` et ``label_col``.
        id_col:     Nom de la colonne identifiant (doit correspondre aux noms
                    de dossiers dans ``data_dir``).
        label_col:  Nom de la colonne label (valeurs entières ou factorisées).

    Returns:
        folders: Liste de chemins de dossiers existants.
        labels:  Liste d'entiers correspondants.

    Raises:
        FileNotFoundError: Si ``data_dir`` ou ``labels_csv`` est introuvable.
        ValueError: Si aucun dossier valide n'est trouvé après filtrage.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data_dir introuvable : {data_dir}")
    if not Path(labels_csv).is_file():
        raise FileNotFoundError(f"labels_csv introuvable : {labels_csv}")

    df = pd.read_csv(labels_csv)
    df[id_col] = df[id_col].astype(str).str.replace(r"\.[^.]+$", "", regex=True)
    df = df.dropna(subset=[label_col])

    # Factorisation si les labels ne sont pas déjà entiers
    if not pd.api.types.is_integer_dtype(df[label_col]):
        df[label_col] = pd.factorize(df[label_col])[0]

    folders: list[Path] = []
    labels: list[int] = []
    missing: list[str] = []

    for _, row in df.iterrows():
        folder = data_dir / str(row[id_col])
        if folder.is_dir():
            folders.append(folder)
            labels.append(int(row[label_col]))
        else:
            missing.append(str(row[id_col]))

    if missing:
        print(f"[WARN] {len(missing)} patient(s) sans dossier ignorés : {missing[:5]}{'…' if len(missing) > 5 else ''}")

    if not folders:
        raise ValueError(
            f"Aucun dossier valide trouvé dans {data_dir} pour les patients du CSV."
        )

    return folders, labels


def stratified_split(
    folders: list[Path],
    labels: list[int],
    val_split: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Split stratifié : chaque classe est proportionnellement représentée.

    Returns:
        train_indices, val_indices
    """
    rng = np.random.default_rng(seed)
    class_indices: dict[int, list[int]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        class_indices[lbl].append(i)

    train_idx, val_idx = [], []
    for _, idxs in sorted(class_indices.items()):
        arr = np.array(idxs)
        rng.shuffle(arr)
        n_val = max(1, int(len(arr) * val_split))
        val_idx.extend(arr[:n_val].tolist())
        train_idx.extend(arr[n_val:].tolist())

    return train_idx, val_idx


def build_dataloaders(
    folders: list[Path],
    labels: list[int],
    val_split: float,
    batch_size: int,
    seed: int,
    use_coords: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Construit les DataLoaders train et val avec split stratifié."""
    train_idx, val_idx = stratified_split(folders, labels, val_split, seed)

    train_folders = [folders[i] for i in train_idx]
    train_labels  = [labels[i]  for i in train_idx]
    val_folders   = [folders[i] for i in val_idx]
    val_labels    = [labels[i]  for i in val_idx]

    train_ds = EmbeddingBagDataset(train_folders, train_labels, use_coords=use_coords)
    val_ds   = EmbeddingBagDataset(val_folders,   val_labels,   use_coords=use_coords)

    # WeightedRandomSampler pour corriger le déséquilibre de classes
    train_arr     = np.array(train_labels)
    class_counts  = np.bincount(train_arr)
    sample_weights = (1.0 / class_counts[train_arr]).astype(np.float32)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_bags,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_bags,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    label_arr = np.array(labels)
    print(
        f"Dataset — total: {len(folders)} | "
        f"train: {len(train_ds)} | val: {len(val_ds)} | "
        f"classes: {dict(zip(*np.unique(label_arr, return_counts=True)))}"
    )
    return train_loader, val_loader


# ======================================================================
# Modèle
# ======================================================================


def build_model(
    in_dim: int,
    n_classes: int,
    lr: float,
    epochs: int,
    device: torch.device,
) -> tuple[ABMIL, torch.optim.Adam, torch.optim.lr_scheduler.CosineAnnealingLR]:
    """Instancie ABMIL, remplace la tête pour n_classes sorties.

    Args:
        in_dim:    Dimension des features d'entrée.
        n_classes: Nombre de classes (≥ 2).
        lr:        Learning rate pour Adam.
        epochs:    Nombre total d'époques (pour CosineAnnealing T_max).
        device:    Device cible.

    Returns:
        model, optimizer, scheduler
    """
    criterion = nn.CrossEntropyLoss()
    model = ABMIL(in_shape=(in_dim,), criterion=criterion)

    # Initialisation des couches lazy via un forward factice
    dummy = torch.zeros(1, 1, in_dim)
    with torch.no_grad():
        model(dummy)

    # Remplacement de la tête binaire (1 sortie) → n_classes sorties
    in_f: int = model.classifier.module.in_features
    model.classifier = nn.Linear(in_f, n_classes)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    return model, optimizer, scheduler


# ======================================================================
# Entraînement
# ======================================================================


def run_epoch(
    model: ABMIL,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool = True,
) -> dict[str, float]:
    """Exécute une époque complète et retourne les métriques.

    Returns:
        dict avec les clés : loss, acc, auc, ap
    """
    model.train() if train else model.eval()

    total_loss = 0.0
    n_batches = 0
    all_preds: list[int]   = []
    all_probs: list[list]  = []
    all_targets: list[int] = []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for bags, labels, mask in loader:
            bags   = bags.to(device)
            labels = labels.to(device)
            mask   = mask.to(device)

            if train:
                optimizer.zero_grad()

            # ABMIL avec n_classes sorties → logits shape (B, n_classes)
            logits = model(bags, mask)  # (B, n_classes)

            loss = model.criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            probs = torch.softmax(logits.detach(), dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1).tolist()
            tgts  = labels.detach().cpu().numpy().tolist()

            all_probs.extend(probs.tolist())
            all_preds.extend(preds)
            all_targets.extend(tgts)
            total_loss += loss.item()
            n_batches  += 1

    y      = np.array(all_targets)
    prob   = np.array(all_probs)
    pred   = np.array(all_preds)
    loss_  = total_loss / max(n_batches, 1)
    acc    = float((pred == y).mean())

    try:
        n_cls = prob.shape[1]
        auc = roc_auc_score(y, prob, multi_class="ovr") if n_cls > 2 else roc_auc_score(y, prob[:, 1])
        ap  = average_precision_score(y, prob[:, 1]) if n_cls == 2 else float("nan")
    except ValueError:
        auc = ap = float("nan")

    return dict(loss=loss_, acc=acc, auc=auc, ap=ap)


def train(
    model: ABMIL,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: torch.device,
    output_dir: Path,
) -> tuple[dict, int, float]:
    """Boucle d'entraînement principale.

    Sauvegarde le meilleur modèle (val AUC) dans ``output_dir/best_model.pth``.

    Returns:
        history:      dict de listes de métriques par époque.
        best_epoch:   numéro de l'époque avec le meilleur val AUC.
        best_val_auc: valeur du meilleur val AUC.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    history: dict[str, list[float]] = {
        k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc", "train_auc", "val_auc"]
    }
    best_val_auc = -1.0
    best_epoch   = 0

    for epoch in range(1, epochs + 1):
        train_m = run_epoch(model, train_loader, optimizer, device, train=True)
        val_m   = run_epoch(model, val_loader,   optimizer, device, train=False)
        scheduler.step()

        for phase, m in [("train", train_m), ("val", val_m)]:
            for key in ["loss", "acc", "auc"]:
                history[f"{phase}_{key}"].append(m[key])

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train — loss: {train_m['loss']:.4f}  acc: {train_m['acc']:.3f}  AUC: {train_m['auc']:.3f}  AP: {train_m['ap']:.3f} | "
            f"Val   — loss: {val_m['loss']:.4f}  acc: {val_m['acc']:.3f}  AUC: {val_m['auc']:.3f}  AP: {val_m['ap']:.3f}"
        )

        if val_m["auc"] > best_val_auc:
            best_val_auc = val_m["auc"]
            best_epoch   = epoch
            torch.save(model.state_dict(), output_dir / "best_model.pth")

    print(f"\nMeilleur modèle — epoch {best_epoch}, val AUC: {best_val_auc:.4f}")
    print(f"Modèle sauvegardé → {output_dir / 'best_model.pth'}")
    return history, best_epoch, best_val_auc


# ======================================================================
# Configuration — modifier ces variables avant de lancer le script
# ======================================================================

DATA_DIR    = "data/feather/BCL2/features_conch_v15"  # dossier racine (un sous-dossier par bag)
LABELS_CSV  = "csv/BCL2_feather.csv"                  # CSV avec colonnes ID_COL et LABEL_COL
ID_COL      = "patient_id"                            # colonne identifiant dans LABELS_CSV
LABEL_COL   = "Status"                                # colonne label dans LABELS_CSV

IN_DIM      = 768    # dimension des embeddings (D)
N_CLASSES   = 2      # nombre de classes
USE_COORDS  = False  # True pour inclure x,y dans les features

EPOCHS      = 60
BATCH_SIZE  = 4
LR          = 1e-4
VAL_SPLIT   = 0.2
SEED        = 42

OUTPUT_DIR  = "outputs/abmil"  # dossier de sortie pour best_model.pth

# ======================================================================
# Point d'entrée
# ======================================================================

if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else " [CPU]"))

    # ---- Données ----
    folders, labels = load_bag_data(DATA_DIR, LABELS_CSV, ID_COL, LABEL_COL)
    train_loader, val_loader = build_dataloaders(
        folders, labels,
        val_split=VAL_SPLIT,
        batch_size=BATCH_SIZE,
        seed=SEED,
        use_coords=USE_COORDS,
    )

    # ---- Modèle ----
    model, optimizer, scheduler = build_model(
        in_dim=IN_DIM,
        n_classes=N_CLASSES,
        lr=LR,
        epochs=EPOCHS,
        device=device,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ABMIL — in_dim: {IN_DIM}, n_classes: {N_CLASSES}, params: {n_params:,}")

    # ---- Entraînement ----
    history, best_epoch, best_val_auc = train(
        model, optimizer, scheduler,
        train_loader, val_loader,
        epochs=EPOCHS,
        device=device,
        output_dir=Path(OUTPUT_DIR),
    )
