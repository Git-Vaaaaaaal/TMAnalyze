"""
train_concatenation.py — Entraînement MIL (ABMIL / TransMIL / DSMIL)
sur des embeddings chargés depuis des dossiers de CSV (tile-level).

Structure attendue :
    data/<encoder>/<marker>/<tiles_subdir>/
        <patient_id>/        ← un dossier par bag
            patch_0.csv
            patch_1.csv      ← optionnel, plusieurs CSV concaténés
"""

from __future__ import annotations

import inspect
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchmil.models import ABMIL, DSMIL, TransMIL

from src.embedding_concatenation import EmbeddingBagDataset, collate_bags
from src.dataloader import _replace_head
from src.training_mil import plot_dashboard


# ======================================================================
# Config — même structure que train.py
# ======================================================================

marker_list  = ["HE"]#["BCL2", "BCL6", "CD10", "HE", "MUM1", "MYC"]
encoder_list = ["prism", "titan", "feather"]
mil_list     = ["transmil", "abmil", "dsmil"]
status_list  = ["LDH", "status"]#["status", "LDH", "Stage", "IPI Score",  "RIPI Risk Group"]

ENCODER_CFG = {
    "prism":   dict(in_shape=2560, tiles_subdir="features_virchow"),
    "titan":   dict(in_shape=768,  tiles_subdir="features_conch_v15"),
    "feather": dict(in_shape=768,  tiles_subdir="features_conch_v15"),
}

dataframe_id = "common_patients_labels.csv"

EPOCHS     = 60
BATCH_SIZE = 4
LR         = 1e-4
VAL_SPLIT  = 0.3
SEED       = 42
ID_COL     = "patient_id"

os.makedirs("output_concat_HE", exist_ok=True)

# ======================================================================
# CSV helpers
# ======================================================================

def build_label_dict(df_path: str, element: str) -> dict[str, int]:
    """Construit label_dict {old_patient_id: label} depuis le CSV commun.

    Une seule ligne par patient est conservee (dedupliquee sur old_patient_id).
    Les valeurs du label sont encodees en entiers via factorize.
    """
    df = pd.read_csv(df_path)
    df = df[["old_patient_id", element]].drop_duplicates(subset=["old_patient_id"])
    df = df.dropna(subset=[element])
    df[element] = pd.factorize(df[element])[0]
    return dict(zip(df["old_patient_id"].astype(str), df[element].astype(int)))


def build_bag_dict_from_csv(
    csv_path: str | Path,
    encoder:  str,
    base_dir: str = "data",
) -> dict[str, list[Path]]:
    """Construit bag_dict a partir d'un CSV de mapping patients/images.

    Colonnes attendues dans le CSV :
        old_patient_id  -- identifiant unique du patient (cle du dict)
        marqueur        -- marker de l'image (BCL2, BCL6, ...)
        patient_id      -- nom du fichier image (sans ou avec .csv)

    Path reconstruit :
        {base_dir}/{encoder}/{marqueur}/{tiles_subdir}/{patient_id}.csv
    """
    tiles_subdir = ENCODER_CFG[encoder]["tiles_subdir"]
    df = pd.read_csv(csv_path)
    df["patient_id"] = df["patient_id"].astype(str).str.replace(r"\.csv$", "", regex=True)

    bag_dict: dict[str, list[Path]] = defaultdict(list)
    missing: list[str] = []

    for _, row in df.iterrows():
        pid    = str(row["old_patient_id"])
        marker = str(row["stain"])
        img_id = str(row["patient_id"])
        path   = Path(base_dir) / encoder / marker / tiles_subdir / f"{img_id}.csv"
        if path.exists():
            bag_dict[pid].append(path)
        else:
            missing.append(str(path))

    if missing:
        n = len(missing)
        print(f"[WARN] {n} fichier(s) introuvable(s) : {missing[:5]}{'...' if n > 5 else ''}")

    return {pid: sorted(paths) for pid, paths in bag_dict.items()}


def build_bag_dict(
    data_dir:   str | Path,
    labels_csv: str | Path,
    id_col:     str = "patient_id",
    label_col:  str = "Status",
) -> tuple[dict[str, list[Path]], dict[str, int]]:
    """Construit bag_dict et label_dict pour EmbeddingBagDataset.

    Chaque patient -> liste de tous les CSV de son dossier (triés).
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data_dir introuvable : {data_dir}")

    df = pd.read_csv(labels_csv)
    df[id_col] = df[id_col].astype(str).str.replace(r"\.[^.]+$", "", regex=True)
    df = df.dropna(subset=[label_col])

    bag_dict:   dict[str, list[Path]] = {}
    label_dict: dict[str, int]        = {}
    missing: list[str] = []

    for _, row in df.iterrows():
        pid    = str(row[id_col])
        folder = data_dir / pid
        if folder.is_dir():
            csv_files = sorted(folder.glob("*.csv"))
            if csv_files:
                bag_dict[pid]   = csv_files
                label_dict[pid] = int(row[label_col])
            else:
                missing.append(pid)
        else:
            missing.append(pid)

    if missing:
        print(f"[WARN] {len(missing)} patient(s) ignores : {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if not bag_dict:
        raise ValueError(f"Aucun dossier valide dans {data_dir}")

    return bag_dict, label_dict


# ======================================================================
# Données
# ======================================================================

def build_dataloaders(
    bag_dict:   dict[str, list[Path]],
    label_dict: dict[str, int],
    val_split:  float,
    batch_size: int,
    seed:       int,
) -> tuple[DataLoader, DataLoader]:
    patient_ids = sorted(set(bag_dict.keys()) & set(label_dict.keys()))
    labels      = [label_dict[pid] for pid in patient_ids]

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

    train_pids = [patient_ids[i] for i in train_idx]
    val_pids   = [patient_ids[i] for i in val_idx]

    train_bag   = {pid: bag_dict[pid]   for pid in train_pids}
    train_label = {pid: label_dict[pid] for pid in train_pids}
    val_bag     = {pid: bag_dict[pid]   for pid in val_pids}
    val_label   = {pid: label_dict[pid] for pid in val_pids}

    train_ds = EmbeddingBagDataset(train_bag, train_label)
    val_ds   = EmbeddingBagDataset(val_bag,   val_label)

    train_labels_arr = np.array([label_dict[pid] for pid in train_pids])
    class_counts     = np.bincount(train_labels_arr)
    sample_weights   = (1.0 / class_counts[train_labels_arr]).astype(np.float32)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              collate_fn=collate_bags, num_workers=0,
                              pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_bags, num_workers=0,
                              pin_memory=torch.cuda.is_available())

    all_labels = np.array(labels)
    classes, counts = np.unique(all_labels, return_counts=True)
    print(f"Dataset -- total: {len(patient_ids)} | train: {len(train_ds)} | val: {len(val_ds)} | "
          f"classes: {dict(zip(classes.tolist(), counts.tolist()))}")
    return train_loader, val_loader


# ======================================================================
# Modèle
# ======================================================================

def build_model(
    mil_name:  str,
    in_dim:    int,
    n_classes: int,
    lr:        float,
    epochs:    int,
    device:    torch.device,
):
    is_multi  = n_classes > 2
    criterion = nn.CrossEntropyLoss() if is_multi else nn.BCEWithLogitsLoss()
    in_shape  = (in_dim,)

    model_cls = {"transmil": TransMIL, "abmil": ABMIL, "dsmil": DSMIL}[mil_name]
    model = model_cls(in_shape=in_shape, criterion=criterion)

    if is_multi:
        dummy = torch.zeros(1, 1, in_dim)
        with torch.no_grad():
            model(dummy)
        model = _replace_head(model, mil_name, n_classes)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    return model.to(device), optimizer, scheduler


# ======================================================================
# Epoch -- gère mask optionnel + binaire/multi-class + bags 4D -> 3D
# ======================================================================

class _Tracker:
    def __init__(self):
        self.targets: list = []
        self.probs:   list = []
        self.preds:   list = []


def run_epoch(
    model,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    n_classes: int,
    train:     bool = True,
) -> tuple[dict[str, float], _Tracker]:
    model.train() if train else model.eval()

    _has_mask     = "mask" in inspect.signature(model.forward).parameters
    is_multiclass = n_classes > 2
    total_loss    = 0.0
    n_batches     = 0
    tracker       = _Tracker()

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for bags, labels, mask in loader:
            # bags: (B, K, N, D) -> flatten K*N patches pour les modeles MIL 1D
            B, K, N, D = bags.shape
            bags_3d = bags.view(B, K * N, D).to(device)
            labels  = labels.to(device)

            if train:
                optimizer.zero_grad()

            if _has_mask:
                mask_2d = mask.view(B, K * N).to(device)
                out = model(bags_3d, mask_2d)
            else:
                out = model(bags_3d)

            logits = out if isinstance(out, torch.Tensor) else out[0]

            # DSMIL multi-class: bag_classifier produit [B, 1, n_classes]
            if logits.dim() == 3 and logits.shape[1] == 1:
                logits = logits.squeeze(1)

            if is_multiclass:
                loss  = model.criterion(logits, labels.long().flatten())
                probs = torch.softmax(logits.detach(), dim=-1).cpu().numpy()
                preds = probs.argmax(axis=-1).tolist()
                probs = probs.tolist()
            else:
                loss  = model.criterion(logits.flatten(), labels.float().flatten())
                probs = torch.sigmoid(logits.detach()).cpu().numpy().ravel()
                preds = (probs >= 0.5).astype(int).tolist()
                probs = probs.tolist()

            if train:
                loss.backward()
                optimizer.step()

            tgts = labels.detach().cpu().numpy().ravel().astype(int).tolist()
            tracker.targets.extend(tgts)
            tracker.probs.extend(probs)
            tracker.preds.extend(preds)
            total_loss += loss.item()
            n_batches  += 1

    y     = np.array(tracker.targets)
    prob  = np.array(tracker.probs)
    pred  = np.array(tracker.preds)
    loss_ = total_loss / max(n_batches, 1)
    acc   = float((pred == y).mean())

    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        if prob.ndim == 2:
            auc = roc_auc_score(y, prob, multi_class="ovr")
            ap  = float("nan")
        else:
            auc = roc_auc_score(y, prob)
            ap  = average_precision_score(y, prob)
    except ValueError:
        auc = ap = float("nan")

    return dict(loss=loss_, acc=acc, auc=auc, ap=ap), tracker


# ======================================================================
# Boucle d'entraînement
# ======================================================================

def train_loop(
    model,
    optimizer,
    scheduler,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    epochs:       int,
    device:       torch.device,
    n_classes:    int,
    run_label:    str,
) -> tuple[dict, int, float, _Tracker]:
    history = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc",
                                "train_auc", "val_auc", "train_ap", "val_ap"]}
    best_val_auc  = -1.0
    best_epoch    = 0
    final_tracker = None

    for epoch in range(1, epochs + 1):
        train_m, _      = run_epoch(model, train_loader, optimizer, device, n_classes, train=True)
        val_m, tracker  = run_epoch(model, val_loader,   optimizer, device, n_classes, train=False)
        scheduler.step()

        for phase, m in [("train", train_m), ("val", val_m)]:
            for key in ["loss", "acc", "auc", "ap"]:
                history[f"{phase}_{key}"].append(m[key])

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train -- loss: {train_m['loss']:.4f}  acc: {train_m['acc']:.3f}  AUC: {train_m['auc']:.3f} | "
            f"Val   -- loss: {val_m['loss']:.4f}  acc: {val_m['acc']:.3f}  AUC: {val_m['auc']:.3f}"
        )

        if val_m["auc"] > best_val_auc:
            best_val_auc  = val_m["auc"]
            best_epoch    = epoch
            final_tracker = tracker

    msg = f"\n{run_label} -- meilleur epoch {best_epoch}, val AUC: {best_val_auc:.4f}"
    print(msg)
    with open(os.path.join("output_concat_HE", "output_logs_concat.txt"), "a") as f:
        f.write(msg + "\n")

    return history, best_epoch, best_val_auc, final_tracker


# ======================================================================
# Main
# ======================================================================

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for encoder in encoder_list:
    bag_dict = build_bag_dict_from_csv(dataframe_id, encoder)

    for mil in mil_list:
        for status in status_list:
            label_dict = build_label_dict(dataframe_id, status)
            n_classes  = len(set(label_dict.values()))
            in_dim     = ENCODER_CFG[encoder]["in_shape"]
            run_label  = f"{status} | {encoder} | {mil}"

            print(f"\n{'='*62}")
            print(f"  {run_label}")
            print(f"  Device : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else " [CPU]"))
            print(f"  in_dim={in_dim}  n_classes={n_classes}  patients={len(bag_dict)}")
            print(f"{'='*62}")

            if n_classes < 2:
                print(f"[SKIP] Une seule classe presente")
                continue

            train_loader, val_loader = build_dataloaders(
                bag_dict, label_dict, val_split=VAL_SPLIT,
                batch_size=BATCH_SIZE, seed=SEED,
            )

            model, optimizer, scheduler = build_model(
                mil_name=mil, in_dim=in_dim, n_classes=n_classes,
                lr=LR, epochs=EPOCHS, device=device,
            )

            history, best_epoch, best_val_auc, final_tracker = train_loop(
                model, optimizer, scheduler,
                train_loader, val_loader,
                epochs=EPOCHS, device=device, n_classes=n_classes,
                run_label=run_label,
            )

            save_path = os.path.join("output_concat_HE", f"{encoder}_{mil}_{status}.png")
            os.makedirs("output_concat", exist_ok=True)
            if final_tracker is not None:
                plot_dashboard(history, best_epoch, final_tracker, save_path)
