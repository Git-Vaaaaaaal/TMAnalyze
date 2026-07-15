from trident import Processor
import os
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchmil.models import ABMIL, DSMIL, TransMIL, CLAM_SB, DTFDMIL, PatchGCN, DeepGraphSurv
from torchmil.data import collate_fn
from src.class_mil import CSVMILDataset
from src.training_mil import run_epoch




def initialize_processor():
    return Processor(
        job_dir=JOB_DIR,
        wsi_source=WSI_DIR,
        wsi_ext=WSI_EXT,
        wsi_cache=WSI_CACHE,
        skip_errors=SKIP_ERRORS,
        custom_mpp_keys=CUSTOM_MPP_KEYS,
        custom_list_of_wsis=CUSTOM_LIST_OF_WSIS,
        max_workers=MAX_WORKERS,
        reader_type=READER_TYPE,
        search_nested=SEARCH_NESTED,
    )


def build_dataloaders(cfg):
    dataset = CSVMILDataset(
        slide_features_csv=cfg["slide_features_csv"],
        tiles_dir=cfg["tiles_dir"],
        labels_csv=cfg["labels_csv"],
        bag_keys=cfg["bag_keys"],
        slide_id_col=cfg.get("slide_id_col", "patient_id"),
    )

    labels  = np.array([int(l.flat[0]) for l in dataset.get_bag_labels()])
    wsi_ids = dataset.get_bag_names()   # ordre identique à labels[]
    rng     = np.random.default_rng(cfg["seed"])
    test_split = cfg.get("test_split", 0.0)

    # Lecture du CSV labels pour récupérer old_patient_id
    labels_df = pd.read_csv(cfg["labels_csv"])
    labels_df["patient_id"] = labels_df["patient_id"].astype(str).apply(
        lambda x: os.path.splitext(x)[0]
    )

    if "old_patient_id" in labels_df.columns:
        # ── Split au niveau du patient réel ──────────────────────────────
        wsi_to_patient = dict(zip(
            labels_df["patient_id"],
            labels_df["old_patient_id"].astype(str),
        ))

        # Groupe les indices WSI par patient réel
        patient_to_idxs = defaultdict(list)
        for i, wsi_id in enumerate(wsi_ids):
            pid = wsi_to_patient.get(wsi_id, wsi_id)
            patient_to_idxs[pid].append(i)

        # Label du patient = vote majoritaire de ses images
        patient_by_class = defaultdict(list)
        for pid, idxs in patient_to_idxs.items():
            patient_label = int(np.bincount(labels[idxs]).argmax())
            patient_by_class[patient_label].append(pid)

        train_patients, val_patients, test_patients = [], [], []
        for _, patients in patient_by_class.items():
            patients = np.array(patients)
            rng.shuffle(patients)
            n_test = max(1, int(len(patients) * test_split)) if test_split > 0 else 0
            test_patients.extend(patients[:n_test].tolist())
            remaining = patients[n_test:]
            n_val = max(1, int(len(remaining) * cfg["val_split"]))
            val_patients.extend(remaining[:n_val].tolist())
            train_patients.extend(remaining[n_val:].tolist())

        train_indices = [i for p in train_patients for i in patient_to_idxs[p]]
        val_indices   = [i for p in val_patients   for i in patient_to_idxs[p]]
        test_indices  = [i for p in test_patients  for i in patient_to_idxs[p]]

        print(f"Split patient-level — train: {len(train_patients)} | val: {len(val_patients)} "
              f"| test: {len(test_patients)} patients ({len(patient_to_idxs)} total)")
    else:
        # ── Fallback : split au niveau image (comportement original) ─────
        class_indices = defaultdict(list)
        for i, label in enumerate(labels):
            class_indices[label].append(i)

        train_indices, val_indices, test_indices = [], [], []
        for _, idxs in class_indices.items():
            idxs = np.array(idxs)
            rng.shuffle(idxs)
            n_test = max(1, int(len(idxs) * test_split)) if test_split > 0 else 0
            test_indices.extend(idxs[:n_test].tolist())
            remaining = idxs[n_test:]
            n_val = max(1, int(len(remaining) * cfg["val_split"]))
            val_indices.extend(remaining[:n_val].tolist())
            train_indices.extend(remaining[n_val:].tolist())

    train_ds = dataset.subset(train_indices)
    val_ds   = dataset.subset(val_indices)
    test_ds  = dataset.subset(test_indices)

    # WeightedRandomSampler : suréchantillonne la classe minoritaire
    train_labels   = labels[train_indices]
    class_counts   = np.bincount(train_labels)
    sample_weights = 1.0 / class_counts[train_labels]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,   collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False,   collate_fn=collate_fn)

    print(f"WSI — train: {len(train_ds)} (cls0: {(train_labels==0).sum()}, cls1: {(train_labels==1).sum()}) "
          f"| val: {len(val_ds)} | test: {len(test_ds)}")
    return train_loader, val_loader, test_loader


def _replace_head(model, mil_name, n_classes):
    """Replace the 1-output classifier with n_classes outputs for multi-class."""
    if mil_name == "abmil":
        in_f = model.classifier.module.in_features
        model.classifier = torch.nn.Linear(in_f, n_classes)
    elif mil_name == "transmil":
        in_f = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_f, n_classes)
    elif mil_name == "dsmil":
        in_f = model.bag_classifier.in_features
        model.bag_classifier = torch.nn.Linear(in_f, n_classes)
        # inst_classifier kept at 1 output: used for critical-instance selection, must stay binary
    return model


def build_model(cfg):
    is_multi  = cfg["type_class"] == "multi_class"
    n_classes = cfg.get("n_classes", 2)
    criterion = torch.nn.CrossEntropyLoss() if is_multi else torch.nn.BCEWithLogitsLoss()
    mil       = cfg["model_mil"]

    model_cls = {"transmil": TransMIL, "abmil": ABMIL, "dsmil": DSMIL, "clam": CLAM_SB, "dtfdmil": DTFDMIL, "patchgcn": PatchGCN, "deepgraphsurv": DeepGraphSurv}[mil]
    model = model_cls(in_shape=cfg["in_shape"], criterion=criterion)

    if is_multi:
        # Initialize lazy layers with dummy forward pass, then swap head to n_classes outputs
        dummy = torch.zeros(1, 1, cfg["in_shape"][0])
        with torch.no_grad():
            model(dummy)
        model = _replace_head(model, mil, n_classes)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-6)
    return model.to(cfg["device"]), optimizer, scheduler



