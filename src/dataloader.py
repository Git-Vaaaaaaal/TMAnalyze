from trident import Processor
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchmil.models import ABMIL, DSMIL, TransMIL
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

    # Récupère les labels pour le split stratifié
    labels = np.array([int(l.flat[0]) for l in dataset.get_bag_labels()])

    # Split stratifié : chaque classe est proportionnellement représentée dans train et val
    rng = np.random.default_rng(cfg["seed"])
    class_indices = defaultdict(list)
    for i, label in enumerate(labels):
        class_indices[label].append(i)

    train_indices, val_indices = [], []
    for _, idxs in class_indices.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * cfg["val_split"]))
        val_indices.extend(idxs[:n_val].tolist())
        train_indices.extend(idxs[n_val:].tolist())

    train_ds = dataset.subset(train_indices)
    val_ds   = dataset.subset(val_indices)

    # WeightedRandomSampler : surech antillonne la classe minoritaire à l'entraînement
    train_labels  = labels[train_indices]
    class_counts  = np.bincount(train_labels)
    sample_weights = 1.0 / class_counts[train_labels]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,   collate_fn=collate_fn)

    print(f"Dataset — train: {len(train_ds)} (cls0: {(train_labels==0).sum()}, cls1: {(train_labels==1).sum()}) | val: {len(val_ds)}")
    return train_loader, val_loader


def build_model(cfg):
    if cfg["type_class"] == "binary" :
        if cfg["model_mil"] == "transmil":
            model     = TransMIL(in_shape=cfg["in_shape"], criterion=torch.nn.BCEWithLogitsLoss())
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
            return model.to(cfg["device"]), optimizer, scheduler
        elif cfg["model_mil"] == "abmil":
            model     = ABMIL(in_shape=cfg["in_shape"], criterion=torch.nn.BCEWithLogitsLoss())
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
            return model.to(cfg["device"]), optimizer, scheduler
        elif cfg["model_mil"] == "dsmil":
            model     = DSMIL(in_shape=cfg["in_shape"], criterion=torch.nn.BCEWithLogitsLoss())
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
            return model.to(cfg["device"]), optimizer, scheduler
    if cfg["type_class"] == "multi_class" :
        if cfg["model_mil"] == "transmil":
            model     = TransMIL(in_shape=cfg["in_shape"], criterion=torch.nn.CrossEntropyLoss())
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
            return model.to(cfg["device"]), optimizer, scheduler
        elif cfg["model_mil"] == "abmil":
            model     = ABMIL(in_shape=cfg["in_shape"], criterion=torch.nn.CrossEntropyLoss())
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
            return model.to(cfg["device"]), optimizer, scheduler
        elif cfg["model_mil"] == "dsmil":
            model     = DSMIL(in_shape=cfg["in_shape"], criterion=torch.nn.CrossEntropyLoss())
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
            return model.to(cfg["device"]), optimizer, scheduler



