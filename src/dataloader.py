from trident import Processor
import torch
from torch.utils.data import DataLoader, random_split
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
    )
    n_val   = int(len(dataset) * cfg["val_split"])
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate_fn)
    print(f"Dataset — train: {n_train} | val: {n_val}")
    return train_loader, val_loader


def build_model(cfg):
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



