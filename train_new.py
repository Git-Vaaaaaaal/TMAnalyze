import os
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchmil.data import collate_fn

from src.training_mil import run_epoch, plot_dashboard
from src.dataloader import build_dataloaders, build_model
from src.evaluation import evaluate, generate_heatmaps
from src.csv_status import cleaning_csv

from sklearn.model_selection import KFold  # ou LeaveOneOut

"""
dropped_column = ["patient_id", "old_patient_id", "stain", "Age", "Status", "ECOG PS", "LDH", "EN", "Stage", "IPI Score", "IPI Risk Group (4 Class)", "RIPI Risk Group",
                "OS", "PFS"]
"""

# --- Sharding multi-GPU -------------------------------------------------
# Chaque processus traite les combinaisons dont (index % NUM_SHARDS) == SHARD_ID.
# Lancé par le .slurm avec un process par GPU (CUDA_VISIBLE_DEVICES isolant 1 GPU).
SHARD_ID    = int(os.environ.get("SHARD_ID", "0"))
NUM_SHARDS  = int(os.environ.get("NUM_SHARDS", "1"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))


marker_list = ["HE", "BCL2", "BCL6", "CD10", "MUM1", "MYC"]

dataframe_id = os.path.join("csv", "multi_label_patient_id.csv")

encoder_list = ["gpfm", "virchow2", "openmidnight", "musk", "hibou_l"] #["prism", "titan", "feather"]

mil_list = ["dtfdmil", "patchgcn", "deepgraphsurv"] #"abmil", "dsmil", "clam",

status_list = ["IPI Score"]

ENCODER_CFG = {
    "prism":   dict(in_shape=2560, tiles_subdir="features_virchow",   slide_subdir="slide_features_prism",  slide_csv="prism_encoder.csv"),
    "titan":   dict(in_shape=768,  tiles_subdir="features_conch_v15", slide_subdir="slide_features_titan",  slide_csv="titan_encoder.csv"),
    "feather": dict(in_shape=768,  tiles_subdir="features_conch_v15", slide_subdir="slide_features_feather", slide_csv="feather_encoder.csv"),
    "gpfm": dict(in_shape=1024,  tiles_subdir="features_gpfm", slide_subdir="", slide_csv=""),
    "musk": dict(in_shape=1024,  tiles_subdir="features_musk", slide_subdir="", slide_csv=""),
    "openmidnight": dict(in_shape=1536,  tiles_subdir="features_openmidnight", slide_subdir="", slide_csv=""),
    "hibou_l": dict(in_shape=1024,  tiles_subdir="features_hibou_l", slide_subdir="", slide_csv=""),
    "virchow2": dict(in_shape=2560,  tiles_subdir="features_virchow2", slide_subdir="", slide_csv=""),
}


from itertools import product


def train(cfg, model, optimizer, scheduler, train_loader, val_loader, run_label, log_path):
    history      = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc", "train_auc", "val_auc", "train_ap", "val_ap"]}
    best_score   = -float("inf")
    best_epoch   = 0
    best_val_auc = -1.0

    with open(log_path, "w", buffering=1) as log:
        header = f"=== {run_label} ===\n"
        print(header, end="")
        log.write(header)

        for epoch in range(1, cfg["epochs"] + 1):
            train_metrics, _ = run_epoch(model, train_loader, optimizer, cfg["device"], train=True)
            val_metrics, _   = run_epoch(model, val_loader,   optimizer, cfg["device"], train=False)
            scheduler.step()

            for phase, m in [("train", train_metrics), ("val", val_metrics)]:
                for key in ["loss", "acc", "auc", "ap"]:
                    history[f"{phase}_{key}"].append(m[key])

            train_score = train_metrics["acc"] - train_metrics["loss"]

            line = (
                f"Epoch {epoch:03d}/{cfg['epochs']} | "
                f"Train — loss: {train_metrics['loss']:.4f}  acc: {train_metrics['acc']:.3f}  AUC: {train_metrics['auc']:.3f}  score: {train_score:.3f} | "
                f"Val   — loss: {val_metrics['loss']:.4f}  acc: {val_metrics['acc']:.3f}  AUC: {val_metrics['auc']:.3f}\n"
            )
            print(line, end="")
            log.write(line)

            if train_score > best_score:
                best_score   = train_score
                best_epoch   = epoch
                best_val_auc = val_metrics["auc"]

        summary = f"\nMeilleur epoch {best_epoch} (score={best_score:.3f}), val AUC: {best_val_auc:.4f}\n"
        print(summary, end="")
        log.write(summary)

    return history, best_epoch, best_val_auc


# ======================================================================
# Main — sharding
# ======================================================================

all_combinations = list(product(encoder_list, marker_list, mil_list, status_list))
my_combinations  = [(idx, combo) for idx, combo in enumerate(all_combinations) if idx % NUM_SHARDS == SHARD_ID]

print(f"[Shard {SHARD_ID}/{NUM_SHARDS}] {len(all_combinations)} combinaisons totales, "
      f"{len(my_combinations)} assignees a ce shard.")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"GPU : {torch.cuda.get_device_name(0)}")

for idx, (encoder, marker, mil, status) in my_combinations:
    enc       = ENCODER_CFG[encoder]
    run_dir   = os.path.join("output_ipi", encoder, marker, mil, status)
    run_label = f"{status} | {encoder} | {mil} | {marker}"
    os.makedirs(run_dir, exist_ok=True)

    out_csv_marker = cleaning_csv(dataframe_id, marker, encoder, status)
    n_classes  = int(pd.read_csv(out_csv_marker)["Status"].nunique())
    type_class = "binary" if n_classes == 2 else "multi_class"

    tiles_dir = os.path.join("data_224", encoder, marker, enc["tiles_subdir"])
    if not os.path.isdir(tiles_dir):
        print(f"[SKIP] tiles_dir introuvable : {tiles_dir}")
        continue

    slide_features_csv = None
    bag_keys           = ["X", "Y", "coords"]
    if enc["slide_csv"]:
        candidate = os.path.join("data_224", encoder, marker, enc["slide_subdir"], enc["slide_csv"])
        if os.path.isfile(candidate):
            slide_features_csv = candidate
            bag_keys           = ["X", "X_slide", "Y", "coords"]

    if mil in ("patchgcn", "deepgraphsurv"):
        bag_keys.append("adj")

    CFG = dict(
        slide_features_csv = slide_features_csv,
        slide_id_col       = "wsi_name",
        tiles_dir          = tiles_dir,
        labels_csv         = out_csv_marker,
        bag_keys           = bag_keys,
        model_mil          = mil,
        in_shape           = (enc["in_shape"],),
        lr                 = 1e-4,
        epochs             = 80,
        batch_size         = 256,
        val_split          = 0.3,
        device             = str(device),
        output_dir         = run_dir,
        seed               = 42,
        type_class         = type_class,
        n_classes          = n_classes,
    )

    print(f"\n{'='*60}")
    print(f"  [{idx+1}/{len(all_combinations)}] {run_label}")
    print(f"  Device : {device}  |  Output : {run_dir}")
    print(f"{'='*60}")

    torch.manual_seed(CFG["seed"])

    train_loader, val_loader          = build_dataloaders(CFG)
    model, optimizer, scheduler       = build_model(CFG)
    history, best_epoch, best_val_auc = train(
        CFG, model, optimizer, scheduler, train_loader, val_loader,
        run_label, os.path.join(run_dir, "training_log.txt"),
    )
    final_tracker = evaluate(CFG, model, val_loader, optimizer)

    plot_dashboard(history, best_epoch, final_tracker, os.path.join(run_dir, "dashboard.png"))
    generate_heatmaps(CFG, model)



