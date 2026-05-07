import os
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchmil.data import collate_fn

from src.class_mil import CSVMILDataset
from src.training_mil import run_epoch, plot_dashboard
from src.dataloader import build_dataloaders, build_model
from src.evaluation import evaluate, generate_heatmaps


new_MUM1_csv = os.path.join("csv", "MUM1_only_id_labels.csv")
CFG = dict(
    slide_features_csv = r"rebuilt.prism\prism\MUM1\slide_encoder_MUM1.csv",
    tiles_dir          = r"rebuilt.prism\prism\MUM1\job_dir\20.0x_64px_0px_overlap\features_virchow",
    labels_csv         = new_MUM1_csv,
    bag_keys           = ["X", "X_slide", "Y", "coords"],
    model_mil          = "transmil",  # abmil, dsmil
    in_shape           = (2560,),
    lr                 = 1e-4,
    epochs             = 30,
    batch_size         = 4,
    val_split          = 0.2,
    device             = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir         = "outputs/",
    seed               = 42,
)


df = pd.read_csv("csv/MUM1_labels.csv")
df = df.drop(columns=["stain", "old_patient_id", "null"])
print(df.head(5))
df.to_csv(new_MUM1_csv, index=False)


def train(cfg, model, optimizer, scheduler, train_loader, val_loader):
    history      = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc", "train_auc", "val_auc", "train_ap", "val_ap"]}
    best_val_auc = -1.0
    best_epoch   = 0

    for epoch in range(1, cfg["epochs"] + 1):
        train_metrics, _ = run_epoch(model, train_loader, optimizer, cfg["device"], train=True)
        val_metrics, _   = run_epoch(model, val_loader,   optimizer, cfg["device"], train=False)
        scheduler.step()

        for phase, m in [("train", train_metrics), ("val", val_metrics)]:
            for key in ["loss", "acc", "auc", "ap"]:
                history[f"{phase}_{key}"].append(m[key])

        print(
            f"Epoch {epoch:03d}/{cfg['epochs']} | "
            f"Train — loss: {train_metrics['loss']:.4f}  acc: {train_metrics['acc']:.3f}  AUC: {train_metrics['auc']:.3f} | "
            f"Val   — loss: {val_metrics['loss']:.4f}  acc: {val_metrics['acc']:.3f}  AUC: {val_metrics['auc']:.3f}"
        )

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_epoch   = epoch
            torch.save(model.state_dict(), os.path.join(cfg["output_dir"], "best_model.pth"))

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(cfg["output_dir"], f"model_epoch{epoch:03d}.pth"))

    print(f"\nMeilleur modèle — epoch {best_epoch}, val AUC: {best_val_auc:.4f}")
    return history, best_epoch, best_val_auc




os.makedirs(CFG["output_dir"], exist_ok=True)
torch.manual_seed(CFG["seed"])

train_loader, val_loader          = build_dataloaders(CFG)
model, optimizer, scheduler       = build_model(CFG)
history, best_epoch, best_val_auc = train(CFG, model, optimizer, scheduler, train_loader, val_loader)
final_tracker                     = evaluate(CFG, model, val_loader, optimizer)

plot_dashboard(history, best_epoch, best_val_auc, final_tracker, CFG["output_dir"])
generate_heatmaps(CFG, model)

