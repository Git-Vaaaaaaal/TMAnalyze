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


"""
dropped_column = ["patient_id", "old_patient_id", "stain", "Age", "Status", "ECOG PS", "LDH", "EN", "Stage", "IPI Score", "IPI Risk Group (4 Class)", "RIPI Risk Group",
                "OS", "PFS"]
"""


def cleaning_csv(df_path, marker, encoder, element):
    df_label = pd.read_csv(df_path)
    df_label = df_label[df_label["stain"] == marker]
    df_label = df_label[["patient_id", element]].rename(columns={element: "Status"})
    df_label["Status"] = df_label["Status"].astype(int)
    out_csv_marker = os.path.join("csv", f"{marker}_{encoder}.csv")
    df_label.to_csv(out_csv_marker, index=False)
    return out_csv_marker


marker_list = ["BCL2", "BCL6", "CD10", "HE", "MUM1", "MYC"]

dataframe_id = os.path.join("csv", "multi_label_patient_id.csv")

encoder_list = ["prism", "titan", "feather"]

mil_list = ["transmil", "abmil", "dsmil"]

status_list = ["status"]#["ECOG PS", "LDH", "EN", "Stage", "IPI Score", "IPI Risk Group (4 Class)", "RIPI Risk Group"]
binary_list = ["status", "LDH"]

ENCODER_CFG = {
    "prism":   dict(in_shape=2560, tiles_subdir="features_virchow",   slide_subdir="slide_features_prism",  slide_csv="prism_encoder.csv"),
    "titan":   dict(in_shape=768,  tiles_subdir="features_conch_v15", slide_subdir="slide_features_titan",  slide_csv="titan_encoder.csv"),
    "feather": dict(in_shape=768,  tiles_subdir="features_conch_v15", slide_subdir="slide_features_feather", slide_csv="feather_encoder.csv"),
}

for marker in marker_list :
    for encoder in encoder_list:
        for mil in mil_list :
            for status in status_list:
                enc = ENCODER_CFG[encoder]
                out_csv_marker = cleaning_csv(dataframe_id, marker, encoder, status)

                if binary_list.contains(status):
                    type_class = "binary"
                else : 
                    type_class = "multi_class"

                os.makedirs(os.path.join("outputs_" + str(status)), exist_ok=True)

                CFG = dict(
                    slide_features_csv = os.path.join("data", encoder, marker, enc["slide_subdir"], enc["slide_csv"]),
                    slide_id_col       = "wsi_name",
                    tiles_dir          = os.path.join("data", encoder, marker, enc["tiles_subdir"]),
                    labels_csv         = out_csv_marker,
                    bag_keys           = ["X", "X_slide", "Y", "coords"],
                    model_mil          = mil,
                    in_shape           = (enc["in_shape"],),
                    lr                 = 1e-4,
                    epochs             = 60,
                    batch_size         = 4,
                    val_split          = 0.2,
                    device             = "cuda" if torch.cuda.is_available() else "cpu",
                    output_dir         = os.path.join("outputs_" + str(status), encoder, marker, mil),
                    seed               = 42,
                    type_class         = type_class, # "binary" or "multi_class"
                )


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
                            # torch.save(model.state_dict(), os.path.join(cfg["output_dir"], "best_model.pth"))

                        # if epoch % 10 == 0:
                        #     torch.save(model.state_dict(), os.path.join(cfg["output_dir"], f"model_epoch{epoch:03d}.pth"))

                    msg = f"\n{status}, {encoder}, {mil}, {marker}, meilleur modèle — epoch {best_epoch}, val AUC: {best_val_auc:.4f}"
                    print(msg)
                    with open("output_logs.txt", "a") as f:
                        f.write(msg + "\n")
                    return history, best_epoch, best_val_auc




                torch.manual_seed(CFG["seed"])

                device = CFG["device"]
                print(f"\n{'='*60}")
                print(f"  {status} | {encoder} | {mil} | {marker}")
                print(f"  Device : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else " [WARNING: CUDA non disponible]"))
                print(f"{'='*60}")

                train_loader, val_loader          = build_dataloaders(CFG)
                model, optimizer, scheduler       = build_model(CFG)
                history, best_epoch, best_val_auc = train(CFG, model, optimizer, scheduler, train_loader, val_loader)
                final_tracker                     = evaluate(CFG, model, val_loader, optimizer)

                save_path = os.path.join("output", f"{marker}_{encoder}_{mil}.png")
                plot_dashboard(history, best_epoch, best_val_auc, final_tracker, save_path)
                #generate_heatmaps(CFG, model)



