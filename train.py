import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchmil.models import ABMIL, DSMIL, TransMIL
from torchmil.data import collate_fn
from src.class_mil import CSVMILDataset
from src.training_mil import run_epoch, plot_dashboard
import pandas as pd

new_MUM1_csv = os.path.join("csv", "MUM1_only_id_labels.csv")
CFG = dict(
    slide_features_csv = r"rebuilt.prism\prism\MUM1\slide_encoder_MUM1.csv",
    tiles_dir          = r"rebuilt.prism\prism\MUM1\job_dir\20.0x_64px_0px_overlap\features_virchow",
    labels_csv         = new_MUM1_csv,
    bag_keys           = ["X", "X_slide", "Y", "coords"],
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
    model     = TransMIL(in_shape=cfg["in_shape"], criterion=torch.nn.BCEWithLogitsLoss())
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    return model.to(cfg["device"]), optimizer, scheduler


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


def evaluate(cfg, model, val_loader, optimizer):
    model.load_state_dict(torch.load(os.path.join(cfg["output_dir"], "best_model.pth"), map_location=cfg["device"]))
    final_metrics, final_tracker = run_epoch(model, val_loader, optimizer, cfg["device"], train=False)

    print("\n" + "=" * 60)
    print("RÉSUMÉ FINAL (meilleur modèle, val set)")
    print("=" * 60)
    print(f"  Loss      : {final_metrics['loss']:.4f}")
    print(f"  Accuracy  : {final_metrics['acc']:.4f}")
    print(f"  AUC-ROC   : {final_metrics['auc']:.4f}")
    print(f"  Avg Prec  : {final_metrics['ap']:.4f}")
    print("=" * 60)
    return final_tracker


def generate_heatmaps(cfg, model):
    out_dir = os.path.join(cfg["output_dir"], "heatmaps")
    os.makedirs(out_dir, exist_ok=True)

    ds = CSVMILDataset(tiles_dir=cfg["tiles_dir"], labels_csv=cfg.get("labels_csv"),
                       bag_keys=["X", "coords"] + (["Y"] if cfg.get("labels_csv") else []))
    model.eval()
    with torch.no_grad():
        for pid in ds.patient_ids:
            bag    = ds._build_bag(pid)
            X      = bag["X"].unsqueeze(0).to(cfg["device"])
            coords = bag["coords"].numpy()

            Y_pred, att = model(X, return_att=True)
            att = att.squeeze(0).cpu().numpy()
            att = (att - att.min()) / (att.max() - att.min() + 1e-8)

            xs, ys = coords[:, 0], coords[:, 1]
            step   = max(1, int(np.median(np.diff(np.sort(np.unique(xs)))))) if len(np.unique(xs)) > 1 else 1
            col    = (np.round(xs / step) - np.round(xs / step).min()).astype(int)
            row    = (np.round(ys / step) - np.round(ys / step).min()).astype(int)

            grid = np.full((row.max() + 1, col.max() + 1), np.nan)
            for j in range(len(att)):
                grid[row[j], col[j]] = att[j]

            cmap = plt.get_cmap("RdYlGn_r").copy()
            cmap.set_bad("#e8e8e8")
            fig, ax = plt.subplots(figsize=(min(20, max(6, col.max() // 3)),
                                            min(20, max(6, row.max() // 3))))
            im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
            fig.colorbar(im, ax=ax, fraction=0.03, label="Attention")
            label = f"  label={int(bag['Y'].item())}" if "Y" in bag else ""
            ax.set_title(f"{pid}{label}  P(pos)={torch.sigmoid(Y_pred).item():.3f}", fontsize=9)
            ax.axis("off")
            fig.savefig(os.path.join(out_dir, f"{pid}_heatmap.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)
    print(f"Heatmaps → {out_dir}")


def main():
    os.makedirs(CFG["output_dir"], exist_ok=True)
    torch.manual_seed(CFG["seed"])

    train_loader, val_loader          = build_dataloaders(CFG)
    model, optimizer, scheduler       = build_model(CFG)
    history, best_epoch, best_val_auc = train(CFG, model, optimizer, scheduler, train_loader, val_loader)
    final_tracker                     = evaluate(CFG, model, val_loader, optimizer)

    plot_dashboard(history, best_epoch, best_val_auc, final_tracker, CFG["output_dir"])
    generate_heatmaps(CFG, model)


if __name__ == "__main__":
    main()