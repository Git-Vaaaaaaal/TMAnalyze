import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.class_mil import CSVMILDataset
from src.training_mil import run_epoch



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
