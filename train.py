import os
import torch
from torch.utils.data import DataLoader, random_split
from torchmil.models import ABMIL
from torchmil.data import collate_fn
from class_mil import CSVMILDataset
from training_mil import run_epoch, plot_dashboard
import torch



 
CFG = dict(
    slide_features_csv = "slides_features.csv",
    tiles_dir          = "tiles/",
    labels_csv         = "labels.csv",
    bag_keys           = ["X", "X_slide", "Y", "coords"],
    in_shape           = (2048,),
    lr                 = 1e-4,
    epochs             = 30,
    batch_size         = 4,
    val_split          = 0.2,
    device             = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir         = "outputs/",
    seed               = 42,
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
    model     = ABMIL(in_shape=cfg["in_shape"], criterion=torch.nn.BCEWithLogitsLoss())
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


def main():
    os.makedirs(CFG["output_dir"], exist_ok=True)
    torch.manual_seed(CFG["seed"])

    train_loader, val_loader          = build_dataloaders(CFG)
    model, optimizer, scheduler       = build_model(CFG)
    history, best_epoch, best_val_auc = train(CFG, model, optimizer, scheduler, train_loader, val_loader)
    final_tracker                     = evaluate(CFG, model, val_loader, optimizer)

    plot_dashboard(history, best_epoch, best_val_auc, final_tracker, CFG["output_dir"])


if __name__ == "__main__":
    main()
