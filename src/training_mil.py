import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, precision_recall_curve,
)
from torch.utils.data import DataLoader
import inspect


class MetricTracker:
    """Accumule les prédictions et targets pour calculer les métriques."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.preds    = []
        self.probs    = []
        self.targets  = []
        self.loss_sum  = 0.0
        self.n_batches = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: float):
        tgts = targets.detach().cpu().numpy().ravel().astype(int)
        is_multiclass = logits.dim() > 1 and logits.shape[-1] > 1
        if is_multiclass:
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()  # [B, n_classes]
            preds = probs.argmax(axis=-1)
            self.probs.extend(probs.tolist())
        else:
            probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
            preds = (probs >= 0.5).astype(int)
            self.probs.extend(probs.tolist())
        self.preds.extend(preds.tolist())
        self.targets.extend(tgts.tolist())
        self.loss_sum  += loss
        self.n_batches += 1

    def compute(self) -> dict:
        y    = np.array(self.targets)
        prob = np.array(self.probs)
        pred = np.array(self.preds)
        loss = self.loss_sum / max(self.n_batches, 1)
        acc  = (pred == y).mean()
        try:
            if prob.ndim == 2:  # multi-class: probs is [N, n_classes]
                auc = roc_auc_score(y, prob, multi_class="ovr")
                ap  = float("nan")
            else:
                auc = roc_auc_score(y, prob)
                ap  = average_precision_score(y, prob)
        except ValueError:
            auc = ap = float("nan")
        return dict(loss=loss, acc=acc, auc=auc, ap=ap)


def run_epoch(model, loader, optimizer, device, train=True):
    """Exécute une epoch complète (train ou val) et retourne les métriques."""
    model.train() if train else model.eval()
    tracker = MetricTracker()
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for batch in loader:
            batch = batch.to(device)
            X     = batch["X"]
            Y     = batch["Y"]

            if train:
                optimizer.zero_grad()

            out    = model(X)
            logits = out if isinstance(out, torch.Tensor) else out[0]
            # DSMIL multi-class: bag_classifier output shape is [B, 1, n_classes] — squeeze the middle dim
            if logits.dim() == 3 and logits.shape[1] == 1:
                logits = logits.squeeze(1)

            is_multiclass = logits.dim() > 1 and logits.shape[-1] > 1
            if is_multiclass:
                loss = model.criterion(logits, Y.long().flatten())
            else:
                loss = model.criterion(logits.flatten(), Y.float().flatten())

            if train:
                loss.backward()
                optimizer.step()

            tracker.update(logits, Y, loss.item())

    return tracker.compute(), tracker


def plot_dashboard(history, best_epoch, best_val_auc, final_tracker, save_path):
    """Génère et sauvegarde le dashboard matplotlib (binaire: 6 panels, multi-class: 4 panels)."""
    n_epochs = len(history["train_loss"])
    EPOCHS   = list(range(1, n_epochs + 1))
    COLOR    = dict(train="#3B6D11", val="#185FA5")

    y_true = np.array(final_tracker.targets)
    y_prob = np.array(final_tracker.probs)
    y_pred = np.array(final_tracker.preds)
    cm     = confusion_matrix(y_true, y_pred)
    is_multiclass = y_prob.ndim == 2

    if is_multiclass:
        fig = plt.figure(figsize=(16, 8))
        fig.suptitle("MIL Training Dashboard", fontsize=16, fontweight="bold", y=0.98)
        gs  = gridspec.GridSpec(1, 4, figure=fig, hspace=0.4, wspace=0.35)
        axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

        # 1. Loss
        axes[0].plot(EPOCHS, history["train_loss"], label="Train", color=COLOR["train"], linewidth=2)
        axes[0].plot(EPOCHS, history["val_loss"],   label="Val",   color=COLOR["val"],   linewidth=2, linestyle="--")
        axes[0].axvline(best_epoch, color="#A32D2D", linestyle=":", linewidth=1.5, label=f"Best (ep {best_epoch})")
        axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

        # 2. AUC (OvR)
        axes[1].plot(EPOCHS, history["train_auc"], label="Train", color=COLOR["train"], linewidth=2)
        axes[1].plot(EPOCHS, history["val_auc"],   label="Val",   color=COLOR["val"],   linewidth=2, linestyle="--")
        axes[1].axvline(best_epoch, color="#A32D2D", linestyle=":", linewidth=1.5)
        axes[1].set_title("AUC-ROC (OvR)"); axes[1].set_xlabel("Epoch"); axes[1].set_ylim(0, 1)
        axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

        # 3. Accuracy
        axes[2].plot(EPOCHS, history["train_acc"], label="Train", color=COLOR["train"], linewidth=2)
        axes[2].plot(EPOCHS, history["val_acc"],   label="Val",   color=COLOR["val"],   linewidth=2, linestyle="--")
        axes[2].axvline(best_epoch, color="#A32D2D", linestyle=":", linewidth=1.5)
        axes[2].set_title("Accuracy"); axes[2].set_xlabel("Epoch"); axes[2].set_ylim(0, 1)
        axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

        # 4. Confusion matrix
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=axes[3], colorbar=False, cmap="Blues")
        axes[3].set_title("Confusion Matrix (best model)")

    else:
        fpr,  tpr,  _ = roc_curve(y_true, y_prob)
        prec, rec,  _ = precision_recall_curve(y_true, y_prob)

        fig = plt.figure(figsize=(18, 12))
        fig.suptitle("MIL Training Dashboard", fontsize=16, fontweight="bold", y=0.98)
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(EPOCHS, history["train_loss"], label="Train", color=COLOR["train"], linewidth=2)
        ax1.plot(EPOCHS, history["val_loss"],   label="Val",   color=COLOR["val"],   linewidth=2, linestyle="--")
        ax1.axvline(best_epoch, color="#A32D2D", linestyle=":", linewidth=1.5, label=f"Best (ep {best_epoch})")
        ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(EPOCHS, history["train_auc"], label="Train", color=COLOR["train"], linewidth=2)
        ax2.plot(EPOCHS, history["val_auc"],   label="Val",   color=COLOR["val"],   linewidth=2, linestyle="--")
        ax2.axvline(best_epoch, color="#A32D2D", linestyle=":", linewidth=1.5)
        ax2.axhline(best_val_auc, color="#A32D2D", linestyle=":", linewidth=1, alpha=0.5)
        ax2.set_title("AUC-ROC"); ax2.set_xlabel("Epoch"); ax2.set_ylim(0, 1); ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(EPOCHS, history["train_acc"], label="Train", color=COLOR["train"], linewidth=2)
        ax3.plot(EPOCHS, history["val_acc"],   label="Val",   color=COLOR["val"],   linewidth=2, linestyle="--")
        ax3.axvline(best_epoch, color="#A32D2D", linestyle=":", linewidth=1.5)
        ax3.set_title("Accuracy"); ax3.set_xlabel("Epoch"); ax3.set_ylim(0, 1); ax3.legend(fontsize=9); ax3.grid(alpha=0.3)

        ax4 = fig.add_subplot(gs[1, 0])
        auc_val = roc_auc_score(y_true, y_prob)
        ax4.plot(fpr, tpr, color=COLOR["val"], linewidth=2, label=f"AUC = {auc_val:.3f}")
        ax4.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4)
        ax4.set_title("ROC Curve (best model)"); ax4.set_xlabel("FPR"); ax4.set_ylabel("TPR")
        ax4.legend(fontsize=10); ax4.grid(alpha=0.3)

        ax5 = fig.add_subplot(gs[1, 1])
        ap_val = average_precision_score(y_true, y_prob)
        ax5.plot(rec, prec, color=COLOR["train"], linewidth=2, label=f"AP = {ap_val:.3f}")
        ax5.set_title("Precision-Recall (best model)"); ax5.set_xlabel("Recall"); ax5.set_ylabel("Precision")
        ax5.set_xlim(0, 1); ax5.set_ylim(0, 1.05); ax5.legend(fontsize=10); ax5.grid(alpha=0.3)

        ax6 = fig.add_subplot(gs[1, 2])
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax6, colorbar=False, cmap="Blues")
        ax6.set_title("Confusion Matrix (best model)")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Dashboard sauvegardé → {save_path}")




# ======================================================================
# Concaténation
# ======================================================================

# ======================================================================
# Epoch — gère mask optionnel (TransMIL n'en accepte pas) + binaire/multi-class
# ======================================================================

class _Tracker:
    """Accumulateur léger compatible avec plot_dashboard."""
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

    _has_mask    = "mask" in inspect.signature(model.forward).parameters
    is_multiclass = n_classes > 2
    total_loss   = 0.0
    n_batches    = 0
    tracker      = _Tracker()

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for bags, labels, mask in loader:
            bags   = bags.to(device)
            labels = labels.to(device)
            mask   = mask.to(device)

            if train:
                optimizer.zero_grad()

            out    = model(bags, mask) if _has_mask else model(bags)
            logits = out if isinstance(out, torch.Tensor) else out[0]

            # DSMIL multi-class: bag_classifier produit [B, 1, n_classes]
            if logits.dim() == 3 and logits.shape[1] == 1:
                logits = logits.squeeze(1)

            if is_multiclass:
                loss = model.criterion(logits, labels.long().flatten())
                probs = torch.softmax(logits.detach(), dim=-1).cpu().numpy()
                preds = probs.argmax(axis=-1).tolist()
            else:
                loss = model.criterion(logits.flatten(), labels.float().flatten())
                probs = torch.sigmoid(logits.detach()).cpu().numpy().ravel()
                preds = (probs >= 0.5).astype(int).tolist()
                probs = probs.tolist()

            if train:
                loss.backward()
                optimizer.step()

            tgts = labels.detach().cpu().numpy().ravel().astype(int).tolist()
            tracker.targets.extend(tgts)
            tracker.probs.extend(probs if isinstance(probs, list) else probs.tolist())
            tracker.preds.extend(preds)
            total_loss += loss.item()
            n_batches  += 1

    y    = np.array(tracker.targets)
    prob = np.array(tracker.probs)
    pred = np.array(tracker.preds)
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
    best_val_auc = -1.0
    best_epoch   = 0
    final_tracker = None

    for epoch in range(1, epochs + 1):
        train_m, _       = run_epoch(model, train_loader, optimizer, device, n_classes, train=True)
        val_m, tracker   = run_epoch(model, val_loader,   optimizer, device, n_classes, train=False)
        scheduler.step()

        for phase, m in [("train", train_m), ("val", val_m)]:
            for key in ["loss", "acc", "auc", "ap"]:
                history[f"{phase}_{key}"].append(m[key])

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train — loss: {train_m['loss']:.4f}  acc: {train_m['acc']:.3f}  AUC: {train_m['auc']:.3f} | "
            f"Val   — loss: {val_m['loss']:.4f}  acc: {val_m['acc']:.3f}  AUC: {val_m['auc']:.3f}"
        )

        if val_m["auc"] > best_val_auc:
            best_val_auc  = val_m["auc"]
            best_epoch    = epoch
            final_tracker = tracker
            # torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))

    msg = f"\n{run_label} — meilleur epoch {best_epoch}, val AUC: {best_val_auc:.4f}"
    print(msg)
    with open("output_logs_concat.txt", "a") as f:
        f.write(msg + "\n")

    return history, best_epoch, best_val_auc, final_tracker

