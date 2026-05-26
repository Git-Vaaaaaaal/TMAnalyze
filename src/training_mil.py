import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve,
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


def plot_dashboard(history, best_epoch, final_tracker, save_path):
    """Génère et sauvegarde le dashboard matplotlib : Loss | ROC Curve | Confusion Matrix."""
    n_epochs = len(history["train_loss"])
    EPOCHS   = list(range(1, n_epochs + 1))
    COLOR    = dict(train="#3B6D11", val="#185FA5")

    y_true = np.array(final_tracker.targets)
    y_prob = np.array(final_tracker.probs)
    y_pred = np.array(final_tracker.preds)
    cm     = confusion_matrix(y_true, y_pred)
    is_multiclass = y_prob.ndim == 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("MIL Training Dashboard", fontsize=14, fontweight="bold")

    # 1. Loss
    ax = axes[0]
    ax.plot(EPOCHS, history["train_loss"], label="Train", color=COLOR["train"], linewidth=2)
    ax.plot(EPOCHS, history["val_loss"],   label="Val",   color=COLOR["val"],   linewidth=2, linestyle="--")
    ax.axvline(best_epoch, color="#A32D2D", linestyle=":", linewidth=1.5, label=f"Best (ep {best_epoch})")
    ax.set_title("Loss"); ax.set_xlabel("Epoch"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # 2. ROC Curve
    ax = axes[1]
    try:
        if is_multiclass:
            n_classes = y_prob.shape[1]
            for c in range(n_classes):
                fpr, tpr, _ = roc_curve((y_true == c).astype(int), y_prob[:, c])
                auc_c = roc_auc_score((y_true == c).astype(int), y_prob[:, c])
                ax.plot(fpr, tpr, linewidth=2, label=f"Class {c} (AUC={auc_c:.2f})")
        else:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_val = roc_auc_score(y_true, y_prob)
            ax.plot(fpr, tpr, color=COLOR["val"], linewidth=2, label=f"AUC = {auc_val:.3f}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4)
    except ValueError:
        ax.text(0.5, 0.5, "ROC non disponible", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("ROC Curve (best model)"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # 3. Confusion Matrix
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=axes[2], colorbar=False, cmap="Blues")
    axes[2].set_title("Confusion Matrix (best model)")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Dashboard sauvegardé → {save_path}")

