import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
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

            adj = batch.get("adj", None)
            out = model(X, adj) if adj is not None else model(X)
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


def plot_accuracy_curves(history, best_epoch, save_path):
    """Courbes d'accuracy train / validation sur les epochs."""
    n_epochs = len(history["train_acc"])
    EPOCHS   = list(range(1, n_epochs + 1))
    COLOR    = dict(train="#3B6D11", val="#185FA5")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(EPOCHS, history["train_acc"], label="Train", color=COLOR["train"], linewidth=2)
    ax.plot(EPOCHS, history["val_acc"],   label="Val",   color=COLOR["val"],   linewidth=2, linestyle="--")
    ax.axvline(best_epoch, color="#A32D2D", linestyle=":", linewidth=1.5, label=f"Best (ep {best_epoch})")
    ax.set_title("Accuracy — Train / Validation", fontsize=13)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1); ax.legend(fontsize=10); ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Courbes accuracy → {save_path}")


def plot_confusion_matrix(final_tracker, save_path):
    """Matrice de confusion sur le jeu de test."""
    y_true = np.array(final_tracker.targets)
    y_pred = np.array(final_tracker.preds)
    cm     = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Matrice de confusion (test)", fontsize=13)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Matrice confusion → {save_path}")

