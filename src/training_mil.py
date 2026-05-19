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
        probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
        preds = (probs >= 0.5).astype(int)
        tgts  = targets.detach().cpu().numpy().ravel().astype(int)
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
            Y     = batch["Y"].float()

            if train:
                optimizer.zero_grad()

            out    = model(X)
            logits = out if isinstance(out, torch.Tensor) else out[0]
            loss   = model.criterion(logits.squeeze(), Y.squeeze())

            if train:
                loss.backward()
                optimizer.step()

            tracker.update(logits.squeeze(), Y.squeeze(), loss.item())

    return tracker.compute(), tracker


def plot_dashboard(history, best_epoch, best_val_auc, final_tracker, save_path):
    """Génère et sauvegarde le dashboard matplotlib (6 panels)."""
    n_epochs = len(history["train_loss"])
    EPOCHS   = list(range(1, n_epochs + 1))
    COLOR    = dict(train="#3B6D11", val="#185FA5")

    y_true = np.array(final_tracker.targets)
    y_prob = np.array(final_tracker.probs)
    y_pred = np.array(final_tracker.preds)

    fpr,  tpr,  _ = roc_curve(y_true, y_prob)
    prec, rec,  _ = precision_recall_curve(y_true, y_prob)
    cm            = confusion_matrix(y_true, y_pred)

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("MIL Training Dashboard", fontsize=16, fontweight="bold", y=0.98)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1. Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(EPOCHS, history["train_loss"], label="Train", color=COLOR["train"], linewidth=2)
    ax1.plot(EPOCHS, history["val_loss"],   label="Val",   color=COLOR["val"],   linewidth=2, linestyle="--")
    ax1.axvline(best_epoch, color="#A32D2D", linestyle=":", linewidth=1.5, label=f"Best (ep {best_epoch})")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    # 2. AUC
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(EPOCHS, history["train_auc"], label="Train", color=COLOR["train"], linewidth=2)
    ax2.plot(EPOCHS, history["val_auc"],   label="Val",   color=COLOR["val"],   linewidth=2, linestyle="--")
    ax2.axvline(best_epoch, color="#A32D2D", linestyle=":", linewidth=1.5)
    ax2.axhline(best_val_auc, color="#A32D2D", linestyle=":", linewidth=1, alpha=0.5)
    ax2.set_title("AUC-ROC"); ax2.set_xlabel("Epoch"); ax2.set_ylim(0, 1); ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    # 3. Accuracy
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(EPOCHS, history["train_acc"], label="Train", color=COLOR["train"], linewidth=2)
    ax3.plot(EPOCHS, history["val_acc"],   label="Val",   color=COLOR["val"],   linewidth=2, linestyle="--")
    ax3.axvline(best_epoch, color="#A32D2D", linestyle=":", linewidth=1.5)
    ax3.set_title("Accuracy"); ax3.set_xlabel("Epoch"); ax3.set_ylim(0, 1); ax3.legend(fontsize=9); ax3.grid(alpha=0.3)

    # 4. ROC
    ax4 = fig.add_subplot(gs[1, 0])
    auc_val = roc_auc_score(y_true, y_prob)
    ax4.plot(fpr, tpr, color=COLOR["val"], linewidth=2, label=f"AUC = {auc_val:.3f}")
    ax4.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4)
    ax4.set_title("ROC Curve (best model)"); ax4.set_xlabel("FPR"); ax4.set_ylabel("TPR")
    ax4.legend(fontsize=10); ax4.grid(alpha=0.3)

    # 5. Precision-Recall
    ax5 = fig.add_subplot(gs[1, 1])
    ap_val = average_precision_score(y_true, y_prob)
    ax5.plot(rec, prec, color=COLOR["train"], linewidth=2, label=f"AP = {ap_val:.3f}")
    ax5.set_title("Precision-Recall (best model)"); ax5.set_xlabel("Recall"); ax5.set_ylabel("Precision")
    ax5.set_xlim(0, 1); ax5.set_ylim(0, 1.05); ax5.legend(fontsize=10); ax5.grid(alpha=0.3)

    # 6. Confusion matrix
    ax6 = fig.add_subplot(gs[1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax6, colorbar=False, cmap="Blues")
    ax6.set_title("Confusion Matrix (best model)")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Dashboard sauvegardé → {save_path}")
