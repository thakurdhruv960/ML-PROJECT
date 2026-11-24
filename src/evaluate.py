"""
Evaluate the trained model on the held-out test speakers.
Prints per-class precision/recall/F1 and saves a confusion matrix plot.

Usage:
    python src/evaluate.py
"""
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from config import CHECKPOINT_DIR, BATCH_SIZE, EMOTIONS, PROJECT_ROOT
from dataset import load_manifest, speaker_split, SERDataset
from model import CNNBiLSTMAttention
from train import get_device


def evaluate():
    device = get_device()
    manifest = load_manifest()
    _, _, test_df = speaker_split(manifest)
    test_ds = SERDataset(test_df, train=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = CNNBiLSTMAttention().to(device)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val_loss {checkpoint['val_loss']:.4f}, val_acc {checkpoint['val_acc']:.4f})")

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    print("\nClassification report (test speakers, never seen in training):\n")
    print(classification_report(all_labels, all_preds, target_names=EMOTIONS, digits=3))

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(EMOTIONS))))
    _plot_confusion_matrix(cm, EMOTIONS, filename="confusion_matrix.png")


def _plot_confusion_matrix(cm, labels, filename="confusion_matrix.png", title_suffix=""):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (row-normalized) — held-out test speakers{title_suffix}")

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm_norm[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                     color="white" if val > 0.5 else "black", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    out_path = os.path.join(PROJECT_ROOT, "checkpoints", filename)
    fig.savefig(out_path, dpi=150)
    print(f"\nConfusion matrix saved to {out_path}")


if __name__ == "__main__":
    evaluate()
