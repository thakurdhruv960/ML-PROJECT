"""
Speaker-independent k-fold cross-validation for the CNN-BiLSTM.

With ~120 speakers, a single train/val/test split is noisy — you saw this
firsthand when val accuracy swung 15-19 points epoch to epoch. Running
k folds and averaging gives a far more trustworthy accuracy estimate for
your report ("our model achieves 67.2% +/- 3.1% across 5 speaker-independent
folds" is a stronger, more defensible claim than a single 65.7% number).

This does NOT replace train.py for producing your final deployed model —
it's an evaluation exercise. Once you've confirmed a stable estimate here,
keep using train.py's single-split checkpoint for live_inference.py.

Usage:
    python src/train_kfold.py
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold

from config import (
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, LABEL_SMOOTHING, NUM_EPOCHS,
)
from dataset import load_manifest, SERDataset, get_class_weights
from model import CNNBiLSTMAttention
from train import get_device, run_epoch

N_FOLDS = 5
# Fewer epochs per fold than the main train.py run — this is for a stable
# *estimate*, not for producing a deployable model, so we cap it to keep
# total runtime (N_FOLDS x epochs) reasonable. Raise if you have time to spare.
EPOCHS_PER_FOLD = 20
PATIENCE_PER_FOLD = 6


def run_fold(train_df, val_df, device, fold_num):
    train_ds = SERDataset(train_df, train=True)
    val_ds = SERDataset(val_df, train=False)

    loader_kwargs = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)

    model = CNNBiLSTMAttention().to(device)
    class_weights = get_class_weights(train_df).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS_PER_FOLD + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        print(f"  Fold {fold_num} | epoch {epoch:2d}/{EPOCHS_PER_FOLD} | "
              f"train acc {train_acc:.4f} | val acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE_PER_FOLD:
                print(f"  Fold {fold_num}: early stop at epoch {epoch}")
                break

    return best_val_acc


def main():
    device = get_device()
    print(f"Using device: {device}")

    manifest = load_manifest()
    speakers = manifest["actor_id"].unique()
    print(f"Total speakers: {len(speakers)} | Running {N_FOLDS}-fold "
          f"speaker-independent cross-validation\n")

    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_accuracies = []

    # GroupKFold needs one row per sample with its group (speaker) label
    X = manifest.index.values
    groups = manifest["actor_id"].values

    for fold_num, (train_idx, val_idx) in enumerate(gkf.split(X, groups=groups), start=1):
        train_df = manifest.iloc[train_idx].reset_index(drop=True)
        val_df = manifest.iloc[val_idx].reset_index(drop=True)

        # Sanity check: confirm no speaker overlap for this fold
        overlap = set(train_df["actor_id"]) & set(val_df["actor_id"])
        assert not overlap, f"Fold {fold_num}: speaker leak detected: {overlap}"

        print(f"=== Fold {fold_num}/{N_FOLDS} "
              f"(train speakers: {train_df['actor_id'].nunique()}, "
              f"val speakers: {val_df['actor_id'].nunique()}) ===")
        acc = run_fold(train_df, val_df, device, fold_num)
        fold_accuracies.append(acc)
        print(f"=== Fold {fold_num} best val accuracy: {acc:.4f} ===\n")

    fold_accuracies = np.array(fold_accuracies)
    print("=" * 60)
    print("K-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 60)
    for i, acc in enumerate(fold_accuracies, start=1):
        print(f"  Fold {i}: {acc:.4f}")
    print(f"\nMean accuracy: {fold_accuracies.mean():.4f} +/- {fold_accuracies.std():.4f}")
    print("\nUse this mean +/- std as your reported accuracy estimate — it's more "
          "defensible than any single split's number, especially with this few speakers.")


if __name__ == "__main__":
    main()
