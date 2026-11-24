"""
Train the CNN-BiLSTM-Attention model.

Usage:
    python src/train.py
"""
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    CHECKPOINT_DIR, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOP_PATIENCE, DEVICE, LABEL_SMOOTHING, VAL_LOSS_EMA_ALPHA,
)
from dataset import load_manifest, speaker_split, SERDataset, get_class_weights
from model import CNNBiLSTMAttention


def get_device():
    if DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    torch.set_grad_enabled(train)
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def main():
    device = get_device()
    print(f"Using device: {device}")

    manifest = load_manifest()
    train_df, val_df, test_df = speaker_split(manifest)
    print(f"Speakers -> train: {train_df['actor_id'].nunique()}, "
          f"val: {val_df['actor_id'].nunique()}, test: {test_df['actor_id'].nunique()}")
    print(f"Clips    -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    train_ds = SERDataset(train_df, train=True)
    val_ds = SERDataset(val_df, train=False)

    # num_workers parallelizes the CPU-bound librosa feature extraction across
    # cores so the GPU isn't left waiting on it — raise this toward your CPU's
    # core count if epochs are still slow (try os.cpu_count()).
    loader_kwargs = dict(num_workers=6, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)

    model = CNNBiLSTMAttention().to(device)
    class_weights = get_class_weights(train_df).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_ema_val_loss = float("inf")
    ema_val_loss = None
    epochs_no_improve = 0

    print("\nStarting training...\n")
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        # Smooth the noisy per-epoch val_loss so checkpoint selection/early
        # stopping react to the underlying trend, not epoch-to-epoch noise
        # from a small validation set.
        ema_val_loss = val_loss if ema_val_loss is None else (
            VAL_LOSS_EMA_ALPHA * val_loss + (1 - VAL_LOSS_EMA_ALPHA) * ema_val_loss
        )
        scheduler.step(ema_val_loss)
        dt = time.time() - t0

        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
              f"train_loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val_loss {val_loss:.4f} (ema {ema_val_loss:.4f}) acc {val_acc:.4f} | {dt:.1f}s")

        if ema_val_loss < best_ema_val_loss:
            best_ema_val_loss = ema_val_loss
            epochs_no_improve = 0
            ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "ema_val_loss": ema_val_loss,
                "val_acc": val_acc,
            }, ckpt_path)
            print(f"  -> saved new best checkpoint (ema_val_loss {ema_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping: no improvement for {EARLY_STOP_PATIENCE} epochs.")
                break

    print("\nTraining complete. Best model saved to checkpoints/best_model.pt")
    print("Run `python src/evaluate.py` to test on the held-out speakers.")


if __name__ == "__main__":
    main()
