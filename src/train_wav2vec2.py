"""
Fine-tune wav2vec2 for speech emotion recognition.

Usage:
    python src/train_wav2vec2.py

This is the pretrained-embedding alternative to train.py's from-scratch
CNN-BiLSTM. Same speaker-independent split, same manifest, same class
weighting/label smoothing philosophy — only the model and its optimal
hyperparameters differ (much lower LR, far fewer epochs needed).
"""
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    WAV2VEC2_CHECKPOINT_DIR, WAV2VEC2_BATCH_SIZE, WAV2VEC2_NUM_EPOCHS,
    WAV2VEC2_LEARNING_RATE, WAV2VEC2_WEIGHT_DECAY, WAV2VEC2_EARLY_STOP_PATIENCE,
    WAV2VEC2_MODEL_NAME, WAV2VEC2_FREEZE_FEATURE_ENCODER,
    LABEL_SMOOTHING, VAL_LOSS_EMA_ALPHA, NUM_CLASSES,
)
from dataset import load_manifest, speaker_split, get_class_weights
from wav2vec2_dataset import Wav2Vec2SERDataset
from wav2vec2_model import build_wav2vec2_model
from train import get_device


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    torch.set_grad_enabled(train)
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad()
        logits = model(input_values=x).logits
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
    if str(device) == "cpu":
        print("WARNING: fine-tuning wav2vec2 on CPU will be very slow. "
              "A GPU is strongly recommended for this script.")

    manifest = load_manifest()
    train_df, val_df, test_df = speaker_split(manifest)
    print(f"Speakers -> train: {train_df['actor_id'].nunique()}, "
          f"val: {val_df['actor_id'].nunique()}, test: {test_df['actor_id'].nunique()}")
    print(f"Clips    -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    train_ds = Wav2Vec2SERDataset(train_df, train=True)
    val_ds = Wav2Vec2SERDataset(val_df, train=False)

    loader_kwargs = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_ds, batch_size=WAV2VEC2_BATCH_SIZE, shuffle=True,
                               drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=WAV2VEC2_BATCH_SIZE, shuffle=False,
                             **loader_kwargs)

    print(f"\nLoading pretrained model: {WAV2VEC2_MODEL_NAME} (requires internet "
          f"on first run to download weights from huggingface.co)")
    model = build_wav2vec2_model(
        num_labels=NUM_CLASSES,
        pretrained_name=WAV2VEC2_MODEL_NAME,
        freeze_feature_encoder=WAV2VEC2_FREEZE_FEATURE_ENCODER,
    ).to(device)

    class_weights = get_class_weights(train_df).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=WAV2VEC2_LEARNING_RATE,
                                   weight_decay=WAV2VEC2_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    os.makedirs(WAV2VEC2_CHECKPOINT_DIR, exist_ok=True)
    best_ema_val_loss = float("inf")
    ema_val_loss = None
    epochs_no_improve = 0

    print("\nStarting fine-tuning...\n")
    for epoch in range(1, WAV2VEC2_NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        ema_val_loss = val_loss if ema_val_loss is None else (
            VAL_LOSS_EMA_ALPHA * val_loss + (1 - VAL_LOSS_EMA_ALPHA) * ema_val_loss
        )
        scheduler.step(ema_val_loss)
        dt = time.time() - t0

        print(f"Epoch {epoch:3d}/{WAV2VEC2_NUM_EPOCHS} | "
              f"train_loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val_loss {val_loss:.4f} (ema {ema_val_loss:.4f}) acc {val_acc:.4f} | {dt:.1f}s")

        if ema_val_loss < best_ema_val_loss:
            best_ema_val_loss = ema_val_loss
            epochs_no_improve = 0
            ckpt_path = os.path.join(WAV2VEC2_CHECKPOINT_DIR, "best_model.pt")
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
            if epochs_no_improve >= WAV2VEC2_EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping: no improvement for {WAV2VEC2_EARLY_STOP_PATIENCE} epochs.")
                break

    print(f"\nFine-tuning complete. Best model saved to "
          f"{os.path.join(WAV2VEC2_CHECKPOINT_DIR, 'best_model.pt')}")
    print("Run `python src/evaluate_wav2vec2.py` to test on the held-out speakers.")


if __name__ == "__main__":
    main()
