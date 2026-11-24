"""
Evaluate the fine-tuned wav2vec2 model on the held-out test speakers.

Usage:
    python src/evaluate_wav2vec2.py
"""
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from config import WAV2VEC2_CHECKPOINT_DIR, WAV2VEC2_BATCH_SIZE, WAV2VEC2_MODEL_NAME, EMOTIONS, NUM_CLASSES, PROJECT_ROOT
from dataset import load_manifest, speaker_split
from wav2vec2_dataset import Wav2Vec2SERDataset
from wav2vec2_model import build_wav2vec2_model
from train import get_device
from evaluate import _plot_confusion_matrix


def evaluate():
    device = get_device()
    manifest = load_manifest()
    _, _, test_df = speaker_split(manifest)
    test_ds = Wav2Vec2SERDataset(test_df, train=False)
    test_loader = DataLoader(test_ds, batch_size=WAV2VEC2_BATCH_SIZE, shuffle=False, num_workers=2)

    model = build_wav2vec2_model(num_labels=NUM_CLASSES, pretrained_name=WAV2VEC2_MODEL_NAME,
                                  freeze_feature_encoder=True).to(device)
    ckpt_path = os.path.join(WAV2VEC2_CHECKPOINT_DIR, "best_model.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val_loss {checkpoint['val_loss']:.4f}, val_acc {checkpoint['val_acc']:.4f})")

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(input_values=x).logits
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    print("\nClassification report (test speakers, never seen in training):\n")
    print(classification_report(all_labels, all_preds, target_names=EMOTIONS, digits=3))

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(EMOTIONS))))
    _plot_confusion_matrix(cm, EMOTIONS, filename="wav2vec2_confusion_matrix.png",
                            title_suffix=" (wav2vec2)")


if __name__ == "__main__":
    evaluate()
