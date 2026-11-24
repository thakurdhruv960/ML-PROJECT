"""
Ensemble the CNN-BiLSTM and fine-tuned wav2vec2 models by averaging their
predicted probabilities on the test set. Requires both models to have been
trained already (checkpoints/best_model.pt and
checkpoints/wav2vec2/best_model.pt).

Usage:
    python src/ensemble.py
"""
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from config import (
    CHECKPOINT_DIR, WAV2VEC2_CHECKPOINT_DIR, BATCH_SIZE, WAV2VEC2_BATCH_SIZE,
    WAV2VEC2_MODEL_NAME, EMOTIONS, NUM_CLASSES,
)
from dataset import load_manifest, speaker_split, SERDataset
from wav2vec2_dataset import Wav2Vec2SERDataset
from model import CNNBiLSTMAttention
from wav2vec2_model import build_wav2vec2_model
from train import get_device
from evaluate import _plot_confusion_matrix


def get_cnn_probs(test_df, device):
    ds = SERDataset(test_df, train=False)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = CNNBiLSTMAttention().to(device)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            probs = F.softmax(model(x), dim=1).cpu()
            all_probs.append(probs)
            all_labels.append(y)
    return torch.cat(all_probs), torch.cat(all_labels)


def get_wav2vec2_probs(test_df, device):
    ds = Wav2Vec2SERDataset(test_df, train=False)
    loader = DataLoader(ds, batch_size=WAV2VEC2_BATCH_SIZE, shuffle=False, num_workers=2)

    model = build_wav2vec2_model(num_labels=NUM_CLASSES, pretrained_name=WAV2VEC2_MODEL_NAME,
                                  freeze_feature_encoder=True).to(device)
    ckpt_path = os.path.join(WAV2VEC2_CHECKPOINT_DIR, "best_model.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            probs = F.softmax(model(input_values=x).logits, dim=1).cpu()
            all_probs.append(probs)
            all_labels.append(y)
    return torch.cat(all_probs), torch.cat(all_labels)


def main():
    device = get_device()
    manifest = load_manifest()
    _, _, test_df = speaker_split(manifest)
    # Both loaders iterate the same test_df with shuffle=False, so row order
    # matches — this is what lets us average probabilities index-for-index.
    test_df = test_df.reset_index(drop=True)

    print("Running CNN-BiLSTM inference...")
    cnn_probs, cnn_labels = get_cnn_probs(test_df, device)

    print("Running wav2vec2 inference...")
    w2v_probs, w2v_labels = get_wav2vec2_probs(test_df, device)

    assert torch.equal(cnn_labels, w2v_labels), (
        "Label order mismatch between the two models' loaders — both datasets "
        "must be built from the identical, reset-index test_df with shuffle=False."
    )

    ensemble_probs = (cnn_probs + w2v_probs) / 2
    preds = ensemble_probs.argmax(dim=1).numpy()
    labels = cnn_labels.numpy()

    print("\n" + "=" * 60)
    print("ENSEMBLE (CNN-BiLSTM + wav2vec2, averaged probabilities)")
    print("=" * 60)
    print(classification_report(labels, preds, target_names=EMOTIONS, digits=3))

    cm = confusion_matrix(labels, preds, labels=list(range(len(EMOTIONS))))
    _plot_confusion_matrix(cm, EMOTIONS, filename="ensemble_confusion_matrix.png",
                            title_suffix=" (ensemble)")

    # Also print each individual model's accuracy for direct comparison
    cnn_acc = (cnn_probs.argmax(dim=1) == cnn_labels).float().mean().item()
    w2v_acc = (w2v_probs.argmax(dim=1) == w2v_labels).float().mean().item()
    ensemble_acc = (torch.tensor(preds) == cnn_labels).float().mean().item()
    print(f"\nCNN-BiLSTM alone : {cnn_acc:.4f}")
    print(f"wav2vec2 alone   : {w2v_acc:.4f}")
    print(f"Ensemble         : {ensemble_acc:.4f}")


if __name__ == "__main__":
    main()
