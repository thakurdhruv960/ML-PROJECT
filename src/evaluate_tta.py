"""
Evaluate the CNN-BiLSTM with test-time augmentation (TTA): each test clip is
run through the original version plus a few mildly-augmented versions, and
their predicted probabilities are averaged. Usually a small, consistent
accuracy gain over evaluating the original clip alone.

Usage:
    python src/evaluate_tta.py
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from config import CHECKPOINT_DIR, EMOTIONS, EMOTION_TO_IDX
from dataset import load_manifest, speaker_split
from preprocessing import preprocess_clip, pitch_shift, time_shift, normalize_amplitude
from features import extract_log_mel
from model import CNNBiLSTMAttention
from train import get_device
from evaluate import _plot_confusion_matrix

# Mild, fixed perturbations for TTA — deliberately gentler than training
# augmentation, since the point is robustness-checking around the original
# clip, not simulating heavy real-world distortion.
TTA_VARIANTS = [
    lambda audio: audio,                                     # original, unaugmented
    lambda audio: normalize_amplitude(pitch_shift(audio, semitone_range=(-1, 1))),
    lambda audio: normalize_amplitude(time_shift(audio, shift_max_frac=0.05)),
]


def predict_with_tta(model, device, filepath):
    audio = preprocess_clip(filepath)
    probs_sum = None

    for variant_fn in TTA_VARIANTS:
        variant_audio = variant_fn(audio.copy())
        log_mel = extract_log_mel(variant_audio)
        x = torch.from_numpy(log_mel).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = F.softmax(model(x), dim=1).cpu().numpy()[0]
        probs_sum = probs if probs_sum is None else probs_sum + probs

    avg_probs = probs_sum / len(TTA_VARIANTS)
    return avg_probs


def evaluate():
    device = get_device()
    manifest = load_manifest()
    _, _, test_df = speaker_split(manifest)
    test_df = test_df.reset_index(drop=True)

    model = CNNBiLSTMAttention().to(device)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Running TTA with {len(TTA_VARIANTS)} variants per clip "
          f"({len(test_df)} clips -> {len(test_df) * len(TTA_VARIANTS)} forward passes)...\n")

    all_preds, all_labels = [], []
    for i, row in test_df.iterrows():
        avg_probs = predict_with_tta(model, device, row["filepath"])
        pred = int(np.argmax(avg_probs))
        all_preds.append(pred)
        all_labels.append(EMOTION_TO_IDX[row["emotion"]])
        if (i + 1) % 500 == 0:
            print(f"  ...{i + 1}/{len(test_df)} clips processed")

    print("\nClassification report WITH test-time augmentation:\n")
    print(classification_report(all_labels, all_preds, target_names=EMOTIONS, digits=3))

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(EMOTIONS))))
    _plot_confusion_matrix(cm, EMOTIONS, filename="tta_confusion_matrix.png",
                            title_suffix=" (with TTA)")

    print("\nCompare this classification report against `python src/evaluate.py`'s "
          "(no TTA) output on the same checkpoint to see the actual gain from TTA.")


if __name__ == "__main__":
    evaluate()
