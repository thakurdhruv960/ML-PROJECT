"""
Record a single fixed-duration clip from the microphone, then predict its
emotion once and print the result. Unlike live_inference.py (which streams
continuously), this is a discrete "speak now -> here's the answer" flow —
better suited for a live demo in front of examiners.

Usage:
    python src/record_and_predict.py
    python src/record_and_predict.py --seconds 5
"""
import os
import argparse
import numpy as np
import sounddevice as sd
import torch
import torch.nn.functional as F

from config import SAMPLE_RATE, EMOTIONS, CHECKPOINT_DIR
from preprocessing import trim_silence, normalize_amplitude, pad_or_truncate
from features import extract_log_mel
from model import CNNBiLSTMAttention
from train import get_device


def load_model():
    device = get_device()
    model = CNNBiLSTMAttention().to(device)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device


def record_clip(seconds, sr=SAMPLE_RATE):
    print(f"Recording for {seconds} seconds... speak now.")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()  # blocks until the recording finishes
    print("Done recording.")
    return audio[:, 0]  # drop the channel dimension -> 1D array


def predict_emotion(model, device, audio):
    """audio: 1D float32 numpy array, already at SAMPLE_RATE. Same pipeline
    used everywhere else in this project (preprocessing.py + features.py)."""
    audio = trim_silence(audio)
    audio = normalize_amplitude(audio)
    audio = pad_or_truncate(audio)
    log_mel = extract_log_mel(audio)
    x = torch.from_numpy(log_mel).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,mels,T)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    top_idx = int(np.argmax(probs))
    return EMOTIONS[top_idx], probs[top_idx], probs


def main():
    parser = argparse.ArgumentParser(description="Record once, predict emotion once.")
    parser.add_argument("--seconds", type=float, default=5.0,
                         help="Recording duration in seconds (default: 5)")
    args = parser.parse_args()

    print("Loading model...")
    model, device = load_model()

    input("Press Enter, then start speaking immediately...")
    audio = record_clip(args.seconds)

    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 0.003:
        print("\nWarning: very low audio energy detected — check your microphone "
              "is unmuted and picking up sound. Predicting anyway.\n")

    emotion, confidence, probs = predict_emotion(model, device, audio)

    print("\n" + "=" * 40)
    print(f"  PREDICTED EMOTION: {emotion.upper()}  ({confidence*100:.1f}% confidence)")
    print("=" * 40)
    print("\nFull breakdown:")
    for e, p in sorted(zip(EMOTIONS, probs), key=lambda pair: -pair[1]):
        bar = "#" * int(p * 40)
        print(f"  {e:10s} {p*100:5.1f}%  {bar}")


if __name__ == "__main__":
    main()