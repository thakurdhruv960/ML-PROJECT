"""
Audio preprocessing: load, resample, trim silence, normalize, pad/truncate,
and train-only augmentation (noise, pitch shift, time-stretch, time-shift).
"""
import numpy as np
import librosa

from config import SAMPLE_RATE, CLIP_LENGTH_SAMPLES


def load_audio(filepath, sr=SAMPLE_RATE):
    """Load an audio file, resampled to sr, mono."""
    audio, _ = librosa.load(filepath, sr=sr, mono=True)
    return audio


def trim_silence(audio, top_db=25):
    """Trim leading/trailing silence."""
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    # Guard against fully-silent clips being trimmed to nothing
    return trimmed if len(trimmed) > 0 else audio


def normalize_amplitude(audio):
    """Peak-normalize to [-1, 1]."""
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio / peak
    return audio.astype(np.float32)


def pad_or_truncate(audio, length=CLIP_LENGTH_SAMPLES):
    if len(audio) >= length:
        # Center crop
        start = (len(audio) - length) // 2
        return audio[start:start + length]
    pad_width = length - len(audio)
    left = pad_width // 2
    right = pad_width - left
    return np.pad(audio, (left, right), mode="constant")


def preprocess_clip(filepath):
    """Full preprocessing chain used for both train and inference (no augmentation)."""
    audio = load_audio(filepath)
    audio = trim_silence(audio)
    audio = normalize_amplitude(audio)
    audio = pad_or_truncate(audio)
    return audio


# ---------------------------------------------------------------------------
# Augmentation (apply ONLY to training split, never val/test)
# ---------------------------------------------------------------------------
def add_background_noise(audio, noise_factor_range=(0.001, 0.01)):
    noise_factor = np.random.uniform(*noise_factor_range)
    noise = np.random.randn(len(audio))
    return (audio + noise_factor * noise).astype(np.float32)


def pitch_shift(audio, sr=SAMPLE_RATE, semitone_range=(-2, 2)):
    n_steps = np.random.uniform(*semitone_range)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps).astype(np.float32)


def time_stretch(audio, rate_range=(0.9, 1.1)):
    rate = np.random.uniform(*rate_range)
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    return pad_or_truncate(stretched)  # stretching changes length, restore fixed size


def time_shift(audio, shift_max_frac=0.1):
    shift = int(len(audio) * np.random.uniform(-shift_max_frac, shift_max_frac))
    return np.roll(audio, shift).astype(np.float32)


def augment_clip(audio):
    """
    Randomly apply a subset of augmentations. Called only on training samples.
    Each augmentation has an independent chance of firing, so clips see varied
    combinations across epochs.

    Dialed back from the max-aggressive version (0.7/0.5/0.5/0.5) — that
    combo, stacked with higher dropout/weight-decay, over-corrected and
    dropped accuracy rather than improving it. Settling between the
    original (0.5/0.3/0.3/0.3) and that peak.
    """
    if np.random.rand() < 0.6:
        audio = add_background_noise(audio)
    if np.random.rand() < 0.4:
        audio = pitch_shift(audio)
    if np.random.rand() < 0.4:
        audio = time_stretch(audio)
    if np.random.rand() < 0.4:
        audio = time_shift(audio)
    return normalize_amplitude(audio)
