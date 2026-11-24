"""
Log-mel spectrogram feature extraction.

Output shape per clip: (N_MELS, T) e.g. (128, ~137) for a 3.5s clip at 16kHz
with hop_length=256. This 2D "image" is what the CNN front-end consumes.
"""
import numpy as np
import librosa

from config import SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH, WIN_LENGTH


def extract_log_mel(audio, sr=SAMPLE_RATE):
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    # Per-clip standardization (zero mean, unit variance) helps convergence
    mean, std = log_mel.mean(), log_mel.std()
    log_mel = (log_mel - mean) / (std + 1e-8)
    return log_mel.astype(np.float32)


def spec_augment(log_mel, num_freq_masks=2, freq_mask_width=12,
                  num_time_masks=2, time_mask_width=15):
    """
    SpecAugment: randomly zero out horizontal (frequency) and vertical (time)
    bands directly on the spectrogram. Forces the model to rely on the whole
    time-frequency pattern rather than memorizing a few distinctive bands or
    frames from a specific clip — the standard fix for overfitting in
    spectrogram-based models. Apply AFTER extract_log_mel, training data only.
    """
    augmented = log_mel.copy()
    n_mels, n_time = augmented.shape

    for _ in range(num_freq_masks):
        width = np.random.randint(0, freq_mask_width + 1)
        if width == 0 or width >= n_mels:
            continue
        start = np.random.randint(0, n_mels - width)
        augmented[start:start + width, :] = 0.0

    for _ in range(num_time_masks):
        width = np.random.randint(0, time_mask_width + 1)
        if width == 0 or width >= n_time:
            continue
        start = np.random.randint(0, n_time - width)
        augmented[:, start:start + width] = 0.0

    return augmented
