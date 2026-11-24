"""
Speaker-independent split + PyTorch Dataset.

Splitting by actor_id (not randomly per-clip) is the single most important
correctness step in this pipeline: if the same speaker appears in both train
and test, the model can learn to recognize *voices* instead of *emotions*,
and your reported accuracy will be inflated and won't hold up in a viva or
on the live mic demo.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import (
    MANIFEST_CSV, EMOTION_TO_IDX, TEST_SPEAKER_FRACTION,
    VAL_SPEAKER_FRACTION, RANDOM_SEED,
)
from preprocessing import preprocess_clip, augment_clip
from features import extract_log_mel, spec_augment


def speaker_split(manifest_df, test_frac=TEST_SPEAKER_FRACTION,
                   val_frac=VAL_SPEAKER_FRACTION, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    # Force a plain numpy object array before shuffling — pandas' StringArray
    # is not guaranteed to shuffle correctly in-place with numpy's Generator
    # (can silently duplicate entries), which would risk real speaker leakage.
    speakers = np.array(manifest_df["actor_id"].unique().tolist(), dtype=object)
    rng.shuffle(speakers)

    n = len(speakers)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))

    test_speakers = set(speakers[:n_test])
    val_speakers = set(speakers[n_test:n_test + n_val])
    train_speakers = set(speakers[n_test + n_val:])

    # Safety check: confirm no speaker landed in more than one split
    assert not (test_speakers & val_speakers), "Speaker leak: test/val overlap"
    assert not (test_speakers & train_speakers), "Speaker leak: test/train overlap"
    assert not (val_speakers & train_speakers), "Speaker leak: val/train overlap"
    assert len(test_speakers) + len(val_speakers) + len(train_speakers) == n, \
        "Speaker count mismatch after split — possible duplicate/dropped speaker"

    train_df = manifest_df[manifest_df["actor_id"].isin(train_speakers)].reset_index(drop=True)
    val_df = manifest_df[manifest_df["actor_id"].isin(val_speakers)].reset_index(drop=True)
    test_df = manifest_df[manifest_df["actor_id"].isin(test_speakers)].reset_index(drop=True)

    return train_df, val_df, test_df


class SERDataset(Dataset):
    """
    train=True  -> applies waveform augmentation before feature extraction
    train=False -> deterministic preprocessing only (val/test/inference)
    """

    def __init__(self, dataframe, train=False):
        self.df = dataframe.reset_index(drop=True)
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio = preprocess_clip(row["filepath"])
        if self.train:
            audio = augment_clip(audio)
        log_mel = extract_log_mel(audio)              # (n_mels, T)
        if self.train:
            log_mel = spec_augment(log_mel)            # mask random time/freq bands
        x = torch.from_numpy(log_mel).unsqueeze(0)     # (1, n_mels, T) -> CNN channel dim
        y = EMOTION_TO_IDX[row["emotion"]]
        return x, y


def load_manifest():
    return pd.read_csv(MANIFEST_CSV)


def get_class_weights(train_df):
    """Inverse-frequency class weights for the loss function, to counter
    the imbalance between datasets (e.g. CREMA-D has no 'surprise')."""
    from config import EMOTIONS
    counts = train_df["emotion"].value_counts()
    weights = []
    for emo in EMOTIONS:
        c = counts.get(emo, 0)
        weights.append(1.0 / c if c > 0 else 0.0)
    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.sum() * len(weights)  # normalize
    return torch.tensor(weights, dtype=torch.float32)
