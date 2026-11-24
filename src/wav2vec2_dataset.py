"""
Dataset for wav2vec2/HuBERT fine-tuning.

Unlike SERDataset (which extracts log-mel spectrograms for the CNN-BiLSTM),
this returns raw waveforms — wav2vec2's own convolutional feature encoder
does its feature extraction internally, so we only need to hand it
correctly preprocessed raw audio.

Reuses the exact same trim/pad/augment functions as the CNN-BiLSTM pipeline
(preprocessing.py) so both approaches see equivalent underlying audio —
only the final normalization differs, matching what wav2vec2 was
pretrained to expect (zero mean, unit variance) rather than peak
normalization to [-1, 1].
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from config import EMOTION_TO_IDX
from preprocessing import preprocess_clip, augment_clip


def normalize_for_wav2vec2(audio):
    """Zero mean, unit variance — matches HuggingFace's Wav2Vec2FeatureExtractor
    do_normalize=True behavior, which is what these models were pretrained on."""
    mean = audio.mean()
    std = audio.std()
    return ((audio - mean) / (std + 1e-7)).astype(np.float32)


class Wav2Vec2SERDataset(Dataset):
    def __init__(self, dataframe, train=False):
        self.df = dataframe.reset_index(drop=True)
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio = preprocess_clip(row["filepath"])   # trim/peak-normalize/pad to fixed length
        if self.train:
            audio = augment_clip(audio)
        audio = normalize_for_wav2vec2(audio)
        x = torch.from_numpy(audio)                 # (samples,) — no channel dim needed
        y = EMOTION_TO_IDX[row["emotion"]]
        return x, y
