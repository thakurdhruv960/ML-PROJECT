# Speech Emotion Recognition (SER)

A speech emotion recognition system that classifies a speaker's emotional state from a short audio clip. The project was developed in three phases, moving from classical machine learning to a from-scratch deep learning model, and finally to fine-tuning a pretrained speech transformer (wav2vec2). It is evaluated on a merged multi-corpus dataset with a strict speaker-independent split, and includes a live microphone demo that runs fully offline.

**Final result:** 74.9% test accuracy (macro-F1 0.728) on six emotion classes, on speakers never seen during training.

---

## 1. Problem Statement

Given a short speech recording, predict which emotion the speaker is expressing, using only the acoustic signal (pitch, energy, rhythm, timbre) and not the words spoken.

**Emotion classes (6):** `neutral`, `happy`, `angry`, `fear`, `disgust`, `surprise`

The label "sad" is deliberately merged into "neutral", along with RAVDESS's "calm". In early experiments, sad and neutral were the most heavily confused pair (sad recall was under 10%), because both are low-arousal speech. Merging them removed the dominant error mode.

---

## 2. Project Evolution

| Phase | Approach | Features | Outcome |
|---|---|---|---|
| 1 | SVM, then Random Forest | MFCC (averaged over time) | Working baseline and live demo; single corpus (RAVDESS only) |
| 2 | CNN-BiLSTM with attention, trained from scratch | Log-mel spectrograms | About 61-66% on the harder multi-corpus, speaker-independent setup |
| 3 | Fine-tuned wav2vec2-base (pretrained) | Raw waveform | **74.9%** on the same setup |

**Why each transition happened**

- **Phase 1 to 2:** averaging MFCCs over time discards how emotion evolves within an utterance, and a single clean corpus is an easy, unrealistic test. Phase 2 preserves temporal structure and generalises across four corpora.
- **Phase 2 to 3:** extensive tuning of the from-scratch model plateaued in the mid-60s. With only about 120 speakers, a model learning audio structure from zero is data-limited. A pretrained model already encodes prosody and speaking style from tens of thousands of hours of speech, so only the emotion mapping has to be learned.

---

## 3. Datasets

Four public corpora are merged into one manifest. Each encodes labels differently in its filenames, so each has its own parser.

| Dataset | Speakers | Notes |
|---|---|---|
| RAVDESS | 24 actors | Studio quality, acted emotions |
| CREMA-D | 91 actors | Diverse ages and ethnicities, crowd-rated, largest corpus |
| TESS | 2 speakers | Very clean, female speakers only |
| SAVEE | 4 speakers | Male British speakers |

Download links:

- RAVDESS: https://zenodo.org/records/1188976
- CREMA-D: https://www.kaggle.com/datasets/ejlok1/cremad
- TESS: https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess
- SAVEE: https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee

Extract each into `data/raw/<DatasetName>/`. Subfolder nesting does not matter, but filenames must remain unchanged because they encode the labels.

---

## 4. Methodology

### 4.1 Speaker-independent splitting
The split is made by speaker, not by clip. If the same person's voice appeared in both train and test, the model could learn to recognise voices instead of emotions and report inflated accuracy. Roughly 65% of speakers go to training, 15% to validation and 20% to testing.

### 4.2 Preprocessing
Audio is resampled to 16 kHz mono, trimmed of leading and trailing silence, peak-normalised, and padded or truncated to a fixed 3.5-second window.

### 4.3 Augmentation (training split only)
- Background noise injection
- Pitch shifting
- Time stretching
- Time shifting
- SpecAugment (random time and frequency masking) for the spectrogram model

### 4.4 Models

**CNN-BiLSTM with attention (from scratch, about 2.7M parameters)**

```
Log-mel spectrogram (1 x 128 x ~137)
  -> 3 x [Conv2D -> BatchNorm -> ReLU -> MaxPool]   local time-frequency patterns
  -> reshape to a sequence over time
  -> Bidirectional LSTM (2 layers, 128 hidden)       temporal dynamics
  -> Attention pooling                               weights emotionally salient frames
  -> Dropout -> Dense -> Softmax (6 classes)
```

**wav2vec2 fine-tuning (pretrained, about 95M parameters)**

`facebook/wav2vec2-base` with the convolutional feature encoder frozen. The transformer layers and a newly initialised classification head are fine-tuned with a small learning rate (3e-5) for a few epochs.

### 4.5 Training details
- Class-weighted cross-entropy (inverse frequency) to offset dataset imbalance
- Label smoothing (0.1)
- Validation loss smoothed with an exponential moving average for checkpoint selection, because a small validation speaker pool made raw per-epoch values noisy
- Early stopping and learning-rate scheduling

---

## 5. Results

Held-out test set: 4,873 clips from speakers never seen in training.

**Fine-tuned wav2vec2**

| Emotion | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| neutral | 0.930 | 0.805 | 0.863 | 1465 |
| happy | 0.606 | 0.430 | 0.503 | 732 |
| angry | 0.832 | 0.630 | 0.717 | 732 |
| fear | 0.578 | 0.839 | 0.685 | 732 |
| disgust | 0.865 | 0.857 | 0.861 | 732 |
| surprise | 0.608 | 0.942 | 0.739 | 480 |
| **accuracy** | | | **0.749** | 4873 |
| macro avg | 0.736 | 0.750 | 0.728 | 4873 |

**Observations**

- Neutral, disgust and angry are classified reliably.
- Happy is the weakest class in every phase. It overlaps acoustically with other high-arousal emotions, and its delivery style differs across the four corpora.
- Surprise has high recall but lower precision, so the model over-predicts it slightly.
- The gap between training and validation accuracy remains (about 92% vs 72%), so the model still overfits to some degree.

A realistic ceiling for multi-corpus, speaker-independent SER with this much data is roughly 70-80%. Figures above 90% in the literature typically come from single-corpus evaluation or from splits where speakers leak between train and test.

---

## 6. Engineering Lessons Learned

**Overfitting diagnosis.** The first CNN-BiLSTM run showed 82.7% training vs 62% validation accuracy. Dropout, weight decay, stronger augmentation and SpecAugment were added. Applying all four at once overcorrected and lowered accuracy, so they were later dialled back to moderate values. Changing one thing at a time would have been clearer.

**Speaker-leakage bug.** A pandas `StringArray` shuffle warning turned out to be a real correctness risk: the in-place shuffle of speaker IDs was not guaranteed to behave correctly and could duplicate entries across splits. The speaker list is now converted to a plain NumPy array before shuffling, and assertions fail loudly if any speaker lands in more than one split.

**Data-loading bottleneck.** Even with GPU training, epochs were slow because librosa feature extraction ran on the CPU for every clip. More DataLoader workers, pinned memory and persistent workers helped. Caching spectrograms would help further.

---

## 7. Project Structure

```
ser_project/
  README.md
  requirements.txt
  data/
    raw/                       # RAVDESS, CREMA-D, TESS, SAVEE (not needed for the demo)
    processed/manifest.csv     # unified index built by build_manifest.py
  checkpoints/
    best_model.pt              # CNN-BiLSTM weights
    wav2vec2/best_model.pt     # fine-tuned wav2vec2 weights
  src/
    config.py                  # paths, label set, hyperparameters
    data_loaders.py            # per-dataset filename parsers
    build_manifest.py          # merge datasets, print EDA, write manifest
    preprocessing.py           # trim, normalise, pad, augment
    features.py                # log-mel extraction and SpecAugment
    dataset.py                 # speaker-independent split and Dataset (CNN-BiLSTM)
    model.py                   # CNN-BiLSTM-Attention
    train.py / evaluate.py     # train and evaluate the CNN-BiLSTM
    wav2vec2_dataset.py        # raw-waveform Dataset
    wav2vec2_model.py          # wav2vec2 wrapper
    train_wav2vec2.py          # fine-tuning loop
    evaluate_wav2vec2.py       # wav2vec2 evaluation
    ensemble.py                # average both models' probabilities
    evaluate_tta.py            # test-time augmentation
    train_kfold.py             # speaker-independent k-fold cross-validation
    live_inference.py          # continuous microphone streaming
    record_and_predict.py      # record once, predict once (CNN-BiLSTM)
    record_and_predict_compare.py   # record once, compare both models side by side
```

---

## 8. Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

For GPU training, install the CUDA build of PyTorch (for example `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121`). Verify with:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 9. Usage

```powershell
python src\build_manifest.py        # build manifest.csv (check that no dataset shows 0 clips)
python src\train.py                 # train the CNN-BiLSTM
python src\evaluate.py              # test metrics and confusion matrix
python src\train_wav2vec2.py        # fine-tune wav2vec2 (downloads weights on first run)
python src\evaluate_wav2vec2.py     # test metrics for wav2vec2
python src\ensemble.py              # optional: combine both models
```

Training is done once. Evaluation and demo scripts only load the saved checkpoints.

**Live demo**

```powershell
python src\record_and_predict_compare.py      # 5 s recording, both models side by side
python src\record_and_predict_compare.py --seconds 7
python src\record_and_predict.py              # CNN-BiLSTM only
python src\live_inference.py                  # continuous streaming
```

---

## 10. Offline Deployment

Inference needs no internet and no dataset. Prepare these items while online:

| Item | Purpose |
|---|---|
| `ser_project/` (code and both checkpoints) | The project itself |
| `wheelhouse/` and `requirements-lock.txt` | Offline pip install |
| `huggingface_cache` | Pretrained wav2vec2 backbone (the checkpoint holds only the fine-tuned weights) |
| Python installer (exact version used to build the wheelhouse, 3.12) | Wheels are version-specific (`cp312`) |
| `vc_redist.x64.exe` | Runtime required by PyTorch on Windows |
| Optional: VS Code installer and the Python and Python Debugger `.vsix` files | Editor fallback |

**Build the wheelhouse (online machine)**

```powershell
pip freeze > requirements-lock.txt          # then remove the torch and torchaudio lines
pip download -r requirements-lock.txt -d wheelhouse
pip download torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu -d wheelhouse
```

CPU-only PyTorch is used so the demo does not depend on unknown GPU hardware or drivers. A single 5-second prediction runs comfortably on CPU.

**Install and run (offline machine)**

```powershell
# 1. Install Python (tick "Add python.exe to PATH") and vc_redist.x64.exe
pip install --no-index --find-links=wheelhouse -r requirements-lock.txt
pip install --no-index --find-links=wheelhouse torch torchaudio
xcopy huggingface_cache "C:\Users\<username>\.cache\huggingface" /E /I /H
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
python src\record_and_predict_compare.py
```

**Checklist**

- Check that the microphone is detected: `python -c "import sounddevice as sd; print(sd.query_devices())"`
- Set the intended mic as the Windows default input and allow microphone access under Privacy settings.
- If the machine has a different Python version, install 3.12 alongside it and call it with `py -3.12`.
- Rehearse the full sequence once on a second machine with Wi-Fi disabled.

---

## 11. Limitations

- Emotions are acted or semi-acted, so performance on spontaneous real-world speech will be lower.
- Total speaker diversity is small (about 120 speakers), which limits generalisation.
- "Happy" remains hard to separate from other high-arousal emotions.
- Merging sad into neutral removes the ability to detect sadness as its own class.
- A single train/validation/test split gives a noisy estimate, so k-fold results would be more reliable.

## 12. Future Work

- Ensemble the CNN-BiLSTM and wav2vec2 models and report the gain.
- Fine-tune larger pretrained models (wav2vec2-large, HuBERT) with more layers unfrozen.
- Add more naturalistic data such as IEMOCAP.
- Report mean and standard deviation from speaker-independent k-fold cross-validation.
- Cache spectrograms to speed up training.
- Package the model as a desktop or mobile application.

---

## References

- Livingstone & Russo (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).
- Cao et al. (2014). CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset.
- Dupuis & Pichora-Fuller (2010). Toronto Emotional Speech Set (TESS).
- Haq & Jackson. Surrey Audio-Visual Expressed Emotion (SAVEE) database.
- Baevski et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.
- Park et al. (2019). SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition.
