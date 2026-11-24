"""
Central configuration for the Speech Emotion Recognition project.
Edit paths here if your folder layout differs.
"""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

RAVDESS_DIR = os.path.join(DATA_RAW_DIR, "RAVDESS")
CREMA_D_DIR = os.path.join(DATA_RAW_DIR, "CREMA-D")
TESS_DIR = os.path.join(DATA_RAW_DIR, "TESS")
SAVEE_DIR = os.path.join(DATA_RAW_DIR, "SAVEE")

MANIFEST_CSV = os.path.join(DATA_PROCESSED_DIR, "manifest.csv")
FEATURES_DIR = os.path.join(DATA_PROCESSED_DIR, "features")

# ---------------------------------------------------------------------------
# Unified label set
# ---------------------------------------------------------------------------
# Every dataset's native labels get mapped onto this common set.
# RAVDESS "calm" AND all "sad" samples fold into "neutral" (low-arousal
# emotions collapsed into one class — this was the worst-confused pair in
# the 7-class version, so merging them directly removes that error mode
# and increases support for neutral).
EMOTIONS = ["neutral", "happy", "angry", "fear", "disgust", "surprise"]
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}
IDX_TO_EMOTION = {i: e for e, i in EMOTION_TO_IDX.items()}
NUM_CLASSES = len(EMOTIONS)

# ---------------------------------------------------------------------------
# Audio / feature extraction
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
CLIP_DURATION_SEC = 3.5
CLIP_LENGTH_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION_SEC)

N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256          # -> ~137 time frames for a 3.5s clip at 16kHz
WIN_LENGTH = 1024

# ---------------------------------------------------------------------------
# Train / val / test split (speaker-independent)
# ---------------------------------------------------------------------------
TEST_SPEAKER_FRACTION = 0.20
VAL_SPEAKER_FRACTION = 0.15   # raised from 0.10 — 12 val speakers was too few, val_acc
                               # was swinging 15-19 points epoch to epoch on pure noise
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Model / training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
NUM_EPOCHS = 60
LEARNING_RATE = 2e-4     # slightly lower — val curve was jumpy, not just noisy
WEIGHT_DECAY = 2e-4      # dialed back from 5e-4 — that was overcorrecting, dropped
                          # accuracy 65.7% -> 63.9% -> 61% over the last two rounds
LSTM_HIDDEN = 128
LSTM_LAYERS = 2
DROPOUT = 0.4            # dialed back from 0.5, still above the original 0.35
LABEL_SMOOTHING = 0.1    # softens overconfident wrong predictions
VAL_LOSS_EMA_ALPHA = 0.3 # smooths the noisy per-epoch val_loss for checkpoint
                          # selection/early stopping, instead of reacting to noise
EARLY_STOP_PATIENCE = 12

DEVICE = "cuda"  # train.py falls back to cpu automatically if unavailable

# ---------------------------------------------------------------------------
# wav2vec2 fine-tuning (alternative to the from-scratch CNN-BiLSTM above)
# ---------------------------------------------------------------------------
WAV2VEC2_MODEL_NAME = "facebook/wav2vec2-base"
WAV2VEC2_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, "wav2vec2")
WAV2VEC2_BATCH_SIZE = 8       # wav2vec2-base is far larger than the CNN-BiLSTM;
                               # lower this further if you hit CUDA out-of-memory
WAV2VEC2_LEARNING_RATE = 3e-5 # fine-tuning needs a much smaller LR than
                               # from-scratch training
WAV2VEC2_WEIGHT_DECAY = 1e-4
WAV2VEC2_NUM_EPOCHS = 15      # pretrained models converge in far fewer epochs
WAV2VEC2_EARLY_STOP_PATIENCE = 5
WAV2VEC2_FREEZE_FEATURE_ENCODER = True
