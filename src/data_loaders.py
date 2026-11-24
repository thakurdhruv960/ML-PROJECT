"""
Per-dataset loaders.

Each dataset encodes emotion (and speaker) in its filenames differently.
Each loader below walks its dataset's raw folder and returns a list of dicts:
    {filepath, emotion, source_dataset, actor_id, gender}

`emotion` is already mapped onto the unified label set in config.EMOTIONS.
Rows whose native label has no place in the unified set are dropped.
"""
import os
import glob
from config import RAVDESS_DIR, CREMA_D_DIR, TESS_DIR, SAVEE_DIR, EMOTIONS


def _row(filepath, emotion, source, actor_id, gender):
    if emotion not in EMOTIONS:
        return None
    return {
        "filepath": filepath,
        "emotion": emotion,
        "source_dataset": source,
        "actor_id": actor_id,
        "gender": gender,
    }


# ---------------------------------------------------------------------------
# RAVDESS
# Filename: 03-01-06-01-02-01-12.wav
#           modality-vocalChannel-EMOTION-intensity-statement-repetition-ACTOR
# Emotion codes: 01 neutral, 02 calm, 03 happy, 04 sad, 05 angry,
#                06 fearful, 07 disgust, 08 surprised
# Actor: odd = male, even = female. Folder layout: Actor_01/.../*.wav
# ---------------------------------------------------------------------------
RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "neutral",   # calm folded into neutral
    "03": "happy",
    "04": "neutral",   # sad folded into neutral
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}


def load_ravdess(root_dir=RAVDESS_DIR):
    rows = []
    for filepath in glob.glob(os.path.join(root_dir, "**", "*.wav"), recursive=True):
        name = os.path.basename(filepath).replace(".wav", "")
        parts = name.split("-")
        if len(parts) != 7:
            continue
        emotion_code = parts[2]
        actor_num = int(parts[6])
        emotion = RAVDESS_EMOTION_MAP.get(emotion_code)
        if emotion is None:
            continue
        gender = "male" if actor_num % 2 == 1 else "female"
        actor_id = f"RAVDESS_{actor_num:02d}"
        row = _row(filepath, emotion, "RAVDESS", actor_id, gender)
        if row:
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# CREMA-D
# Filename: 1001_DFA_ANG_XX.wav  -> ActorID_Sentence_EMOTION_Level
# Emotion codes: ANG, DIS, FEA, HAP, NEU, SAD  (no surprise in this dataset)
# ---------------------------------------------------------------------------
CREMA_D_EMOTION_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "neutral",   # sad folded into neutral
}


def load_crema_d(root_dir=CREMA_D_DIR):
    rows = []
    for filepath in glob.glob(os.path.join(root_dir, "**", "*.wav"), recursive=True):
        name = os.path.basename(filepath).replace(".wav", "")
        parts = name.split("_")
        if len(parts) < 3:
            continue
        actor_num, _sentence, emotion_code = parts[0], parts[1], parts[2]
        emotion = CREMA_D_EMOTION_MAP.get(emotion_code)
        if emotion is None:
            continue
        actor_id = f"CREMAD_{actor_num}"
        # Gender is only knowable from CREMA-D's VideoDemographics.csv;
        # left unknown here since it doesn't affect training, only optional EDA.
        row = _row(filepath, emotion, "CREMA-D", actor_id, "unknown")
        if row:
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# TESS
# Folder/filename pattern: OAF_back_angry.wav / YAF_back_happy.wav
# Two speakers only: OAF (older adult female), YAF (young adult female)
# Emotions in filename (last underscore-separated token): angry, disgust,
# fear, happy, neutral, ps (pleasant surprise), sad
# ---------------------------------------------------------------------------
TESS_EMOTION_MAP = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "neutral",   # sad folded into neutral
    "ps": "surprise",
    "pleasant_surprise": "surprise",
    "surprise": "surprise",
}


def load_tess(root_dir=TESS_DIR):
    rows = []
    for filepath in glob.glob(os.path.join(root_dir, "**", "*.wav"), recursive=True):
        name = os.path.basename(filepath).replace(".wav", "").lower()
        tokens = name.split("_")
        if len(tokens) < 2:
            continue
        speaker_tag = tokens[0]  # 'oaf' or 'yaf'
        emotion_token = tokens[-1]
        emotion = TESS_EMOTION_MAP.get(emotion_token)
        if emotion is None:
            continue
        actor_id = f"TESS_{speaker_tag.upper()}"
        row = _row(filepath, emotion, "TESS", actor_id, "female")
        if row:
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# SAVEE
# Filename: DC_a01.wav, DC_sa03.wav, DC_su02.wav ...
#           SPEAKER_EMOTIONCODE+number.wav
# Speakers: DC, JE, JK, KL (all male)
# Emotion codes (check 2-letter codes before 1-letter, 'sa'/'su' vs 'a'):
#   a=anger, d=disgust, f=fear, h=happiness, n=neutral, sa=sadness, su=surprise
# ---------------------------------------------------------------------------
SAVEE_EMOTION_MAP = {
    "sa": "neutral",   # sad folded into neutral
    "su": "surprise",
    "a": "angry",
    "d": "disgust",
    "f": "fear",
    "h": "happy",
    "n": "neutral",
}
# Longer codes must be matched first to avoid 'sa' being read as 's' + 'a'
SAVEE_CODES_BY_LENGTH = sorted(SAVEE_EMOTION_MAP.keys(), key=len, reverse=True)


def _parse_savee_emotion(code_part):
    for code in SAVEE_CODES_BY_LENGTH:
        if code_part.startswith(code):
            return SAVEE_EMOTION_MAP[code]
    return None


def load_savee(root_dir=SAVEE_DIR):
    rows = []
    for filepath in glob.glob(os.path.join(root_dir, "**", "*.wav"), recursive=True):
        name = os.path.basename(filepath).replace(".wav", "")
        if "_" not in name:
            continue
        speaker_tag, rest = name.split("_", 1)
        # strip trailing digits to isolate the emotion code, e.g. 'sa03' -> 'sa'
        code_part = "".join(ch for ch in rest if not ch.isdigit()).lower()
        emotion = _parse_savee_emotion(code_part)
        if emotion is None:
            continue
        actor_id = f"SAVEE_{speaker_tag.upper()}"
        row = _row(filepath, emotion, "SAVEE", actor_id, "male")
        if row:
            rows.append(row)
    return rows


LOADERS = {
    "RAVDESS": load_ravdess,
    "CREMA-D": load_crema_d,
    "TESS": load_tess,
    "SAVEE": load_savee,
}
