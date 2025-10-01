import io
import os
import tempfile
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CACHE_ROOT = BASE_DIR / ".cache"
NUMBA_CACHE_DIR = CACHE_ROOT / "numba"
MPL_CACHE_DIR = CACHE_ROOT / "matplotlib"
for cache_dir in (NUMBA_CACHE_DIR, MPL_CACHE_DIR):
    cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

from features import extract_features
from devices import describe_label


NOTE_NAMES = [
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
]
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"
UPLOAD_DIR.mkdir(exist_ok=True)


def estimate_scale(y: np.ndarray, sr: int) -> str | None:
    """Return a rough musical scale (e.g., 'C major') or None if unclear."""

    if y.size == 0:
        return None
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    if chroma.size == 0:
        return None
    chroma_mean = chroma.mean(axis=1)
    norm = np.linalg.norm(chroma_mean, ord=1)
    if norm == 0:
        return None
    chroma_mean = chroma_mean / norm

    major_scores = [float(np.dot(chroma_mean, np.roll(MAJOR_PROFILE, i))) for i in range(12)]
    minor_scores = [float(np.dot(chroma_mean, np.roll(MINOR_PROFILE, i))) for i in range(12)]

    best_major = int(np.argmax(major_scores))
    best_minor = int(np.argmax(minor_scores))
    best_major_score = major_scores[best_major]
    best_minor_score = minor_scores[best_minor]
    best_score = max(best_major_score, best_minor_score)

    # Require a minimal tonal structure to avoid spurious guesses on noise.
    if best_score < 0.3:
        return None

    if best_major_score >= best_minor_score:
        return f"{NOTE_NAMES[best_major]} major"
    return f"{NOTE_NAMES[best_minor]} minor"

st.set_page_config(page_title="Mic-ID (MVP)", layout="centered")
st.title("Mic-ID (MVP)")
st.caption("Upload ~5s audio → guess the recording device (demo)")

with st.expander("Training data & devices", expanded=False):
    st.markdown(
        """
        - **TAU Urban Acoustic Scenes 2019 Mobile**: 295 parallel scenes where the same moment was captured on three devices – Zoom F8 (device A, clips ending in `-a`), Samsung Galaxy S7 (device B, `-b`), and iPhone SE (device C, `-c`). We only keep folders containing a full `-a/-b/-c` triplet, so each mic has 295 clips.
        - **Local additions**: 4 laptop and 4 iPhone recordings collected with `utils.py` to anchor the classifier on in-house gear.
        - **Features & model**: log-mel + MFCC statistics flow into a histogram-based gradient boosting classifier tuned for this small balanced set.

        Want more coverage? Record new clips under `data/<device>/` or export outtakes with `scripts/export_outtakes.py` before retraining via `python train.py`.
        """
    )


@st.cache_resource
def load_model():
    try:
        clf = joblib.load(MODEL_PATH)
        le = joblib.load(ENCODER_PATH)
        return clf, le
    except Exception as exc:
        st.warning(f"Could not load trained artefacts: {exc}")
        return None, None


clf, le = load_model()
topk = None
if clf and le is not None:
    max_classes = max(1, len(le.classes_))
    default_topk = min(3, max_classes)
    topk = st.slider(
        "How many guesses should we list?",
        min_value=1,
        max_value=max_classes,
        value=default_topk,
        help="Slide right to show more of the lower-confidence device guesses.",
    )
    st.caption("The slider above only changes how many ranked predictions you see.")

file = st.file_uploader("Upload WAV/MP3/M4A", type=["wav","mp3","m4a"])

if file and clf and le is not None:
    data = file.read()
    original_name = Path(file.name or "upload").name
    renamed_name = f"hooks - {original_name}"
    saved_path = UPLOAD_DIR / renamed_name
    saved_path.write_bytes(data)
    st.caption(f"Saved a copy as `{saved_path}`.")
    try:
        y, sr = librosa.load(io.BytesIO(data), sr=16000, mono=True)
    except Exception:
        suffix = os.path.splitext(file.name or "upload")[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(data)
            tmp.flush()
            y, sr = librosa.load(tmp.name, sr=16000, mono=True)
    raw_y = y.copy()
    rms = np.sqrt(np.mean(raw_y**2)) + 1e-8
    scale = estimate_scale(raw_y, sr)
    y = raw_y * (0.05 / rms)  # simple RMS norm
    feats = extract_features(y, 16000).reshape(1, -1)
    proba = clf.predict_proba(feats)[0]
    idx = np.argsort(proba)[::-1]
    st.subheader("Prediction")
    if scale:
        st.write(f"Estimated scale: **{scale}** (experimental)")
    else:
        st.write("Scale detection: the clip lacked clear musical content, so no scale estimate.")
    st.write(f"Input loudness (RMS): {20 * np.log10(rms + 1e-12):.1f} dBFS")
    limit = topk or 3
    for i in idx[:limit]:
        label = le.classes_[i]
        st.write(f"{describe_label(label)} — **{proba[i] * 100:.1f}%**")
    friendly_index = [describe_label(label) for label in le.classes_]
    st.bar_chart(pd.Series(proba, index=friendly_index))

    with st.expander("How the model listens", expanded=False):
        st.markdown(
            "We tidy the audio (level it, pull out key frequencies) and let the classifier score that summary. These charts show the raw waveform and the energy heatmap the model uses to decide." 
        )

        duration = raw_y.size / sr if raw_y.size else 0
        times = np.linspace(0.0, duration, num=raw_y.size, endpoint=False) if raw_y.size else np.array([])

        fig_wave, ax_wave = plt.subplots(figsize=(6, 2))
        if raw_y.size:
            ax_wave.plot(times, raw_y, linewidth=0.8, color="#1f77b4")
            ax_wave.set_xlim(0, max(times) if raw_y.size else 0)
        ax_wave.set_title("Waveform (time vs amplitude)")
        ax_wave.set_xlabel("Time (s)")
        ax_wave.set_ylabel("Amplitude")
        ax_wave.grid(alpha=0.2)
        st.pyplot(fig_wave, use_container_width=True)
        plt.close(fig_wave)

        mel = librosa.feature.melspectrogram(y=raw_y, sr=sr, n_fft=2048, hop_length=512, n_mels=64)
        mel_db = librosa.power_to_db(mel, ref=np.max) if mel.size else mel
        fig_spec, ax_spec = plt.subplots(figsize=(6, 3))
        if mel.size:
            img = librosa.display.specshow(mel_db, sr=sr, hop_length=512, x_axis="time", y_axis="mel", ax=ax_spec)
            cbar = fig_spec.colorbar(img, ax=ax_spec, format="%+2.f dB")
            cbar.set_label("Energy (dB)")
        ax_spec.set_title("Log-mel spectrogram (what the model summarises)")
        st.pyplot(fig_spec, use_container_width=True)
        plt.close(fig_spec)
elif file and not clf:
    st.warning("No trained model found. Run `python train.py` first.")
