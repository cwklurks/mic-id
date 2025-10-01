#!/usr/bin/env python3
"""Quick CLI to score audio clips with the trained Mic-ID model."""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

import joblib
import librosa
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
CACHE_ROOT = BASE_DIR / ".cache"
NUMBA_CACHE_DIR = CACHE_ROOT / "numba"
MPL_CACHE_DIR = CACHE_ROOT / "matplotlib"
for path in (NUMBA_CACHE_DIR, MPL_CACHE_DIR):
    path.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

from features import extract_features
from devices import describe_label

MODEL_PATH = Path("models/model.pkl")
ENCODER_PATH = Path("models/label_encoder.pkl")


def load_model():
    if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
        raise SystemExit("Trained artefacts not found. Run `python train.py` first.")
    clf = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    return clf, le


def load_audio(path: Path, sr: int = 16000) -> tuple[np.ndarray, int]:
    if path.suffix.lower() == ".wav":
        y, sr = librosa.load(path, sr=sr, mono=True)
        return y, sr
    # fall back to BytesIO so we also support .mp3/.m4a just like the Streamlit app
    with path.open("rb") as f:
        data = io.BytesIO(f.read())
    y, sr = librosa.load(data, sr=sr, mono=True)
    return y, sr


def normalise_audio(y: np.ndarray) -> np.ndarray:
    rms = float(np.sqrt(np.mean(y**2)) + 1e-8)
    return y * (0.05 / rms), rms


def main() -> None:
    parser = argparse.ArgumentParser(description="Score WAV/MP3/M4A clips with the Mic-ID classifier.")
    parser.add_argument("paths", nargs="+", type=Path, help="Audio files to score")
    parser.add_argument("--topk", type=int, default=3, help="How many ranked predictions to show per file")
    args = parser.parse_args()

    clf, le = load_model()
    topk = max(1, min(args.topk, len(le.classes_)))

    for path in args.paths:
        if not path.exists():
            print(f"[!] Skipping missing file: {path}")
            continue
        try:
            y, sr = load_audio(path)
        except Exception as exc:  # pragma: no cover - friendly CLI message
            print(f"[!] Failed to load {path}: {exc}")
            continue
        y, rms = normalise_audio(y)
        feats = extract_features(y, sr).reshape(1, -1)
        proba = clf.predict_proba(feats)[0]
        order = np.argsort(proba)[::-1]
        print(f"\nFile: {path}")
        print(f"RMS loudness: {20 * np.log10(rms + 1e-12):.1f} dBFS")
        for rank, idx in enumerate(order[:topk], start=1):
            label = le.classes_[idx]
            friendly = describe_label(label)
            print(f"  {rank}. {friendly} — {proba[idx] * 100:.1f}%")


if __name__ == "__main__":
    main()
