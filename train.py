import os
import glob
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CACHE_ROOT = BASE_DIR / ".cache"
NUMBA_CACHE_DIR = CACHE_ROOT / "numba"
MPL_CACHE_DIR = CACHE_ROOT / "matplotlib"
for path in (NUMBA_CACHE_DIR, MPL_CACHE_DIR):
    path.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

from features import load_mono, extract_features

DATA_DIR, MODEL_DIR, REPORT_DIR = "data", "models", "reports"
os.makedirs(MODEL_DIR, exist_ok=True); os.makedirs(REPORT_DIR, exist_ok=True)

IGNORED_DEVICES = {"outtakes"}
SUFFIX_TO_DEVICE = {
    "a": "audio",
    "b": "audio2",
    "c": "audio9",
}
TAU_DEVICE_DIRS = set(SUFFIX_TO_DEVICE.values())


def resolve_device_label(device_dir: str, wav_path: str) -> str:
    """Infer the correct device label for a wav file.

    TAU scenes live under per-device directories but each folder still contains
    the parallel `-a/-b/-c` recordings. Instead of trusting the directory name
    (which mislabels the clips), derive the device from the filename suffix and
    fall back to the directory label for any locally recorded additions that
    do not follow that convention.
    """

    if device_dir in TAU_DEVICE_DIRS:
        stem = Path(wav_path).stem
        if "-" in stem:
            _, suffix = stem.rsplit("-", 1)
            if suffix in SUFFIX_TO_DEVICE:
                return SUFFIX_TO_DEVICE[suffix]
    return device_dir


def load_dataset():
    X, y = [], []
    seen: set[tuple[str, str]] = set()
    for device in sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
        and not d.startswith(".")
        and d not in IGNORED_DEVICES
    ):
        for wav in glob.glob(os.path.join(DATA_DIR, device, "*.wav")):
            label = resolve_device_label(device, wav)
            key = (os.path.basename(wav), label)
            if key in seen:
                continue
            seen.add(key)
            x, sr = load_mono(wav); feats = extract_features(x, sr)
            X.append(feats); y.append(label)
    return np.array(X), np.array(y)


if __name__ == "__main__":
    X, y = load_dataset()
    le = LabelEncoder(); y_enc = le.fit_transform(y)
    Xtr, Xte, ytr, yte = train_test_split(X, y_enc, test_size=0.25, stratify=y_enc, random_state=42)

    clf = HistGradientBoostingClassifier(max_depth=10, max_iter=400, learning_rate=0.08, random_state=42)
    clf.fit(Xtr, ytr); yhat = clf.predict(Xte)

    report = classification_report(yte, yhat, target_names=le.classes_, output_dict=True)
    with open(os.path.join(REPORT_DIR, "metrics.json"), "w") as f: json.dump(report, f, indent=2)

    cm = confusion_matrix(yte, yhat, normalize="true")
    fig, ax = plt.subplots(figsize=(5,4)); im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(le.classes_))); ax.set_xticklabels(le.classes_, rotation=45, ha="right")
    ax.set_yticks(range(len(le.classes_))); ax.set_yticklabels(le.classes_)
    for i in range(len(le.classes_)):
        for j in range(len(le.classes_)):
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Confusion (normalized)"); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"), dpi=160)

    if hasattr(clf, "_feature_subsample_rng"):
        clf._feature_subsample_rng = None

    joblib.dump(clf, os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(le,  os.path.join(MODEL_DIR, "label_encoder.pkl"))
    print("Saved model + reports.")
