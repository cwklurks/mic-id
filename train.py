from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

BASE_DIR = Path(__file__).resolve().parent
CACHE_ROOT = BASE_DIR / ".cache"
NUMBA_CACHE_DIR = CACHE_ROOT / "numba"
MPL_CACHE_DIR = CACHE_ROOT / "matplotlib"
for path in (NUMBA_CACHE_DIR, MPL_CACHE_DIR):
    path.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import joblib
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from features import extract_features, load_mono

TARGET_SR = 16000
REQUIRED_COLUMNS = {"path", "device", "source", "license", "split", "sha256"}
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class ClipRecord:
    path: Path
    device: str
    source: str
    license: str
    split: str
    sha256: str

    def relative_path(self, root: Path) -> str:
        return self.path.relative_to(root).as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Mic-ID classifier with provenance tracking.")
    parser.add_argument("--config", default="configs/base.yaml", help="YAML config describing data + training parameters.")
    parser.add_argument("--dry-run", action="store_true", help="Validate metadata and show dataset summary without training.")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if "data" not in cfg or "training" not in cfg or "reporting" not in cfg:
        raise SystemExit("Config must include `data`, `training`, and `reporting` sections.")
    return cfg


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_metadata_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        headers = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - headers
        if missing:
            raise SystemExit(f"Metadata file {path} is missing required columns: {sorted(missing)}")
        return list(reader)


def load_clip_records(data_cfg: dict) -> tuple[list[ClipRecord], Path, Path]:
    root = Path(data_cfg.get("root", "data")).resolve()
    metadata_path = Path(data_cfg.get("metadata", root / "metadata.csv")).resolve()
    enforce_hashes = bool(data_cfg.get("enforce_hashes", True))
    splits_filter = set(data_cfg.get("splits", []) or [])
    include_devices = set(data_cfg.get("include_devices", []) or [])

    if not root.exists():
        raise SystemExit(f"Data root does not exist: {root}")
    if not metadata_path.exists():
        raise SystemExit(f"Metadata file not found: {metadata_path}")

    raw_rows = read_metadata_csv(metadata_path)
    records: list[ClipRecord] = []
    seen: set[tuple[str, str]] = set()

    for idx, row in enumerate(raw_rows, start=2):
        rel_path = row["path"].strip()
        device = row["device"].strip()
        source = row["source"].strip()
        license_ = row["license"].strip()
        split = row["split"].strip() or "train"
        sha256 = row["sha256"].strip()

        if include_devices and device not in include_devices:
            continue
        if splits_filter and split not in splits_filter:
            continue

        if not rel_path:
            raise SystemExit(f"Row {idx} is missing a path.")
        if not device:
            raise SystemExit(f"Row {idx} is missing a device label (path={rel_path}).")
        if not source or not license_:
            raise SystemExit(f"Row {idx} missing source/license information (device={device}, path={rel_path}).")

        full_path = root / rel_path
        if not full_path.exists():
            raise SystemExit(f"Audio file referenced in metadata not found: {full_path}")

        if not sha256:
            current_hash = compute_sha256(full_path)
        else:
            current_hash = compute_sha256(full_path) if enforce_hashes else sha256
            if enforce_hashes and current_hash != sha256:
                raise SystemExit(
                    f"Hash mismatch for {rel_path}: metadata={sha256} current={current_hash}. "
                    "Regenerate metadata via scripts/refresh_metadata.py."
                )

        key = (rel_path, device)
        if key in seen:
            raise SystemExit(f"Duplicate clip/device combination detected in metadata: {rel_path} ({device})")
        seen.add(key)

        records.append(
            ClipRecord(
                path=full_path,
                device=device,
                source=source,
                license=license_,
                split=split,
                sha256=current_hash if enforce_hashes else current_hash,
            )
        )

    if include_devices:
        for dev in include_devices:
            if dev not in {record.device for record in records}:
                raise SystemExit(f"No clips found for requested device: {dev}")

    if not records:
        raise SystemExit("No audio clips passed the metadata filters; nothing to train on.")

    return records, root, metadata_path


def ensure_minimum_counts(records: Sequence[ClipRecord], minimum: int) -> Counter:
    counts = Counter(record.device for record in records)
    violations = {device: count for device, count in counts.items() if count < minimum}
    if violations:
        formatted = ", ".join(f"{dev} ({count})" for dev, count in violations.items())
        raise SystemExit(f"Not enough clips per device. Increase data or lower the threshold. Offenders: {formatted}")
    return counts


def summarise_records(records: Sequence[ClipRecord], root: Path) -> dict:
    counts = Counter(record.device for record in records)
    sources = {record.device: record.source for record in records}
    licenses = {record.device: record.license for record in records}
    return {
        "total_clips": len(records),
        "devices": dict(counts),
        "sources": sources,
        "licenses": licenses,
        "first_five_hashes": [
            {"path": record.relative_path(root), "sha256": record.sha256}
            for record in records[: min(5, len(records))]
        ],
    }


def collect_hashes(records: Sequence[ClipRecord], root: Path) -> list[dict]:
    return [
        {"path": record.relative_path(root), "sha256": record.sha256}
        for record in records
    ]


def build_dataset(records: Sequence[ClipRecord]) -> tuple[np.ndarray, np.ndarray]:
    features, labels = [], []
    for record in records:
        audio, sr = load_mono(record.path, sr=TARGET_SR)
        feats = extract_features(audio, sr)
        features.append(feats)
        labels.append(record.device)
    return np.array(features), np.array(labels)


def instantiate_classifier(cfg: dict) -> HistGradientBoostingClassifier:
    clf_cfg = dict(cfg.get("classifier", {}))
    random_state = cfg.get("random_state")
    if random_state is not None:
        clf_cfg.setdefault("random_state", random_state)
    if not clf_cfg:
        clf_cfg = {"max_depth": 10, "max_iter": 400, "learning_rate": 0.08}
        if random_state is not None:
            clf_cfg["random_state"] = random_state
    return HistGradientBoostingClassifier(**clf_cfg)


def plot_confusion_matrix(cm: np.ndarray, labels: Sequence[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Confusion (normalized)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_run_report(
    reporting_cfg: dict,
    config_path: Path,
    config: dict,
    records: Sequence[ClipRecord],
    root: Path,
    metrics: dict,
    dataset_summary: dict,
    hashes: Sequence[dict],
    model_path: Path,
    encoder_path: Path,
) -> Path:
    runs_dir = Path(reporting_cfg.get("runs_dir", REPORT_DIR / "runs")).resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    now_utc = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    timestamp = now_utc.strftime("%Y%m%d-%H%M%S")
    tag = reporting_cfg.get("tag")
    filename = f"run-{timestamp}"
    if tag:
        filename += f"-{tag}"
    run_path = runs_dir / f"{filename}.json"

    payload = {
        "timestamp_utc": now_utc.isoformat().replace("+00:00", "Z"),
        "config_path": str(config_path.resolve()),
        "config_snapshot": config,
        "dataset": {
            **dataset_summary,
            "metadata_root": str(root),
            "hashes": list(hashes),
        },
        "metrics": metrics,
        "artefacts": {
            "model": str(model_path),
            "label_encoder": str(encoder_path),
            "metrics_json": str(Path(reporting_cfg.get("metrics_path", REPORT_DIR / "metrics.json")).resolve()),
            "confusion_matrix": str(Path(reporting_cfg.get("confusion_matrix_path", REPORT_DIR / "confusion_matrix.png")).resolve()),
        },
    }

    with run_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return run_path


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)

    data_cfg = config["data"]
    training_cfg = config["training"]
    reporting_cfg = config["reporting"]

    records, data_root, metadata_path = load_clip_records(data_cfg)
    min_clips = int(data_cfg.get("min_clips_per_device", 1))
    ensure_minimum_counts(records, min_clips)
    dataset_summary = summarise_records(records, data_root)
    hashes = collect_hashes(records, data_root)
    dataset_summary["metadata_file"] = str(metadata_path)

    print("Dataset summary:")
    for key, value in dataset_summary.items():
        print(f"  {key}: {value}")

    if args.dry_run:
        print("Dry run complete. Exiting without training.")
        return

    X, y = build_dataset(records)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    test_size = float(training_cfg.get("test_size", 0.25))
    random_state = training_cfg.get("random_state", 42)
    stratify = training_cfg.get("stratify", True)
    stratify_arg = y_encoded if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        stratify=stratify_arg,
        random_state=random_state,
    )

    clf = instantiate_classifier(training_cfg)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    metrics_path = Path(reporting_cfg.get("metrics_path", REPORT_DIR / "metrics.json"))
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    cm = confusion_matrix(y_test, y_pred, normalize="true")
    confusion_path = Path(reporting_cfg.get("confusion_matrix_path", REPORT_DIR / "confusion_matrix.png"))
    plot_confusion_matrix(cm, label_encoder.classes_, confusion_path)

    # Clean up non-serializable RNG to keep joblib artefacts deterministic.
    if hasattr(clf, "_feature_subsample_rng"):
        clf._feature_subsample_rng = None

    model_path = MODEL_DIR / "model.pkl"
    encoder_path = MODEL_DIR / "label_encoder.pkl"
    joblib.dump(clf, model_path)
    joblib.dump(label_encoder, encoder_path)

    run_report_path = write_run_report(
        reporting_cfg,
        config_path,
        config,
        records,
        data_root,
        report,
        dataset_summary,
        hashes,
        model_path,
        encoder_path,
    )

    print("Saved model + reports.")
    print(f"Run snapshot written to {run_report_path}")


if __name__ == "__main__":
    main()
