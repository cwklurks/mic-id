# Mic-ID: Senior ML Engineer Walkthrough

> Speaker notes styled as a narrative script you can deliver in roughly 12–15 minutes. Adjust pacing depending on how deep you want to go into each subsystem.

## 1. Kick-off: Why Mic Fingerprinting?
- *“We built Mic-ID to fingerprint the capture device from short audio clips. Think of it as device-level source verification for field recordings and voice notes.”*
- Motivations: provenance tracking for crowdsourced datasets, QA on studio rigs, and a showcase that classic audio descriptors still punch above their weight for forensic-style tasks.
- Scope: inference-time classifier + Streamlit demo; optional retraining pipeline when new devices arrive.

## 2. Baseline Capabilities & Repo Overview
- Endpoints: CLI scorer (`predict.py`), Streamlit app (`app.py`), training harness (`train.py`).
- Artefacts: tabular features → `HistGradientBoostingClassifier`, persisted as `models/model.pkl` plus a `LabelEncoder`.
- Layout highlights:
  - `data/metadata.csv` acts as the single source of truth for provenance (device, licence, hashes).
  - `features.py` encapsulates the audio featurisation contract so both training and inference stay in sync.
  - `reports/` houses metrics JSON, confusion matrix PNG, and timestamped run manifests.

## 3. Data Strategy & Provenance Guarantees
- Primary corpus: TAU Urban Acoustic Scenes 2019 Mobile (three synchronous devices per scene: Zoom F8, Galaxy S7, iPhone SE).
- Augmentations: eight in-house clips (`iphone/`, `laptop/`) recorded via `utils.record_clips` for internal gear anchoring; optional “outtakes” exports for demos (`scripts/export_outtakes.py`).
- `scripts/refresh_metadata.py` walks `data/`, hashes every file, fills in source/licence defaults from `configs/base.yaml`, and warns about missing metadata. This keeps `train.py` from running if provenance is incomplete.
- Training config enforces per-device minima (`min_clips_per_device: 5`), split filters, and optional device allowlists to keep the dataset balanced.
- The training entrypoint revalidates hashes at load time (`enforce_hashes: true`), so stale or tampered clips fail fast.

## 4. Signal Conditioning & Feature Stack
- Every audio path funnels through `features.load_mono`: resample to 16 kHz mono, trim 30 dB silence, RMS-normalise to ≈-26 dBFS (0.05 amplitude target). Same transform runs during both training and inference for determinism.
- `extract_features` composes:
  - 64-bin log-mel spectrogram mean & std (captures coarse spectral envelope).
  - 20 MFCCs plus first & second-order deltas (means + stds), with delta width auto-tuned to available frames to stay robust on short clips.
  - Scalar spectral descriptors: zero-crossing rate, centroid, roll-off, and flatness.
- Output is a fixed-length vector (~288 dims) suitable for tabular models without sequence modelling overhead.

## 5. Model Rationale & Training Loop
- Classifier: `HistGradientBoostingClassifier` (sklearn) with moderately deep trees (`max_depth=10`, `max_iter=400`, `learning_rate=0.08`).
  - Chosen for interpretability, calibration-friendly probabilities, and resilience on low-hundreds training samples without GPU dependencies.
  - Random state is plumbed from config for reproducibility; we strip the internal RNG before serialisation to keep joblib dumps picklable.
- Training flow (`train.py`):
  1. Load YAML config, resolve data root/metadata.
  2. Validate every row (presence, hash check, licence/source, duplicate guard).
  3. Summarise dataset (counts, example hashes) and optionally exit in `--dry-run` mode.
  4. Build feature matrix by streaming each clip through `load_mono`/`extract_features`.
  5. Stratified train/test split with configurable `test_size`.
  6. Fit classifier, evaluate via `classification_report` + normalised confusion matrix.
  7. Persist artefacts under `models/` and metrics under `reports/`; emit a JSON snapshot in `reports/runs` with the config, dataset summary, hashes, and artefact paths.

## 6. Evaluation Mindset
- Metrics JSON mirrors sklearn’s per-class precision/recall/F1 plus macro/weighted aggregates for traceability.
- Confusion matrix is normalised row-wise to spotlight systematic device confusions (e.g., Galaxy vs iPhone mix-ups in similar urban scenes).
- Stored metrics allow regression tracking—drop them into CI to diff against previous baselines before promoting a new model.
- For qualitative checks, the outtakes export lets us inspect triplets where the model flips device predictions across near-identical content.

## 7. Inference Surfaces
- **CLI (`predict.py`)**: accepts files or directories, expands recursively, normalises RMS, extracts features, and prints top-k device names using `devices.describe_label`. Works on WAV/MP3/M4A; BytesIO fallback mirrors Streamlit behaviour.
- **Streamlit App (`app.py`)**:
  - Lazy-loads model/encoder via `st.cache_resource`.
  - Saves uploaded clips to `uploads/hooks - <name>` for later audit.
  - Displays RMS loudness, ranked predictions, and a probability bar chart with friendly device labels.
  - `estimate_scale` runs a lightweight chroma profile match to emit “C major” style hints when tonal structure is clear; otherwise the UI explicitly states that no scale was detected.
  - Visual diagnostics: waveform and log-mel spectrogram matplotlib plots so we can sanity-check the energy profile the model sees.
  - Expander summarises the current training corpus to keep demos self-contained.

## 8. Deployment & Operational Notes
- Hugging Face Space ready: copy the repo alongside curated `data/` subsets, `models/`, `requirements.txt`, and `packages.txt`. Streamlit SDK handles the web UI; Debian packages listed in `packages.txt` ensure codecs/libs (e.g., `ffmpeg`, `libsndfile`) are available.
- Local dev: standard Python 3.10 venv, `pip install -r requirements.txt`, optionally rebuild metadata (`scripts/refresh_metadata.py`) and retrain before launching the app.
- Caching: we pre-create `.cache/numba` and `.cache/matplotlib` in both training and inference entrypoints to avoid sandbox write errors and keep runtime deterministic inside constrained environments.

## 9. Reproducibility & Audit Trail
- Every training run produces:
  - `models/model.pkl`, `models/label_encoder.pkl`.
  - `reports/metrics.json`, `reports/confusion_matrix.png`.
  - `reports/runs/run-<timestamp>-<tag>.json` containing the exact config, dataset hashes, and artefact pointers.
- Metadata CSV includes SHA256 hashes so collaborators can re-hash their local corpora and ensure binary identity before retraining.
- `python train.py --dry-run` acts as the guardrail in CI—fails on missing files, wrong hashes, or low clip counts without touching the model.
- Uploaded demo clips persist under `uploads/` so false positives/negatives can be replayed or relabelled into the training set after review.

## 10. Known Limitations & Roadmap Talking Points
- Current features favour stationary background noise; distinguishing near-identical microphones in pristine speech recordings remains challenging. Plan: add targeted augmentations (EQ, impulse responses) to highlight device coloration.
- Model capacity is modest; consider juxtaposing a lightweight CNN or wav2vec-style embedding head as an alternative baseline while preserving explainability.
- Hash enforcement assumes single-writer access to `data/`; in multi-user settings we should gate commits through the metadata refresh script to avoid duelling hashes.
- Scale detection is heuristic; treat it as demo flair, not a musically rigorous estimate. Roadmap item: threshold tuning or optional disable flag.
- Larger-scale deployments would need storage sharding for `uploads/` and a background queue if we expose batch scoring APIs.

## 11. How to Extend
- Adding a device: drop clips under `data/<device>/`, run `scripts/refresh_metadata.py`, retrain. Device names propagate via `devices.MIC_FRIENDLY_NAMES`.
- Swapping features: `features.extract_features` is the single source—wrap new descriptors there and retrain; both CLI and UI pick them up automatically.
- CI hooks: wire `python train.py --dry-run` and `python predict.py <smoke files>` into your pipeline; diff `reports/metrics.json` for regression alerts.

## 12. Closing Soundbite
- *“Mic-ID shows how far disciplined metadata, interpretable features, and a tight training loop get you for hardware fingerprinting. It’s production-light, demo-ready, and leaves room to iterate into deep embeddings once we collect more devices.”*

