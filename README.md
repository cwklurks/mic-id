---
title: Mic-ID
emoji: "🎙️"
colorFrom: red
colorTo: purple
sdk: streamlit
sdk_version: "1.31.1"
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
---

# Mic-ID

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)

A Streamlit front-end around a microphone fingerprinting baseline: drop in a short clip, get the most likely capture device plus an optional tonal hint. 🎙️ Built for quick lab demos, perfect for showing off how far classic features still go.

## Table of Contents
- [Highlights](#highlights)
- [Live Demo Flow](#live-demo-flow)
- [Quick Start](#quick-start)
- [Controls at a Glance](#controls-at-a-glance)
- [Device Recognition](#device-recognition)
- [Scale Detection](#scale-detection)
- [Bundled Example Clips](#bundled-example-clips)
- [Download Contents](#download-contents)
- [Testing](#testing)
- [Project Layout](#project-layout)
- [Roadmap](#roadmap)
- [Contributing](#contributing)

## Highlights
- 🔎 End-to-end workflow for collecting, training, and demoing mic classification in one repo.
- 🎛️ Feature-first approach: log-mel, MFCC, and spectral stats feed a histogram gradient boosting model.
- 🧠 Friendly predictions: class IDs map to real device names so you can narrate results without decoding labels.
- 🗂️ Lightweight artefacts: plain `.wav` folders in `data/`, pickled models in `models/`, metrics and confusion heatmaps in `reports/`.
- ⚙️ Streamlit UI mirrors the CLI helpers, including loudness normalisation and experimental scale read-outs.

## Live Demo Flow
If you are running a live session, keep this script handy:

- 🎧 `streamlit run app.py` from the project root.
- 📂 Use `data/audio/airport-helsinki-204-6138-a.wav` to introduce the core upload flow and the default top-3 guess list.
- 🔄 Swap to `data/audio/airport-helsinki-204-6138-b.wav` or `data/audio/airport-helsinki-204-6138-c.wav` to highlight how the twin scene shifts the predicted device while the environment stays constant.
- 📱 Jump to `data/iphone/clip_05.wav` to show the locally recorded class and talk about adding in-house gear with `utils.py`.
- 📊 Mention the probability bar chart and the saved copy under `uploads/hooks - <filename>` for later analysis.

## Quick Start
⚡ Four commands set everything up:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/refresh_metadata.py  # rebuild hashes + provenance records
python3 train.py --config configs/base.yaml  # optional if you want to refresh the model
```

Then launch the app with `streamlit run app.py` (defaults to http://localhost:8501).

## Hugging Face Space Setup
Want a hosted demo? This repo is ready to drop into a Hugging Face Space using the Streamlit SDK. The short version:

1. `pip install -U "huggingface_hub[cli]"` and run `huggingface-cli login` with a write-scoped access token.
2. `git clone` your Space (for example `https://huggingface.co/spaces/connaaa/mic-id`) into an empty folder.
3. Copy the contents of this repository into that clone, keeping `README.md`, `app.py`, `requirements.txt`, `packages.txt`, `models/`, and the curated `data/` subsets you want online.
4. Commit and `git push`. The Space will build the dependencies listed in `requirements.txt` plus Debian packages from `packages.txt`.

Large training corpora can be trimmed before pushing if you only need the pretrained model for inference.

## Controls at a Glance
| Control | Default | What it does |
| --- | --- | --- |
| File uploader | – | Accepts WAV/MP3/M4A, converts to 16 kHz mono, and normalises loudness before scoring. |
| `How many guesses should we list?` slider | 3 | Sets the length of the ranked prediction list and bar chart. |
| Training data expander | Collapsed | Recaps which datasets went into the current checkpoint, handy during demos. |
| Prediction pane | Auto | Shows the tonal estimate (if any), RMS loudness, ranked devices, and probability chart. |

Each control includes inline help text so presenters can improvise without notes.

## Device Recognition
- 🧱 Audio flows through `features.extract_features`, stitching log-mel and MFCC statistics with zero-crossing, centroid, roll-off, and flatness cues.
- 🌲 `python3 train.py --config configs/base.yaml` reads the provenance metadata, enforces per-device clip minimums, and fits a `HistGradientBoostingClassifier` before saving artefacts to `models/model.pkl` plus the label encoder.
- 📈 Every training run exports `reports/metrics.json`, `reports/confusion_matrix.png`, and a timestamped `reports/runs/run-*.json` snapshot so you can cite precision/recall live.
- 🏷️ The app and CLI surface friendly names (e.g. “Zoom F8 field recorder”) pulled from `devices.describe_label()` to keep the story human-readable.

## Scale Detection
- 🎼 Uses a simple `librosa` chroma profile match across all major/minor keys.
- ✅ High confidence (≥ 0.6) renders a green highlight, 0.4–0.6 shows an amber “low confidence” tag, and anything lower hides the scale suggestion entirely.
- 🥁 Purely percussive or noisy clips skip the tonal hint, which is exactly what you want for location recordings.

## Bundled Example Clips
All sample audio lives under `data/` and mirrors the device IDs referenced in the demo.

| Folder | What it represents | Count* |
| --- | --- | --- |
| `audio/` | TAU Urban Acoustic Scenes clips (device A) – Zoom F8 field recorder | 295 |
| `audio2/` | TAU Urban Acoustic Scenes clips (device B) – Samsung Galaxy S7 | 295 |
| `audio9/` | TAU Urban Acoustic Scenes clips (device C) – iPhone SE | 295 |
| `iphone/` | Locally recorded iPhone speech snippets captured with `utils.py` | 4 |
| `laptop/` | MacBook built-in mic samples recorded in a treated room | 4 |
| `outtakes/` | Extra captures you can promote into training data after curation | varies |

⋆Counts based on the current repo snapshot; refresh `data/` to rebalance as needed.

## Download Contents
Every run generates artefacts you can drop into a slide deck or share with collaborators:

- 🎯 `models/model.pkl` and `models/label_encoder.pkl` store the trained classifier and label map.
- 📊 `reports/metrics.json` plus `reports/confusion_matrix.png` capture evaluation snapshots for the latest training session.
- 🧾 `data/metadata.csv` tracks every clip’s provenance, licence, and hash for reproducible retrains.
- 🗂️ `reports/runs/run-*.json` snapshots record the exact config, dataset summary, and hashes used for each training run.
- 📁 Uploaded clips are preserved under `uploads/hooks - <original-name>` so you can replay or re-label them later.

## Testing
Quick smoke checks live in the scripts themselves:

```bash
# Validate provenance without training
python3 train.py --dry-run

# Rebuild the model, metrics, and run snapshot
python3 train.py --config configs/base.yaml

# Score a few clips and verify probabilities look sane
python3 predict.py data/laptop/clip_01.wav data/iphone/clip_05.wav --topk 5
```

For deeper regression coverage, wire these commands into your CI and compare the resulting metrics JSON against previous baselines.

## Project Layout
```
mic-id/
 ├─ app.py              # Streamlit UI for uploading and scoring clips
 ├─ predict.py          # CLI scorer with friendly device names
 ├─ train.py            # Dataset loader, model trainer, metric exporter
 ├─ configs/            # YAML training configs + device provenance defaults
 ├─ features.py         # Audio feature extraction helpers
 ├─ utils.py            # Command-line recorder for new device samples
 ├─ data/               # Per-device waveforms and provenance metadata
 │   └─ metadata.csv    # Clip-level provenance (source/licence/hash)
 ├─ models/             # Saved classifier + label encoder
 ├─ reports/            # Metrics JSON and confusion matrix plots
 ├─ docs/               # Data sourcing guide and prep notes
 ├─ scripts/            # Dataset preparation helpers (TAU, Freesound, etc.)
 └─ uploads/            # Cached demo uploads saved by the Streamlit app
```

## Roadmap
- 🛰️ Add a lightweight CNN baseline alongside the gradient boosting model for comparison.
- 🧪 Ship augmentation scripts (noise, EQ, impulse responses) to spotlight microphone colouration differences.
- 🔐 Wire metadata/hash validation into CI so new clips are rejected unless provenance is complete.
- 📦 Polish export helpers so the app can bundle probabilities + features in one download.

## Contributing
Issues and pull requests are welcome. 🤝 If you contribute new devices, include a short note (or a `metadata.csv` entry) describing the capture setup so others can reproduce your results and audit licensing.
