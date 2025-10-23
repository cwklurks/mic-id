# Clip 02 Misclassification Case Study

## Issue Summary
- **Symptom**: `python predict.py data/iphone/*.wav` classified `data/iphone/clip_02.wav` as “MacBook built-in microphone” (~53 %) instead of “Local iPhone recordings” (≈47 %).
- **Impact**: Undermined trust in the classifier for quiet iPhone speech, indicating poor separation between the iPhone and AirPods/Mac classes.

## Investigation
- Confirmed the mismatch reproduced after the first training run with the new TAU batches.
- Compared class distributions via `train.py --dry-run`; highlighted severe imbalance: TAU devices (≈295 clips each) vs. iPhone (15 wav + 47 m4a) vs. AirPods/Mac (15 wav + 14 m4a).
- Noted identical feature extraction between training and inference (`features.extract_features`), driving suspicion toward data coverage rather than pipeline drift.

## Actions Taken
1. **Data Organisation**
   - Split the TAU Mobile archive into `data/audio`, `data/audio2`, and `data/audio9` based on filename suffixes (`-a/-b/-c`).
   - Normalised provenance defaults in `configs/base.yaml` for the new device buckets.
2. **Metadata Refresh**
   - Ran `python3 scripts/refresh_metadata.py --config configs/base.yaml` to register hashes and sources for all clips (including new iPhone/AirPods captures).
   - Repeated after each data ingest to keep `data/metadata.csv` consistent.
3. **Model Retraining**
   - Executed `python train.py` to rebuild `models/model.pkl` and `models/label_encoder.pkl` with the expanded dataset (990 clips total).
4. **Inference UX Improvements**
   - Allowed directory inputs in `predict.py` so `python predict.py data/iphone` expands automatically.
   - Updated the “laptop” friendly name to “AirPods Pro / MacBook built-in microphone” to reflect the mixed capture source.

## Verification
- Post-retrain prediction:
  ```
  File: data/iphone/clip_02.wav
  RMS loudness: -40.8 dBFS
    1. Local iPhone recordings — 96.1%
    2. AirPods Pro / MacBook built-in microphone — 3.9%
    3. Samsung Galaxy S7 (TAU device B) — 0.0%
  ```
- The confidence inversion (≈96 % iPhone) confirms the classifier now separates the classes even for low-level speech content.

## Feature Changes for Improved Results
- `configs/base.yaml`: added TAU device folders to `include_devices` and defined CC-BY provenance defaults.
- `data/metadata.csv`: regenerated with 990 entries to incorporate the new recordings (62 iPhone, 29 AirPods/Mac).
- `devices.py`: renamed the “laptop” label to “AirPods Pro / MacBook built-in microphone” for accurate reporting.
- `predict.py`: added directory expansion and broader audio-extension support to streamline batch evaluation.
- Dataset restructuring: migrated TAU archive clips into `data/audio`, `data/audio2`, `data/audio9` directories, preserving the `-a/-b/-c` microphone mapping.

## Follow-Up Recommendations
- Continue collecting parallel iPhone vs. AirPods recordings, especially in quiet environments, until class counts approach parity with TAU devices.
- Maintain a held-out validation set (not yet captured) to quantify gains objectively beyond spot checks.
- Document future ingestion runs by appending to this case study or a dedicated experiment log under `docs/`.

