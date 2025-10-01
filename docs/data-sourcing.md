# Data Sourcing Guide
Mic-ID works best when every class corresponds to a capture device that has enough, diverse, and comparable recordings. The notes below list vetted public corpora and give you recipes for turning them into balanced training data without having to record long sessions yourself.

## What to look for
- **Parallel content**. Prefer datasets where the same scenes were captured on several devices so the label truly reflects the hardware, not the sound source.
- **Consistent preprocessing**. Resample to 16 kHz mono, trim silence, and loudness-normalize (the repo utilities already normalise to ≈-26 dBFS).
- **Usage rights**. Check every licence; most corpora below are CC BY or research-only.

## Recommended open datasets
### TAU Urban Acoustic Scenes 2019 Mobile (DCASE 2019 Task 1B)
- **Why it helps**: Every 10 s clip was recorded in parallel across three devices: Zoom F8 (device A), Samsung Galaxy S7 (device B), and iPhone SE (device C). Treat the device ID as the label.
- **Download**: Register at https://dcase.community/challenge2019/task-acoustic-scene-classification and grab the "TAU Urban Acoustic Scenes 2019 Mobile" archive (`TAU-urban-acoustic-scenes-2019-mobile-development.zip`).
- **Command-line** (after approval/network access):
  ```bash
  mkdir -p data/raw && cd data/raw
  wget https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-mobile-development.zip
  unzip TAU-urban-acoustic-scenes-2019-mobile-development.zip
  ```
- **Prep**: Parallel recordings are stored under `A/`, `B/`, `C/`. Run the provided helper to convert to 16 kHz mono (create this file if it doesn’t exist yet):
  ```bash
  python scripts/prepare_taus_mobile.py --input data/raw/TAU-urban-acoustic-scenes-2019-mobile-development --output data/taus_mobile
  ```
  (See the README below for how to organise `scripts/prepare_taus_mobile.py`.)
- **Licence**: Creative Commons Attribution 4.0.

### ASVspoof 2019 – Physical Access subset
- **Why it helps**: Contains bona fide and replayed speech captured by 26 microphone / recorder pairs (high-quality mics plus several smartphones). Device identity is stored in the metadata (`rec_device`).
- **Download**: Free for research. Create an account at https://www.asvspoof.org/ and request the 2019 PA subset.
- **Command-line**:
  ```bash
  mkdir -p data/raw && cd data/raw
  wget https://datashare.ed.ac.uk/download/handle/10283/3336/ASVspoof2019_PA_dev.zip
  unzip ASVspoof2019_PA_dev.zip
  ```
  (Replace with the exact link you receive; ASVspoof frequently rotates URLs.)
- **Prep**: Use the protocol files `ASVspoof2019_PA_cm_protocols` to map each wav to its `rec_device`. You can pivot those IDs into the folder names Mic-ID expects:
  ```bash
  python scripts/split_by_device.py --metadata data/raw/asvspoof2019/ASVspoof2019_PA_dev_cm_protocols/cm_protocols/PA_dev_cm.txt --audio-root data/raw/asvspoof2019/ASVspoof2019_PA_dev/ASVspoof2019_PA_dev --output-root data/asvspoof_devices
  ```
- **Licence**: Research-only; check that your use case complies.

### Freesound + gear metadata
- **Why it helps**: Many Freesound uploads include `gear` or `recording_device` tags such as "iphone_se" or "zoom_h4n". You can scrape curated lists rather than recording yourself.
- **Download**: Requires a (free) Freesound API key.
- **Command-line**:
  ```bash
  export FREESOUND_API_KEY=...
  python scripts/freesound_pull.py --query "recording_device:iphone" --max-clips 200 --label iphone
  python scripts/freesound_pull.py --query "recording_device:samsung" --max-clips 200 --label galaxy_s7
  ```
- **Prep**: The script should normalise file names and audio format to match `data/<device>/clip_xx.wav`. Keep a CSV with original URLs for attribution.
- **Licence**: Clip-specific; many are CC BY or CC0. Honour attribution where required.

## Balancing and augmentation tips
- Aim for 60–100 clips per class before augmentation. Mix quiet/noisy scenes to avoid overfitting.
- Apply simple augmentations (noise injection, EQ, impulse responses) per device to highlight microphone artefacts rather than content.
- Track provenance in `data/metadata.csv` (`filename,device,source,licence`).
- Keep a held-out validation split per device to spot leakage from near-duplicate clips.

## Next steps
1. Implement the helper scripts mentioned above under `scripts/`. Use `librosa`/`soundfile` so prep stays in Python.
2. Store downloaded archives under `data/raw/` (ignored by git) and export processed clips to `data/<device>/`.
3. Update `metadata.csv` whenever you add or remove external clips so the experiment log in `reports/` stays reproducible.

For more ideas, browse the DCASE and ASVspoof challenge leaderboards—winning teams usually publish their data prep notes and often release additional impulse responses or parallel recordings.
