#!/usr/bin/env python3
"""Copy a handful of parallel TAU Mobile clips into an outtakes folder for manual testing.

Usage:
    python scripts/export_outtakes.py --count 5 --output data/outtakes

The script searches under the training `data/` tree for clip prefixes that include
all three device suffixes (-a/-b/-c) and copies them (without removal) to the
chosen output directory. Each exported clip retains its original file name and
is accompanied by a `manifest.csv` describing the microphone mapping.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

MIC_SUFFIX_MAP = {
    "a": ("Zoom F8 field recorder", "TAU device A"),
    "b": ("Samsung Galaxy S7", "TAU device B"),
    "c": ("iPhone SE", "TAU device C"),
}


def find_triplets(data_root: Path):
    groups: dict[str, dict[str, Path]] = defaultdict(dict)
    for wav_path in data_root.rglob("*.wav"):
        name = wav_path.name
        if len(name) < 6 or not name.endswith(".wav"):
            continue
        suffix = name[-5]
        if suffix not in MIC_SUFFIX_MAP:
            continue
        prefix = name[:-6]
        groups[prefix][suffix] = wav_path
    # Keep only groups with all three devices
    return {prefix: mapping for prefix, mapping in groups.items() if set(mapping) == set(MIC_SUFFIX_MAP)}


def export_triplets(groups: dict[str, dict[str, Path]], output_dir: Path, count: int, seed: int | None) -> list[tuple[str, str, Path]]:
    if not groups:
        return []
    rng = random.Random(seed)
    prefixes = sorted(groups)
    rng.shuffle(prefixes)
    selected = prefixes[: min(count, len(prefixes))]
    manifest_rows: list[tuple[str, str, Path]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for prefix in selected:
        mapping = groups[prefix]
        for suffix, path in mapping.items():
            dest = output_dir / path.name
            shutil.copy2(path, dest)
            manifest_rows.append((dest.name, suffix, path))
    return manifest_rows


def write_manifest(rows: list[tuple[str, str, Path]], manifest_path: Path) -> None:
    if not rows:
        return
    with manifest_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "mic_suffix", "friendly_name", "description", "source_path"])
        for filename, suffix, source in rows:
            friendly, description = MIC_SUFFIX_MAP[suffix]
            writer.writerow([filename, suffix, friendly, description, str(source)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a handful of TAU Mobile triplets for testing.")
    parser.add_argument("--data-root", default="data", type=Path, help="Root directory containing per-device folders (default: data)")
    parser.add_argument("--output", default="data/outtakes", type=Path, help="Where to copy the selected clips")
    parser.add_argument("--count", type=int, default=5, help="Number of triplet prefixes to copy (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    groups = find_triplets(args.data_root)
    if not groups:
        raise SystemExit("No complete triplets (-a/-b/-c) were found under the data root.")

    rows = export_triplets(groups, args.output, args.count, args.seed)
    if not rows:
        raise SystemExit("Triplet export produced no files. Try lowering --count.")

    write_manifest(rows, args.output / "manifest.csv")
    print(f"Exported {len(rows)} wav files to {args.output} (covering {len(rows) // 3} triplets).")


if __name__ == "__main__":
    main()
