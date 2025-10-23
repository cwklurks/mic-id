#!/usr/bin/env python3
"""Generate or refresh data/metadata.csv entries with provenance details."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import yaml


DEFAULT_EXTENSIONS = {".wav", ".mp3", ".m4a"}


@dataclass
class MetadataRow:
    path: Path
    device: str
    source: str
    license: str
    split: str
    sha256: str

    def as_dict(self, root: Path) -> Dict[str, str]:
        rel_path = self.path.relative_to(root).as_posix()
        return {
            "path": rel_path,
            "device": self.device,
            "source": self.source,
            "license": self.license,
            "split": self.split,
            "sha256": self.sha256,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml", help="YAML config that defines data root and defaults.")
    parser.add_argument("--output", help="Override output metadata CSV path. Defaults to the config value.")
    parser.add_argument("--extensions", nargs="*", help="File extensions to include (e.g., .wav .mp3 .m4a). Defaults to built-ins.")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if "data" not in cfg:
        raise SystemExit("Config is missing a `data` section.")
    return cfg


def read_existing_metadata(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return {row["path"]: row for row in reader if "path" in row}


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def gather_files(root: Path, extensions: Iterable[str]) -> Iterable[Path]:
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() in extensions:
            yield file_path


def build_rows(
    files: Iterable[Path],
    existing_rows: Dict[str, dict],
    root: Path,
    device_defaults: Optional[dict],
    include_devices: Optional[set[str]],
) -> Iterable[MetadataRow]:
    for path in files:
        rel_key = path.relative_to(root).as_posix()
        parts = path.relative_to(root).parts
        if not parts:
            continue
        device = parts[0]
        if include_devices and device not in include_devices:
            continue

        defaults = (device_defaults or {}).get(device, {})
        existing = existing_rows.get(rel_key, {})

        source = existing.get("source") or defaults.get("source")
        license_ = existing.get("license") or defaults.get("license")
        split = existing.get("split") or "train"

        if not source or not license_:
            sys.stderr.write(f"[warn] Missing source/license for {rel_key}; fill these in manually.\n")

        sha256 = compute_sha256(path)

        yield MetadataRow(
            path=path,
            device=device,
            source=source or "",
            license=license_ or "",
            split=split,
            sha256=sha256,
        )


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_config(config_path)

    data_cfg = cfg["data"]
    root = Path(data_cfg.get("root", "data")).resolve()
    metadata_path = Path(args.output or data_cfg.get("metadata", root / "metadata.csv")).resolve()
    extensions = {ext.lower() for ext in (args.extensions or data_cfg.get("extensions", DEFAULT_EXTENSIONS))}

    if not root.exists():
        raise SystemExit(f"Data root does not exist: {root}")

    existing_rows = read_existing_metadata(metadata_path)
    device_defaults = data_cfg.get("device_defaults", {})
    include_devices = set(data_cfg.get("include_devices", []) or [])

    files = sorted(gather_files(root, extensions))
    rows = sorted(
        build_rows(files, existing_rows, root, device_defaults, include_devices if include_devices else None),
        key=lambda row: row.path.relative_to(root).as_posix(),
    )

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "device", "source", "license", "split", "sha256"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict(root))

    orphaned = sorted(set(existing_rows) - {row.path.relative_to(root).as_posix() for row in rows})
    if orphaned:
        sys.stderr.write(f"[warn] Orphaned metadata entries (files missing): {len(orphaned)}\n")
        for item in orphaned:
            sys.stderr.write(f"  - {item}\n")

    print(f"Wrote {len(rows)} rows to {metadata_path}")


if __name__ == "__main__":
    main()
