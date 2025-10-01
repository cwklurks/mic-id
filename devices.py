MIC_FRIENDLY_NAMES = {
    "audio": "Zoom F8 field recorder (TAU device A)",
    "audio2": "Samsung Galaxy S7 (TAU device B)",
    "audio9": "iPhone SE (TAU device C)",
    "iphone": "Local iPhone recordings",
    "laptop": "MacBook built-in microphone",
}


def describe_label(label: str) -> str:
    """Return a human-readable microphone description for a raw label."""
    return MIC_FRIENDLY_NAMES.get(label, label)
