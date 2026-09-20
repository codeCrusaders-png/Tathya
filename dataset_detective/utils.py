"""Shared low-level helpers for dataset_detective."""
from __future__ import annotations

import hashlib

__all__ = [
    "IMAGE_EXTENSIONS",
    "NOISE_DIRNAMES",
    "SPLIT_DIRNAMES",
    "GENERIC_DIRNAMES",
    "human_size",
    "human_time",
    "sha256_file",
    "safe_name",
]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".jfif", ".jpe",
    ".png", ".bmp", ".gif", ".webp",
    ".tif", ".tiff",
    ".ppm", ".pgm", ".pbm", ".pnm",
    ".heic", ".heif", ".avif",
}

NOISE_DIRNAMES = {
    ".git", "__pycache__", ".venv", "venv", "node_modules",
    ".idea", ".vscode", ".ipynb_checkpoints",
}

SPLIT_DIRNAMES = {
    "train", "training", "test", "testing",
    "val", "valid", "validation", "eval", "evaluate", "dev", "split",
}

GENERIC_DIRNAMES = {
    "images", "imgs", "img", "image", "pics", "pictures", "photos", "photo",
    "pix", "jpeg", "jpg", "png", "raw", "originals", "source", "data",
    "dataset", "files", "media", "assets",
}


def human_size(num_bytes):
    """Format a byte count in a human friendly way."""
    try:
        value = float(num_bytes)
    except (TypeError, ValueError):
        return "n/a"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def human_time(seconds):
    """Format a duration in seconds."""
    seconds = max(0.0, float(seconds))
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    return f"{int(seconds // 60)}m {seconds % 60:.0f}s"


def sha256_file(path, blocksize=1 << 20):
    """Return the lower-case hex sha256 digest of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(blocksize)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(name):
    """Slugify a short name for use in filenames and anchors."""
    out = []
    for ch in str(name):
        if ch.isalnum():
            out.append(ch.lower())
        elif ch in " _-.":
            out.append("-")
    return "".join(out).strip("-") or "item"
