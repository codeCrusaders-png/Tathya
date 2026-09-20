"""Image-level analysis and dataset statistics aggregation."""
from __future__ import annotations

import math
import statistics

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------- #
# Per-image analysis
# --------------------------------------------------------------------------- #

def read_metadata(rec):
    """Open one image and read cheap metadata.

    Raises ``Exception`` for corrupt / unreadable files (the caller records
    those separately).
    """
    with Image.open(rec.path) as im:
        info = {
            "width": im.width,
            "height": im.height,
            "mode": im.mode,
            "format": (im.format or "UNKNOWN").upper(),
            "exif_orientation": int(im.getexif().get(274, 1)),
            "frames": int(getattr(im, "n_frames", 1)),
        }
    return info


def read_pixel_stats(rec, thumb=192):
    """Compute colour/brightness statistics on a down-scaled version.

    ``thumb`` caps the longest side used for pixel statistics so that huge
    images stay cheap to analyse.
    """
    with Image.open(rec.path) as im:
        rgb = im.convert("RGB")
        rgb.thumbnail((thumb, thumb))
        gray = im.convert("L")
        gray.thumbnail((thumb, thumb))

    rgb_arr = np.asarray(rgb, dtype=np.float32) / 255.0
    gray_arr = np.asarray(gray, dtype=np.float32) / 255.0
    if rgb_arr.size == 0 or gray_arr.size == 0:
        raise ValueError("empty image")

    channel_means = rgb_arr.mean(axis=(0, 1))
    channel_stds = rgb_arr.std(axis=(0, 1))

    rg = rgb_arr[..., 0] - rgb_arr[..., 1]
    yb = 0.5 * (rgb_arr[..., 0] + rgb_arr[..., 1]) - rgb_arr[..., 2]
    colorfulness = float(
        math.sqrt(float(rg.std()) ** 2 + float(yb.std()) ** 2)
        + 0.3 * math.sqrt(float(rg.mean()) ** 2 + float(yb.mean()) ** 2)
    )

    flat = rgb_arr.reshape(-1, 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.corrcoef(flat, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)

    return {
        "channel_means": channel_means.astype(float).tolist(),
        "channel_stds": channel_stds.astype(float).tolist(),
        "brightness": float(gray_arr.mean()),
        "brightness_std": float(gray_arr.std()),
        "colorfulness": colorfulness,
        "dark_frac": float((gray_arr < 0.12).mean()),
        "bright_frac": float((gray_arr > 0.88).mean()),
        "low_contrast": bool(float(gray_arr.std()) < 0.08),
        "channel_corr": corr.astype(float).tolist(),
    }
# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def _stat_block(values, digits=1):
    values = [float(v) for v in values]
    if not values:
        return {}
    ordered = sorted(values)
    n = len(ordered)
    p10 = ordered[max(0, int(round(0.10 * (n - 1))))]
    p90 = ordered[min(n - 1, int(round(0.90 * (n - 1))))]
    if n == 1:
        std = 0.0
    else:
        std = statistics.pstdev(ordered)
    return {
        "n": n,
        "min": round(float(ordered[0]), digits),
        "max": round(float(ordered[-1]), digits),
        "mean": round(float(statistics.mean(ordered)), digits),
        "median": round(float(statistics.median(ordered)), digits),
        "std": round(float(std), digits),
        "p10": round(float(p10), digits),
        "p90": round(float(p90), digits),
    }


def aggregate_geometry(infos):
    """Aggregate width / height / aspect-ratio statistics from metadata dicts."""
    widths, heights, aspects = [], [], []
    for info in infos:
        w, h = info["width"], info["height"]
        widths.append(w)
        heights.append(h)
        aspects.append(w / h)
    return {
        "width": _stat_block(widths, digits=0),
        "height": _stat_block(heights, digits=0),
        "aspect": _stat_block(aspects, digits=3),
    }


def recommended_resize(width_stats, height_stats, multiple=16, lo=32, hi=1024):
    """Suggest a sensible training input size (multiples of 16)."""
    median = max(width_stats.get("median", 0), height_stats.get("median", 0))
    size = int(math.ceil((median or 0) / multiple) * multiple)
    size = max(lo, min(hi, size))
    return f"{size}x{size}"


def gini_coefficient(values):
    """Gini coefficient of class-count distribution (0 = perfectly balanced)."""
    values = sorted(float(v) for v in values)
    n = len(values)
    if n == 0 or sum(values) == 0:
        return 0.0
    cumulative = 0.0
    for i, value in enumerate(values, start=1):
        cumulative += (2 * i - n - 1) * value
    return max(0.0, min(1.0, cumulative / (n * sum(values))))


def summarize_channels(pixel_results):
    """Average per-channel means / stds and the mean correlation matrix."""
    means = np.zeros(3, dtype=np.float64)
    stds = np.zeros(3, dtype=np.float64)
    corr = np.zeros((3, 3), dtype=np.float64)
    n = len(pixel_results)
    for result in pixel_results:
        means += np.asarray(result["channel_means"], dtype=np.float64)
        stds += np.asarray(result["channel_stds"], dtype=np.float64)
        corr += np.asarray(result["channel_corr"], dtype=np.float64)
    if n:
        means /= n
        stds /= n
        corr /= n
    return {
        "n": n,
        "channel_names": ["R", "G", "B"],
        "channel_means": means.round(4).tolist(),
        "channel_stds": stds.round(4).tolist(),
        "channel_corr_mean": corr.round(4).tolist(),
        "brightness_mean": round(float(np.mean([r["brightness"] for r in pixel_results])), 4),
        "brightness_std": round(float(np.mean([r["brightness_std"] for r in pixel_results])), 4),
        "colorfulness_mean": round(float(np.mean([r["colorfulness"] for r in pixel_results])), 2),
        "colorfulness_std": round(float(np.std([r["colorfulness"] for r in pixel_results])), 2),
        "dark_frac_mean": round(float(np.mean([r["dark_frac"] for r in pixel_results])), 4),
        "bright_frac_mean": round(float(np.mean([r["bright_frac"] for r in pixel_results])), 4),
        "low_contrast_frac": round(float(np.mean([1.0 if r["low_contrast"] else 0.0
                                                  for r in pixel_results])), 4),
    }
