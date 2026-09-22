"""Command-line interface and orchestration for tathya."""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from . import __version__
from .analysis import (
    aggregate_geometry,
    gini_coefficient,
    read_metadata,
    read_pixel_stats,
    recommended_resize,
    summarize_channels,
)
from .hashing import find_exact_duplicates, find_near_duplicates, find_split_leakage
from .model import run_baseline
from .plots import build_plots
from .report import render_html, render_markdown
from .sar import compose_sar_alerts_and_recs, is_sar_or_geotiff_dataset, run_sar_audit
from .scanning import detect_layout, discover_images
from .utils import human_size, human_time

# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _sanitize_floats(obj):
    """Recursively replace NaN and Inf with None for strict JSON compliance."""
    import math
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_floats(v) for v in obj]
    return obj


class _Progress:
    def __init__(self, quiet, label, total):
        self.quiet = quiet
        self.label = label
        self.total = max(1, total)
        self.done = 0

    def tick(self):
        self.done += 1
        if not self.quiet:
            step = max(1, self.total // 40)
            if self.done == self.total or self.done % step == 0:
                filled = int(20 * self.done / self.total)
                bar = "#" * filled + "-" * (20 - filled)
                sys.stdout.write(f"\r  {self.label:<12} [{bar}] {self.done}/{self.total}")
                sys.stdout.flush()
                if self.done == self.total:
                    sys.stdout.write("\n")


def _run_parallel(records, fn, workers, label, quiet):
    """Run ``fn(record)`` for every record using a thread pool.

    Returns ``{rel_path: result_or_exception}``.  Exceptions are returned, not
    raised, so unreadable files never abort the analysis.
    """
    progress = _Progress(quiet, label, len(records))
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, rec): rec for rec in records}
        for future in as_completed(futures):
            rec = futures[future]
            try:
                results[rec.rel_path] = future.result()
            except Exception as exc:  # noqa: BLE001
                results[rec.rel_path] = exc
            progress.tick()
    return results


def _metadata_pass(records, workers, quiet):
    """Read metadata for every image; return ``(infos, corrupt)``."""
    raw = _run_parallel(records, read_metadata, workers, "metadata", quiet)
    infos, corrupt = {}, []
    for rec in records:
        value = raw[rec.rel_path]
        if isinstance(value, Exception):
            corrupt.append({
                "rel_path": rec.rel_path,
                "path": rec.path,
                "error": str(value)[:300],
            })
        else:
            infos[rec.rel_path] = value
    return infos, corrupt


def _select_pixel_sample(records, infos, layout, cap, seed=0):
    """Deterministic, class-aware sample for the pixel-statistics pass."""
    usable = [rec for rec in records if rec.rel_path in infos]
    if not usable:
        return []
    rng = np.random.RandomState(seed)
    if not layout.class_of:
        rng.shuffle(usable)
        return usable[:cap]

    from collections import defaultdict

    buckets = defaultdict(list)
    for rec in usable:
        label = layout.class_of.get(rec.rel_path, "unlabelled")
        buckets[label].append(rec)
    for recs in buckets.values():
        recs.sort(key=lambda rec: rec.rel_path)

    per_class = max(1, cap // max(1, len(buckets)))
    selected, seen = [], set()
    for recs in buckets.values():
        if len(recs) <= per_class:
            pick = recs
        else:
            indexes = np.linspace(0, len(recs) - 1, per_class, dtype=int)
            pick = [recs[i] for i in indexes]
        for rec in pick:
            if rec.rel_path not in seen:
                seen.add(rec.rel_path)
                selected.append(rec)
    if len(selected) < cap:
        rng.shuffle(usable)
        for rec in usable:
            if len(selected) >= cap:
                break
            if rec.rel_path not in seen:
                seen.add(rec.rel_path)
                selected.append(rec)
    return selected[:cap]


def _group_repr(groups):
    """Return ``(rel_path_lists, image_count, wasted_bytes)`` for duplicate groups."""
    names, images, wasted = [], 0, 0
    for group in groups:
        sizes = [rec.size_bytes for rec in group]
        names.append([rec.rel_path for rec in group])
        images += len(group)
        wasted += max(0, sum(sizes) - max(sizes))
    return names, images, int(wasted)
# --------------------------------------------------------------------------- #
# Alerts & recommendations
# --------------------------------------------------------------------------- #

UNSUPPORTED_FORMATS = {"HEIC", "HEIF", "AVIF"}


def _compose_alerts(ctx):
    alerts = []
    layout = ctx["layout"]
    classes = ctx["classes"]
    corrupt = ctx["corrupt"]
    duplicates = ctx["duplicates"]
    pixels = ctx["pixels"]
    geometry = ctx["geometry"]
    formats = ctx["formats"]
    splits = layout.get("splits") or []

    if corrupt:
        sample = ", ".join(item["rel_path"] for item in corrupt[:3])
        more = f" (+{len(corrupt) - 3} more)" if len(corrupt) > 3 else ""
        alerts.append({"level": "error",
                       "message": f"{len(corrupt)} corrupt or unreadable image file(s): "
                                  f"{sample}{more}"})

    if duplicates.get("exact_groups"):
        alerts.append({"level": "warn",
                       "message": f"{duplicates['exact_groups']} exact-duplicate group(s) - "
                                  f"{duplicates['exact_images']} images share identical bytes "
                                  f"(~{human_size(duplicates['exact_bytes'])} wasted)."})
    if duplicates.get("near_groups"):
        note = " (sample)" if duplicates.get("near_sampled") else ""
        alerts.append({"level": "warn",
                       "message": f"{duplicates['near_groups']} near-duplicate group(s){note} - "
                                  f"{duplicates['near_images']} perceptually similar images "
                                  f"(~{human_size(duplicates['near_bytes'])} potential duplicate storage)."})
    for leak in duplicates.get("leakage", []):
        alerts.append({"level": "error",
                       "message": f"Potential data leakage risk: {leak['groups']} duplicate group(s) "
                                  f"span {leak['between']} ({leak['images']} images) — manual verification advised."})

    for gleak in layout.get("group_leakage", []):
        alerts.append({
            "level": "error",
            "message": f"Subject/Group leakage: group '{gleak['group']}' spans splits {', '.join(gleak['splits'])} "
                       f"({gleak['count']} images) — images from the same subject must not cross splits."
        })

    sidecar_audit = layout.get("sidecar_audit") or {}
    if sidecar_audit.get("missing_on_disk"):
        n_miss = len(sidecar_audit["missing_on_disk"])
        alerts.append({
            "level": "error",
            "message": f"Sidecar metadata mismatch: {n_miss} file(s) listed in sidecar do not exist on disk."
        })
    if sidecar_audit.get("null_or_empty_labels"):
        n_null = len(sidecar_audit["null_or_empty_labels"])
        alerts.append({
            "level": "warn",
            "message": f"Sidecar metadata: {n_null} entry/entries have empty or missing labels."
        })
    if sidecar_audit.get("duplicate_entries"):
        n_dup = len(sidecar_audit["duplicate_entries"])
        alerts.append({
            "level": "warn",
            "message": f"Sidecar metadata: {n_dup} duplicate image filename(s) listed in sidecar."
        })

    counts = classes.get("counts", {})
    if counts:
        balance = classes.get("balance", {})
        ratio = balance.get("imbalance_ratio", 0) or 0
        if ratio > 100:
            alerts.append({"level": "error",
                           "message": f"Severe class imbalance (max/min = {ratio:.1f}x)."})
        elif ratio > 10:
            alerts.append({"level": "warn",
                           "message": f"Class imbalance is high (max/min = {ratio:.1f}x)."})
        for name in balance.get("small_classes", [])[:6]:
            alerts.append({"level": "warn",
                           "message": f"Class '{name}' has only {counts[name]} image(s)."})
        if len(counts) == 1:
            alerts.append({"level": "info",
                           "message": "Single-class dataset - the class distribution is not "
                                      "meaningful."})

    if geometry.get("width"):
        w, h = geometry["width"], geometry["height"]
        if w.get("min", 9999) < 32 or h.get("min", 9999) < 32:
            alerts.append({"level": "warn",
                           "message": f"Some images are smaller than 32px on a side "
                                      f"(min {w.get('min')}x{h.get('min')})."})
        aspect = geometry.get("aspect", {})
        if aspect and aspect.get("n", 0) > 0:
            extreme = [v for v in ctx["aspects"] if v < 0.5 or v > 2.0]
            if extreme and len(extreme) / len(ctx["aspects"]) > 0.05:
                alerts.append({"level": "warn",
                               "message": f"{len(extreme)}/{len(ctx['aspects'])} images have extreme "
                                          "aspect ratios (<0.5 or >2.0)."})

    unsupported = [fmt for fmt in formats if fmt in UNSUPPORTED_FORMATS]
    if unsupported:
        alerts.append({"level": "warn",
                       "message": f"Cannot decode {sum(formats[f] for f in unsupported)} "
                                  f"{'/'.join(unsupported)} file(s) with the current Pillow build "
                                  "- convert them to JPEG/PNG first."})

    if pixels.get("n"):
        if pixels["dark_frac_mean"] > 0.25 or pixels["bright_frac_mean"] > 0.15:
            alerts.append({"level": "warn",
                           "message": f"Exposure risk: dark {pixels['dark_frac_mean']*100:.0f}% / "
                                      f"bright {pixels['bright_frac_mean']*100:.0f}% of sampled pixels."})
        if pixels["low_contrast_frac"] > 0.20:
            alerts.append({"level": "warn",
                           "message": f"{pixels['low_contrast_frac']*100:.0f}% of sampled images have "
                                      "low contrast."})

    if splits:
        class_names = layout.get("classes", [])
        for row in splits:
            missing = [c for c in class_names if row["class_counts"].get(c, 0) == 0]
            if missing:
                alerts.append({"level": "warn",
                               "message": f"Split '{row['name']}' is missing {len(missing)} "
                                          f"class(es): {', '.join(missing[:5])}"})
        if not any(row["name"].lower() in {"test", "testing"} for row in splits):
            alerts.append({"level": "warn",
                           "message": "No explicit test split was found - plan a held-out "
                                      "evaluation set before training."})
    elif layout.get("kind") == "flat":
        alerts.append({"level": "info",
                       "message": "Dataset is unlabelled (flat folder, no label file). "
                                  "Class-level statistics are skipped."})

    return alerts
def _compose_recommendations(ctx):
    recs = []
    layout = ctx["layout"]
    geometry = ctx["geometry"]
    duplicates = ctx["duplicates"]
    corrupt = ctx["corrupt"]
    classes = ctx["classes"]
    pixels = ctx["pixels"]
    formats = ctx["formats"]
    baseline = ctx["baseline"]

    if corrupt:
        recs.append(f"Remove or re-download the {len(corrupt)} corrupt/unreadable file(s) "
                    "before training; they will crash a DataLoader.")
    if geometry.get("recommended"):
        if geometry.get("resize_strategy") == "letterbox":
            aspect_val = geometry.get("aspect", {}).get("median", 1.0)
            recs.append(f"Pad or letterbox images to {geometry['recommended']} (multiple of 16) rather than "
                        f"stretching; dominant aspect ratio is {aspect_val:.2f}:1.")
        else:
            recs.append(f"Resize or letterbox to {geometry['recommended']} (multiple of 16) and "
                        "adjust for the dominant aspect ratio before training.")
    if duplicates.get("exact_groups"):
        recs.append(f"Delete {duplicates['exact_images']} exact duplicate files "
                    f"(saves ~{human_size(duplicates['exact_bytes'])}).")
    if duplicates.get("near_groups"):
        recs.append(f"Review the {duplicates['near_groups']} near-duplicate groups; keep one "
                    "representative per group and re-split afterwards.")
    for leak in duplicates.get("leakage", []):
        recs.append(f"Re-split the dataset so no near-duplicate group spans {leak['between']}.")
    if layout.get("group_leakage"):
        recs.append("Re-split dataset at subject/group level (e.g. GroupKFold / GroupShuffleSplit) "
                    "so no patient/subject appears across both train and validation/test splits.")
    if (layout.get("sidecar_audit") or {}).get("missing_on_disk"):
        recs.append("Re-generate or edit sidecar metadata to remove references to files missing from disk.")
    ratio = classes.get("balance", {}).get("imbalance_ratio", 1) or 1
    if ratio > 10:
        recs.append("Balance classes via stratified sampling, class weights, or over-sampling "
                    "small classes (e.g. WeightedRandomSampler).")
        recs.append("Use macro-F1 (not accuracy) as the primary metric on imbalanced data.")
    if any(fmt in UNSUPPORTED_FORMATS for fmt in formats):
        recs.append("Convert unsupported formats (HEIC/AVIF/...) to JPEG or PNG with an "
                    "external tool.")
    if pixels.get("n") and pixels["low_contrast_frac"] > 0.20:
        recs.append("Consider contrast normalisation (CLAHE) as a preprocessing step.")
    if baseline:
        acc = baseline["accuracy_mean"]
        if acc < 0.5:
            recs.append("The baseline accuracy is low - check for label noise, duplicate "
                        "classes, or near-identical classes and gather more diverse data.")
        else:
            recs.append(f"A simple linear baseline already reaches {acc*100:.1f}% CV accuracy, "
                        "so the signal is learnable - a CNN should do considerably better.")
    if not recs:
        recs.append("No blocking issues found - the dataset looks ready for a first training run.")
    return recs
# --------------------------------------------------------------------------- #
# Console summary
# --------------------------------------------------------------------------- #

def _print_summary(ctx, findings, elapsed, output_dir):
    layout = ctx["layout"]
    classes = ctx["classes"]
    images = ctx["images"]
    geometry = ctx["geometry"]
    pixels = ctx["pixels"]
    duplicates = ctx["duplicates"]
    baseline = ctx["baseline"]
    corrupt = ctx["corrupt"]
    formats = ctx["formats"]
    meta = findings["meta"]

    print()
    print("=" * 78)
    print(f" TATHYA  v{meta['tool_version']}   analysis summary")
    print("=" * 78)
    rows = [
        ("Root", str(meta["root"])),
        ("Layout", layout["description"]),
        ("Label source", layout["label_source"] or "none"),
        ("Total images", f"{images['total']:,}"),
        ("Disk size", human_size(images["disk_bytes"])),
        ("Classes", f"{classes['total']:,}" if classes["total"] else "n/a (unlabelled)"),
    ]
    for key, value in rows:
        print(f"  {key:<16} {value}")
    if classes.get("counts"):
        balance = classes["balance"]
        print(f"  {'Class balance':<16} max/min {balance['imbalance_ratio']:.1f}x  "
              f"(Gini {balance['gini']:.2f})")
        print(f"  {'Largest/smallest':<16} {balance['largest'][0]} ({balance['largest'][1]})  /  "
              f"{balance['smallest'][0]} ({balance['smallest'][1]})")
    if formats:
        parts = ", ".join(f"{name} {count}" for name, count in sorted(formats.items(),
                                                                      key=lambda kv: -kv[1])[:4])
        print(f"  {'Formats':<16} {parts}")
    print()
    print("  Image geometry")
    if geometry.get("width"):
        print(f"    sizes       {geometry['width']['min']}x{geometry['height']['min']}"
              f" .. {geometry['width']['max']}x{geometry['height']['max']}  "
              f"(median {geometry['width']['median']}x{geometry['height']['median']})")
        print(f"    aspect      min {geometry['aspect']['min']}  median {geometry['aspect']['median']}  "
              f"max {geometry['aspect']['max']}")
        print(f"    recommended resize -> {geometry['recommended']}")
    if corrupt:
        print(f"    CORRUPT     {len(corrupt)} file(s) could not be opened")
    if pixels.get("n"):
        print(f"    pixels      sampled {pixels['n']}; brightness {pixels['brightness_mean']:.2f} "
              f"+- {pixels['brightness_std']:.2f}; colorfulness {pixels['colorfulness_mean']:.1f}; "
              f"low-contrast {pixels['low_contrast_frac']*100:.0f}%")
    if duplicates.get("exact_groups") or duplicates.get("near_groups"):
        print()
        print("  Duplicates")
        if duplicates.get("exact_groups"):
            print(f"    exact       {duplicates['exact_groups']} groups / {duplicates['exact_images']} "
                  f"images  (~{human_size(duplicates['exact_bytes'])} wasted)")
        if duplicates.get("near_groups"):
            note = " on sample" if duplicates.get("near_sampled") else ""
            print(f"    near (dHash){duplicates['near_groups']} groups / {duplicates['near_images']} "
                  f"images  (~{human_size(duplicates['near_bytes'])} wasted){note}")
    if baseline:
        print(f"\n  Trainability baseline: {baseline['accuracy_mean']*100:.1f}% +- "
              f"{baseline['accuracy_std']*100:.1f}% CV "
              f"({baseline['images_used']} images / {baseline['classes_used']} classes, 3 folds)")

    sar = findings.get("sar")
    if sar:
        print()
        print("  SAR & Remote Sensing Integrity")
        pols = sar.get("polarizations", {})
        if pols.get("detected_polarizations"):
            print(f"    polarizations  {', '.join(pols['detected_polarizations'])} "
                  f"({pols.get('dual_pol_pairs', 0)} dual-pol pairs, {len(pols.get('orphan_scenes', []))} orphan scenes)")
        cal = sar.get("radiometry", {})
        if cal.get("scales_detected"):
            nodata_str = ", ".join(cal.get("nodata_types", [])) or "none"
            print(f"    radiometry     scales: {', '.join(cal['scales_detected'])} "
                  f"(mixed: {cal.get('mixed_calibration', False)}, NoData: {nodata_str})")
        geo = sar.get("geospatial", {})
        if geo.get("georeferenced_count", 0) > 0:
            crs_str = ", ".join(geo.get("epsg_counts", {}).keys())
            print(f"    geospatial     {geo['georeferenced_count']} georeferenced tiles ({crs_str})")
        spat = sar.get("spatial_leakage", {})
        if spat.get("leakage_risk", "none") != "none":
            print(f"    spatial risk   {spat['leakage_risk'].upper()} — {len(spat.get('overlapping_pairs', []))} overlapping, "
                  f"{len(spat.get('adjacent_pairs', []))} adjacent cross-split tiles")

    alerts = findings["alerts"]
    warn = sum(1 for a in alerts if a["level"] == "warn")
    err = sum(1 for a in alerts if a["level"] == "error")
    info = sum(1 for a in alerts if a["level"] == "info")
    print()
    print(f"  Alerts: {len(alerts)} ({err} error, {warn} warning, {info} info)")
    print()
    print("  Reports written:")
    for item in sorted(output_dir.rglob("*")):
        if item.is_file():
            print(f"    - {item}")
    print()
    print(f"  Done in {human_time(elapsed)}")
    print("=" * 78)
# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #

def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="tathya",
        description="Tathya — uncover the truth of your dataset before you train.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("root", help="path to the dataset folder")
    parser.add_argument("--output", "-o", default=None,
                        help="report output folder (default: <root>/tathya_report)")
    parser.add_argument("--formats", nargs="+", default=["html", "md", "json"],
                        choices=["html", "md", "json"],
                        help="report formats to write")
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel worker threads (default: half the CPUs)")
    parser.add_argument("--pixel-sample", type=int, default=4000,
                        help="max images used for pixel statistics")
    parser.add_argument("--no-pixel-stats", action="store_true",
                        help="skip the pixel-statistics pass")
    parser.add_argument("--no-dedup", action="store_true",
                        help="skip exact + near-duplicate detection")
    parser.add_argument("--dup-threshold", type=int, default=6,
                        help="dHash Hamming distance for near-duplicate detection")
    parser.add_argument("--near-dup-cap", type=int, default=20000,
                        help="max images considered for near-duplicate detection")
    parser.add_argument("--rotation-invariant", action="store_true",
                        help="detect near-duplicates across 90/180/270 degree rotations and flips")
    parser.add_argument("--thumb-size", type=int, default=192,
                        help="thumbnail size for pixel statistics pass (default: 192)")
    parser.add_argument("--full-res", action="store_true",
                        help="compute pixel statistics on full-resolution images instead of thumbnails")
    parser.add_argument("--sar", action="store_true",
                        help="run SAR (Synthetic Aperture Radar) and remote sensing integrity checks")
    parser.add_argument("--no-baseline", action="store_true",
                        help="skip the trainability baseline")
    parser.add_argument("--no-plots", action="store_true",
                        help="skip chart rendering")
    parser.add_argument("--silent", action="store_true",
                        help="reduce console output")
    parser.add_argument("--open", action="store_true",
                        help="open the HTML report after finishing")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Report writing
# --------------------------------------------------------------------------- #

def _write_reports(findings, output_dir, formats):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if "html" in formats:
        path = output_dir / "report.html"
        path.write_text(render_html(findings), encoding="utf-8")
        written.append(path)
    if "md" in formats:
        assets = output_dir / "assets"
        assets.mkdir(exist_ok=True)
        for index, chart in enumerate(findings.get("plots", []), start=1):
            payload = chart["data_url"].split(",", 1)[1]
            (assets / f"chart_{index:02d}.png").write_bytes(base64.b64decode(payload))
        path = output_dir / "report.md"
        path.write_text(render_markdown(findings, asset_dir="assets"), encoding="utf-8")
        written.append(path)
    if "json" in formats:
        path = output_dir / "report.json"
        clean_findings = _sanitize_floats(findings)
        path.write_text(json.dumps(clean_findings, indent=2, ensure_ascii=False,
                                   default=_json_default), encoding="utf-8")
        written.append(path)
    return written


def _write_empty_report(root, output_dir, formats, elapsed, quiet=False):
    """Handle the 'no images found' edge case with a minimal report."""
    from datetime import datetime, timezone

    findings = {
        "meta": {
            "tool": "tathya",
            "tool_version": __version__,
            "root": str(root),
            "dataset_name": root.name or str(root),
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "elapsed": human_time(elapsed),
            "image_files": 0,
        },
        "layout": {"kind": "empty",
                   "description": "No image files were found under the given folder.",
                   "splits": [], "label_source": "none", "notes": []},
        "images": {"total": 0, "corrupt": [], "disk_bytes": 0},
        "alerts": [{"level": "error",
                    "message": "No image files found - check the path and the file extensions."}],
        "recommendations": ["Double-check the dataset path, or use supported extensions "
                            "(jpg/png/webp/bmp/gif/tiff/...)."],
        "plots": [],
    }
    if not quiet:
        print("[tathya] No image files found under the given folder.")
    written = _write_reports(findings, output_dir, formats) if formats else []
    if not quiet:
        for path in written:
            print(f"  wrote {path}")
    return 0
# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv=None):
    args = _parse_args(argv)
    start = time.perf_counter()

    # Validate numeric options
    if args.workers is not None and args.workers < 1:
        print("[tathya] error: --workers must be >= 1", file=sys.stderr)
        return 2
    if args.pixel_sample < 0:
        print("[tathya] error: --pixel-sample must be >= 0", file=sys.stderr)
        return 2
    if args.dup_threshold < 0:
        print("[tathya] error: --dup-threshold must be >= 0", file=sys.stderr)
        return 2
    if args.near_dup_cap < 1:
        print("[tathya] error: --near-dup-cap must be >= 1", file=sys.stderr)
        return 2
    if args.thumb_size < 1:
        print("[tathya] error: --thumb-size must be >= 1", file=sys.stderr)
        return 2

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"[tathya] error: not a directory: {root}", file=sys.stderr)
        return 2

    output_dir = Path(args.output).expanduser().resolve() if args.output else root / "tathya_report"
    if not args.silent:
        print(f"[tathya] Investigating {root}")

    records, missed_files = discover_images(root, exclude_dirs=[output_dir])
    if not args.silent:
        print(f"  found {len(records):,} image file(s) "
              f"(skipped {missed_files:,} non-image file(s))")
    if not records:
        return _write_empty_report(root, output_dir, args.formats,
                                   time.perf_counter() - start, args.silent)

    layout = detect_layout(records, root)
    workers = args.workers or max(1, (os.cpu_count() or 4) // 2)

    # --- pass 1: metadata for every image ------------------------------------
    infos, corrupt = _metadata_pass(records, workers, args.silent)
    valid_infos = [infos[rec.rel_path] for rec in records if rec.rel_path in infos]

    # --- per-class counts ----------------------------------------------------
    class_counts = {}
    if layout.class_of:
        for rec in records:
            if rec.rel_path not in infos:
                continue
            label = layout.class_of.get(rec.rel_path)
            if label:
                class_counts[label] = class_counts.get(label, 0) + 1

    counts_values = list(class_counts.values())
    largest = max(class_counts.items(), key=lambda kv: kv[1]) if class_counts else ("-", 0)
    smallest = min(class_counts.items(), key=lambda kv: kv[1]) if class_counts else ("-", 0)
    total_images = sum(counts_values)
    small_threshold = max(10, total_images * 0.01) if total_images else 10
    classes_block = {
        "total": len(class_counts),
        "counts": class_counts,
        "balance": {
            "min": min(counts_values) if counts_values else 0,
            "max": max(counts_values) if counts_values else 0,
            "mean": round(float(np.mean(counts_values)), 2) if counts_values else 0.0,
            "median": float(np.median(counts_values)) if counts_values else 0.0,
            "imbalance_ratio": (max(counts_values) / min(counts_values)) if counts_values
                               and min(counts_values) > 0 else 0.0,
            "gini": round(gini_coefficient(counts_values), 3) if counts_values else 0.0,
            "largest": [largest[0], largest[1]],
            "smallest": [smallest[0], smallest[1]],
            "small_classes": [
                name for name, count in sorted(class_counts.items(), key=lambda kv: kv[1])
                if count < small_threshold
            ],
        },
    }
# --- geometry --------------------------------------------------------------
    geometry = aggregate_geometry(valid_infos)
    if geometry.get("width"):
        geometry["recommended"] = recommended_resize(
            geometry["width"], geometry["height"], geometry.get("aspect")
        )
        aspect_med = geometry.get("aspect", {}).get("median", 1.0)
        geometry["resize_strategy"] = "letterbox" if (aspect_med < 0.8 or aspect_med > 1.25) else "direct"

    aspects = []
    for info in valid_infos:
        if info["height"]:
            aspects.append(info["width"] / info["height"])

    formats = {}
    modes = {}
    exif_oriented = 0
    for info in valid_infos:
        formats[info["format"]] = formats.get(info["format"], 0) + 1
        modes[info["mode"]] = modes.get(info["mode"], 0) + 1
        if info["exif_orientation"] not in (1, None):
            exif_oriented += 1

    # --- pixel pass (sample) -------------------------------------------------
    results = []
    pixel_result = {"n": 0}
    pixel_records = []
    if not args.no_pixel_stats:
        pixel_records = _select_pixel_sample(records, infos, layout, args.pixel_sample)
        if pixel_records:
            thumb_arg = None if args.full_res else args.thumb_size
            def _pixel_fn(rec):
                return read_pixel_stats(rec, thumb=thumb_arg)
            raw = _run_parallel(pixel_records, _pixel_fn, workers, "pixels", args.silent)
            results = [raw[rec.rel_path] for rec in pixel_records
                       if not isinstance(raw[rec.rel_path], Exception)]
            if results:
                pixel_result = summarize_channels(results)

    # --- duplicates ----------------------------------------------------------
    exact_names, exact_images, exact_bytes = [], 0, 0
    near_names, near_images, near_bytes = [], 0, 0
    near_sampled = False
    leakage = []
    if not args.no_dedup and records:
        if not args.silent:
            print("  checking for exact duplicates (sha256) ...")
        exact_groups = find_exact_duplicates(records)
        exact_names, exact_images, exact_bytes = _group_repr(exact_groups)

        if not args.silent:
            print("  checking for near-duplicates (dHash) ...")
        near_groups = find_near_duplicates(
            records,
            threshold=args.dup_threshold,
            max_images=args.near_dup_cap,
            rotation_invariant=args.rotation_invariant,
        )
        near_names, near_images, near_bytes = _group_repr(near_groups)
        near_sampled = len(records) > args.near_dup_cap

        # train/val/test leakage check
        if layout.splits:
            leakage = find_split_leakage(exact_groups, near_groups, layout.splits)
    elif not args.silent:
        print("  skipping duplicate detection (--no-dedup)")

    duplicates_block = {
        "threshold": args.dup_threshold,
        "rotation_invariant": args.rotation_invariant,
        "exact_groups": len(exact_names),
        "exact_images": exact_images,
        "exact_bytes": exact_bytes,
        "exact": exact_names[:12],
        "near_groups": len(near_names),
        "near_images": near_images,
        "near_bytes": near_bytes,
        "near": near_names[:12],
        "near_sampled": near_sampled,
        "leakage": leakage,
    }

    # --- baseline model ------------------------------------------------------
    baseline = None
    if not args.no_baseline and layout.class_of:
        labels = [layout.class_of.get(rec.rel_path, rec.rel_path) for rec in records]
        if not args.silent:
            print("  running trainability baseline (scikit-learn) ...")
        baseline = run_baseline(records, labels) if records else None
        if baseline is None and not args.silent:
            print("    (skipped: sklearn not installed or dataset not suitable)")
# --- plots -----------------------------------------------------------------
    plots = []
    if not args.no_plots:
        plot_inputs = {
            "class_counts": class_counts,
            "split_rows": layout.splits,
            "class_names": layout.classes,
            "brightness": [r["brightness"] for r in results],
            "colorfulness": [r["colorfulness"] for r in results],
            "infos": [infos[rec.rel_path] for rec in pixel_records if rec.rel_path in infos],
            "aspects": [infos[rec.rel_path]["width"] / infos[rec.rel_path]["height"]
                        for rec in pixel_records if rec.rel_path in infos],
            "channel_corr": pixel_result.get("channel_corr_mean"),
            "formats": formats,
            "dup_exact_sizes": [len(group) for group in exact_names],
            "dup_near_sizes": [len(group) for group in near_names],
        }
        if not args.silent:
            print("  rendering charts ...")
        plots = build_plots(plot_inputs)

    # --- findings ------------------------------------------------------------
    images_block = {
        "total": len(valid_infos) + len(corrupt),
        "readable": len(valid_infos),
        "corrupt": corrupt,
        "disk_bytes": int(sum(rec.size_bytes for rec in records)),
        "exif_oriented": exif_oriented,
    }
    ctx = {
        "layout": {
            "kind": layout.kind,
            "description": layout.description,
            "label_source": layout.label_source,
            "label_file": layout.label_file,
            "splits": layout.splits,
            "sidecar_audit": layout.sidecar_audit,
            "group_leakage": layout.group_leakage,
            "notes": layout.notes,
        },
        "classes": classes_block,
        "images": images_block,
        "corrupt": corrupt,
        "infos": infos,
        "formats": formats,
        "aspects": aspects,
        "geometry": geometry,
        "pixels": pixel_result,
        "duplicates": duplicates_block,
        "baseline": baseline,
    }
    alerts = _compose_alerts(ctx)
    recommendations = _compose_recommendations(ctx)

    # --- SAR & Remote Sensing pass -------------------------------------------
    sar_audit = None
    if args.sar or is_sar_or_geotiff_dataset(records, root):
        if not args.silent:
            print("  running SAR & remote sensing integrity checks ...")
        sar_audit = run_sar_audit(records, layout, root, sample_size=args.pixel_sample)
        sar_alerts, sar_recs = compose_sar_alerts_and_recs(sar_audit)
        alerts.extend(sar_alerts)
        recommendations.extend(sar_recs)

    ctx["sar"] = sar_audit

    from datetime import datetime, timezone

    findings = {
        "meta": {
            "tool": "tathya",
            "tool_version": __version__,
            "root": str(root),
            "dataset_name": root.name or str(root),
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "elapsed": human_time(time.perf_counter() - start),
            "image_files": len(records),
            "non_image_files": missed_files,
            "workers": workers,
        },
        "layout": ctx["layout"],
        "classes": classes_block,
        "images": images_block,
        "geometry": geometry,
        "pixels": pixel_result,
        "formats": formats,
        "modes": modes,
        "duplicates": duplicates_block,
        "baseline": baseline,
        "sar": sar_audit,
        "alerts": alerts,
        "recommendations": recommendations,
        "plots": plots,
    }

    written = _write_reports(findings, output_dir, args.formats or [])
    _print_summary(ctx, findings, time.perf_counter() - start, output_dir)

    if not args.silent:
        for path in written:
            print(f"[tathya] wrote {path}")

    if args.open:
        html_report = output_dir / "report.html"
        if html_report.exists():
            webbrowser.open(html_report.as_uri())

    return 0
