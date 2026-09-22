"""Dataset structure discovery and layout detection."""
from __future__ import annotations

import csv
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .utils import (
    GENERIC_DIRNAMES,
    IMAGE_EXTENSIONS,
    NOISE_DIRNAMES,
    SPLIT_DIRNAMES,
)

LABEL_FILE_NAMES = {
    "labels.csv", "label.csv", "metadata.csv", "meta.csv", "classes.csv",
    "labels.txt", "classnames.txt", "class_names.txt",
}


@dataclass
class ImageRecord:
    path: str
    rel_path: str        # posix-style path relative to the dataset root
    size_bytes: int


@dataclass
class Layout:
    kind: str = "unknown"
    description: str = ""
    label_source: str = "none"
    classes: list = field(default_factory=list)
    splits: list = field(default_factory=list)    # [{name, count, n_classes, class_counts}]
    class_of: dict = field(default_factory=dict)  # rel_path -> class label
    label_file: str or None = None
    sidecar_audit: dict = field(default_factory=dict)
    group_leakage: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def discover_images(root, exclude_dirs=None):
    """Walk ``root`` collecting every image file.

    Parameters
    ----------
    root : str or Path
        Root directory to scan.
    exclude_dirs : list of (str or Path), optional
        Directories to exclude from scanning (e.g., report output directory).

    Returns
    -------
    (records, missed_files) : (list of ImageRecord, int)
    """
    root = Path(root).resolve()
    resolved_excludes = set()
    if exclude_dirs:
        for ed in exclude_dirs:
            if ed is not None:
                try:
                    resolved_excludes.add(str(Path(ed).resolve()))
                except Exception:
                    pass

    records = []
    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda _: None):
        # Exclude known noise and explicitly excluded paths (like output report folders)
        current_dir = str(Path(dirpath).resolve())
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".")
            and d not in NOISE_DIRNAMES
            and str(Path(dirpath, d).resolve()) not in resolved_excludes
        ]
        if current_dir in resolved_excludes:
            continue

        for filename in filenames:
            if filename.startswith("."):
                continue
            ext = os.path.splitext(filename)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                full = os.path.join(dirpath, filename)
                try:
                    rel = os.path.relpath(full, root).replace("\\", "/")
                    size = os.path.getsize(full)
                except OSError:
                    size = 0
                    rel = filename
                records.append(ImageRecord(path=full, rel_path=rel, size_bytes=size))

    records.sort(key=lambda rec: rec.rel_path)
    missed = _count_non_image_files(root, exclude_dirs=resolved_excludes)
    return records, missed


def _count_non_image_files(root, exclude_dirs=None):
    count = 0
    resolved_excludes = set(exclude_dirs) if exclude_dirs else set()
    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda _: None):
        current_dir = str(Path(dirpath).resolve())
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".")
            and d not in NOISE_DIRNAMES
            and str(Path(dirpath, d).resolve()) not in resolved_excludes
        ]
        if current_dir in resolved_excludes:
            continue

        for filename in filenames:
            if filename.startswith("."):
                continue
            ext = os.path.splitext(filename)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                count += 1
    return count


def detect_layout(records, root=None):
    """Infer the dataset layout from the discovered records."""
    layout = Layout()
    if not records:
        layout.kind = "empty"
        layout.description = "No image files were found under the given folder."
        return layout

    parts_list = [rec.rel_path.split("/")[:-1] for rec in records]
    depths = {len(parts) for parts in parts_list}
    depth = max(depths)
    top_dirs = {parts[0].lower() for parts in parts_list if parts}
    all_split_names = {s.lower() for s in SPLIT_DIRNAMES}
    tops_are_splits = bool(top_dirs) and top_dirs <= all_split_names

    if depth == 0:
        layout.kind = "flat"
        layout.description = (
            "Images sit directly in the root folder with no class sub-folders."
        )
    elif depths == {1}:
        if tops_are_splits:
            layout.kind = "split_flat"
            layout.description = (
                "Top-level folders look like train/test splits but contain images "
                "directly (no class sub-folders)."
            )
        else:
            layout.kind = "class_folders"
            layout.description = "One sub-folder per class; labels come from folder names."
    elif depths == {2}:
        if tops_are_splits:
            layout.kind = "split_class_folders"
            layout.description = "Top-level train/val/test splits with one folder per class inside."
        else:
            layout.kind = "class_folders"
            layout.description = (
                "Two-level sub-folder structure; the deepest folder name is used as the "
                "class label."
            )
    else:
        layout.kind = "irregular"
        layout.description = (
            "Mixed or nested folder depths; the deepest non-generic folder name is used "
            "as the class label."
        )
    # Optional label sidecar file (labels.csv / ...)
    group_mapping = {}
    if root is not None:
        label_res = _load_label_file(root, records=records)
        if label_res is not None:
            label_path, mapping, group_mapping, sidecar_audit = label_res
            layout.label_file = str(label_path)
            layout.label_source = f"{label_path.name}"
            layout.sidecar_audit = sidecar_audit
            # Populate class_of for all records matching the label file (rel_path or basename)
            for rec in records:
                rel = rec.rel_path
                base = Path(rel).name
                if rel in mapping:
                    layout.class_of[rel] = mapping[rel]
                elif base in mapping:
                    layout.class_of[rel] = mapping[base]
            if layout.kind == "flat":
                layout.kind = "label_file"
                layout.description = (
                    f"Images sit directly in the root folder; labels were loaded from "
                    f"{label_path.name}."
                )
            elif layout.class_of:
                layout.notes.append(f"Labels loaded from sidecar file '{label_path.name}' taking precedence over folder names.")
            if sidecar_audit.get("missing_on_disk"):
                layout.notes.append(f"Sidecar file lists {len(sidecar_audit['missing_on_disk'])} file(s) not found on disk.")
            if sidecar_audit.get("null_or_empty_labels"):
                layout.notes.append(f"Sidecar file contains {len(sidecar_audit['null_or_empty_labels'])} empty or null label entries.")

    if layout.kind in {"class_folders", "split_class_folders", "split_flat", "irregular"}:
        for rec, parts in zip(records, parts_list):
            if layout.kind == "split_class_folders":
                label = _deepest_label(parts[1:]) or parts[1]
            elif layout.kind == "split_flat":
                label = _deepest_label(parts) or parts[0]
            else:
                label = _deepest_label(parts) or parts[-1]
            layout.class_of.setdefault(rec.rel_path, label)

    finalize_layout(layout, records)
    if layout.splits:
        layout.group_leakage = detect_group_leakage(records, layout.splits, group_map=group_mapping)
        if layout.group_leakage:
            layout.notes.append(f"Group/subject leakage detected across splits for {len(layout.group_leakage)} group(s).")
    return layout


def finalize_layout(layout, records):
    """Fill in class lists / split tables after labels have been assigned."""
    if not records:
        return layout

    if layout.kind in {"flat"}:
        layout.classes = []
        return layout

    labels = []
    for rec in records:
        lab = layout.class_of.get(rec.rel_path)
        if lab is None:
            lab = layout.class_of.get(Path(rec.rel_path).name)
        if lab is None:
            lab = _label_for_rel_path(rec.rel_path, layout.kind)
        if lab:
            layout.class_of[rec.rel_path] = lab
        labels.append(lab)

    unique = sorted({lab for lab in labels if lab})
    layout.classes = unique

    if layout.kind in {"split_class_folders", "split_flat"}:
        split_names = _ordered_splits(records)
        for sname in split_names:
            s_records = [r for r in records if r.rel_path.split("/")[0] == sname]
            s_labels = [layout.class_of.get(r.rel_path) or r.rel_path for r in s_records]
            counts = {}
            for lab in s_labels:
                counts[lab] = counts.get(lab, 0) + 1
            layout.splits.append({
                "name": sname,
                "count": len(s_records),
                "n_classes": len(counts),
                "class_counts": counts,
            })

    # Notes for structural oddities
    if layout.kind in {"class_folders", "irregular"} and len(layout.classes) == 1:
        layout.notes.append("Only one class folder was found — this looks like a single-class "
                            "(anomaly detection) or unlabeled image collection.")
    if layout.kind == "irregular":
        layout.notes.append("Folder depths are mixed; consider normalising the layout before training.")
    return layout


def _deepest_label(parts):
    """Pick the deepest, most informative folder name as the class label."""
    for part in reversed(parts):
        if part.lower() not in GENERIC_DIRNAMES:
            return part
    return parts[-1] if parts else None


def _label_for_rel_path(rel_path, kind):
    parts = rel_path.split("/")[:-1]
    if kind in {"split_class_folders"}:
        return _deepest_label(parts[1:]) or (parts[1] if len(parts) > 1 else parts[0])
    return _deepest_label(parts) or (parts[-1] if parts else rel_path)


def _ordered_splits(records):
    seen = []
    for rec in records:
        name = rec.rel_path.split("/")[0]
        if name not in seen:
            seen.append(name)
    return seen


# --------------------------------------------------------------------------- #
# Label sidecar files (labels.csv / classes.csv / ...)
# --------------------------------------------------------------------------- #

def _load_label_file(root, records=None):
    """Look for a CSV listing image -> label and return (path, mapping, group_mapping, sidecar_audit) or None."""
    root = Path(root)
    candidates = []
    for name in sorted(LABEL_FILE_NAMES):
        candidate = root / name
        if candidate.is_file():
            candidates.append(candidate)
    candidates.extend(sorted(p for p in root.glob("*.csv") if p.is_file()))
    for label_path in candidates:
        mapping, group_mapping, sidecar_audit = _parse_label_file(label_path, records=records)
        if mapping:
            return label_path, mapping, group_mapping, sidecar_audit
    return None


def _parse_label_file(path, records=None):
    """Parse a CSV with (file, label) columns and audit its contents.

    Returns ``(mapping, group_mapping, sidecar_audit)``.
    """
    mapping = {}
    group_mapping = {}
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
    except (OSError, UnicodeDecodeError):
        return mapping, group_mapping, {}
    if not rows:
        return mapping, group_mapping, {}

    header = [cell.strip().lower() for cell in rows[0]]
    file_col = label_col = group_col = None
    for idx, name in enumerate(header):
        if file_col is None and name in {
            "file", "filename", "image", "img", "name", "path",
            "image_name", "file_name", "image_path", "files", "filepath",
        }:
            file_col = idx
        if label_col is None and name in {
            "label", "class", "classes", "category", "target", "type", "y", "label_name",
        }:
            label_col = idx
        if group_col is None and name in {
            "group", "group_id", "subject", "subject_id", "patient", "patient_id",
            "scene", "scene_id", "case", "case_id", "donor", "donor_id", "person",
            "person_id", "user", "user_id",
        }:
            group_col = idx

    if file_col is None and len(rows[0]) < 2:
        return mapping, group_mapping, {}

    null_or_empty_labels = []
    duplicate_entries = []
    seen_files = set()
    csv_files = []

    for row in rows[1:]:
        if not row or not any(cell.strip() for cell in row):
            continue
        fname = row[file_col if file_col is not None else 0].strip().replace("\\", "/")
        if not fname:
            continue
        csv_files.append(fname)
        if fname in seen_files:
            duplicate_entries.append(fname)
        seen_files.add(fname)

        label = ""
        if label_col is not None:
            label = row[label_col].strip() if len(row) > label_col else ""
        elif len(row) > 1:
            label = row[1].strip()

        if not label:
            null_or_empty_labels.append(fname)
        else:
            mapping[fname] = label

        if group_col is not None and len(row) > group_col:
            grp = row[group_col].strip()
            if grp:
                group_mapping[fname] = grp

    missing_on_disk = []
    unreferenced_on_disk = []
    if records:
        disk_rels = {r.rel_path for r in records}
        disk_bases = {Path(r.rel_path).name for r in records}
        csv_bases = {Path(f).name for f in csv_files}
        csv_all = set(csv_files) | csv_bases

        for f in csv_files:
            if f not in disk_rels and Path(f).name not in disk_bases:
                missing_on_disk.append(f)

        for r in records:
            if r.rel_path not in csv_all and Path(r.rel_path).name not in csv_all:
                unreferenced_on_disk.append(r.rel_path)

    sidecar_audit = {
        "total_rows": len(rows) - 1,
        "valid_labels": len(mapping),
        "missing_on_disk": missing_on_disk,
        "unreferenced_on_disk": unreferenced_on_disk,
        "null_or_empty_labels": null_or_empty_labels,
        "duplicate_entries": duplicate_entries,
        "group_column": header[group_col] if group_col is not None else None,
    }
    return mapping, group_mapping, sidecar_audit


_GROUP_PREFIX_RE = re.compile(
    r"^(patient\d+|subject\d+|scene\d+|case\d+|donor\d+|user\d+|p\d+|s\d+|[a-zA-Z]+_\d+)[_\-]",
    re.IGNORECASE,
)


def detect_group_leakage(records, splits, group_map=None):
    """Detect subject/group leakage across dataset splits.

    Detects when images with the same subject/patient/scene ID appear in multiple
    splits (e.g. both train and test/validation). Identifiers are pulled from
    ``group_map`` (from sidecar metadata) or inferred from filename prefixes.

    Returns a list of dicts:
    [{'group': str, 'splits': [str, ...], 'images': [str, ...], 'count': int}]
    """
    if not splits or len(splits) < 2:
        return []

    group_records = defaultdict(list)

    for rec in records:
        parts = rec.rel_path.split("/")
        if len(parts) < 2:
            continue
        split_name = parts[0]
        base = Path(rec.rel_path).name

        group_id = None
        if group_map:
            group_id = group_map.get(rec.rel_path) or group_map.get(base)

        if not group_id:
            m = _GROUP_PREFIX_RE.match(base)
            if m:
                group_id = m.group(1).lower()

        if group_id:
            group_records[group_id].append((split_name, rec.rel_path))

    leakage = []
    for group_id, items in sorted(group_records.items()):
        split_names = {item[0] for item in items}
        if len(split_names) > 1:
            leakage.append({
                "group": group_id,
                "splits": sorted(split_names),
                "images": [item[1] for item in items],
                "count": len(items),
            })

    return leakage
