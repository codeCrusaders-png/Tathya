"""SAR (Synthetic Aperture Radar) and Remote Sensing dataset auditing module.

Provides specialized integrity checks for SAR and Earth Observation datasets:
1. Polarization consistency & band matching (VV/VH/HH/HV orphans and dimensional parity).
2. Radiometric calibration validation (dB vs linear power vs raw DN, clipping, and NoData).
3. Geospatial coordinate reference system (CRS) & affine transform integrity.
4. Annotation mask & GeoJSON vector alignment.
5. Spatial autocorrelation & train/test spatial split leakage detection.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

POL_PATTERN = re.compile(r"(?:[_\-\.])(VV|VH|HH|HV)(?:[_\-\.]|$)", re.IGNORECASE)
TILE_GRID_PATTERN = re.compile(
    r"(?:tile|patch|grid|chip)?[_\-]?x(?P<x>\d+)[_\-]y(?P<y>\d+)|(?:row(?P<row>\d+)[_\-]col(?P<col>\d+))",
    re.IGNORECASE,
)


def is_sar_or_geotiff_dataset(records, root=None):
    """Detect whether the dataset contains SAR or GeoTIFF remote sensing data."""
    if not records:
        return False

    sample = records[:min(50, len(records))]
    has_pol = any(POL_PATTERN.search(rec.rel_path) for rec in sample)
    if has_pol:
        return True

    # Check for TIFF files with GeoTIFF tags or 32-bit float rasters
    for rec in sample:
        ext = os.path.splitext(rec.path)[1].lower()
        if ext in {".tif", ".tiff"}:
            try:
                with Image.open(rec.path) as im:
                    tags = getattr(im, "tag_v2", getattr(im, "tag", {}))
                    if any(t in tags for t in (33550, 33922, 34735)):
                        return True
                    if im.mode == "F":
                        return True
            except Exception:
                continue

    return False


# --------------------------------------------------------------------------- #
# 1. Polarization Consistency & Band Matching
# --------------------------------------------------------------------------- #

def audit_polarizations(records):
    """Audit polarizations (VV, VH, HH, HV) across records.

    Checks:
    - Detected polarization channels
    - Orphan scenes (e.g. VV present but VH missing in dual-pol datasets)
    - Multi-band channel counts
    - Dimension matching between co-pol and cross-pol channels
    """
    scene_pols = defaultdict(dict)  # scene_key -> {pol: rec}
    multi_band_counts = defaultdict(int)
    all_pols = set()
    unmatched_records = []

    for rec in records:
        ext = os.path.splitext(rec.path)[1].lower()
        m = POL_PATTERN.search(rec.rel_path)
        if m:
            pol = m.group(1).upper()
            all_pols.add(pol)
            # Remove polarization token to get base scene identifier
            prefix = rec.rel_path[:m.start(1) - 1]
            suffix = rec.rel_path[m.end(1):]
            scene_key = f"{prefix}__POL__{suffix}"
            scene_pols[scene_key][pol] = rec
        else:
            unmatched_records.append(rec)

        if ext in {".tif", ".tiff"}:
            try:
                with Image.open(rec.path) as im:
                    bands = len(im.getbands()) if hasattr(im, "getbands") else 1
                    multi_band_counts[bands] += 1
            except Exception:
                pass

    orphan_scenes = []
    dimension_mismatches = []
    dual_pol_pairs = 0

    # Determine expected polarization set (e.g. VV+VH or HH+HV)
    has_vv = "VV" in all_pols
    has_vh = "VH" in all_pols
    has_hh = "HH" in all_pols
    has_hv = "HV" in all_pols

    expected_sets = []
    if has_vv and has_vh:
        expected_sets.append({"VV", "VH"})
    if has_hh and has_hv:
        expected_sets.append({"HH", "HV"})

    for scene_key, pol_dict in sorted(scene_pols.items()):
        present = set(pol_dict.keys())
        for expected in expected_sets:
            overlap = present & expected
            missing = expected - present
            if overlap and missing:
                orphan_scenes.append({
                    "scene": scene_key,
                    "present": sorted(present),
                    "missing": sorted(missing),
                })
            elif overlap == expected:
                dual_pol_pairs += 1

        # Check dimension matching between polarizations of the same scene
        sizes = {}
        for pol, r in pol_dict.items():
            try:
                with Image.open(r.path) as im:
                    sizes[pol] = [im.width, im.height]
            except Exception:
                pass
        unique_sizes = {tuple(s) for s in sizes.values()}
        if len(unique_sizes) > 1:
            dimension_mismatches.append({
                "scene": scene_key,
                "dimensions": sizes,
            })

    return {
        "detected_polarizations": sorted(all_pols),
        "total_pol_scenes": len(scene_pols),
        "dual_pol_pairs": dual_pol_pairs,
        "orphan_scenes": orphan_scenes,
        "dimension_mismatches": dimension_mismatches,
        "multi_band_counts": dict(multi_band_counts),
    }


# --------------------------------------------------------------------------- #
# 2. Radiometric Calibration & Backscatter Validation
# --------------------------------------------------------------------------- #

def audit_radiometric_calibration(records, sample_size=100):
    """Audit SAR backscatter intensity, calibration domain, clipping, and NoData.

    Checks:
    - Scale / domain: Decibel (dB) vs Linear power vs Digital Numbers (DN)
    - Mixed calibration across tiles
    - NoData percentages (NaN, -9999, -32768)
    - Lower clipping (e.g. artificial noise floors at -30 dB)
    - Upper saturation
    """
    usable = records[:min(sample_size, len(records))]
    scales_detected = set()
    clipped_files = []
    excessive_nodata_files = []
    nodata_types = set()
    all_valid_pixels = []

    for rec in usable:
        try:
            with Image.open(rec.path) as im:
                arr = np.asarray(im)
        except Exception:
            continue

        if arr.size == 0:
            continue

        # Check NoData
        total_px = arr.size
        nodata_mask = np.zeros(arr.shape, dtype=bool)

        if np.issubdtype(arr.dtype, np.floating):
            isnan = np.isnan(arr)
            isinf = np.isinf(arr)
            if isnan.any():
                nodata_types.add("NaN")
                nodata_mask |= isnan
            if isinf.any():
                nodata_types.add("Inf")
                nodata_mask |= isinf
            is_neg9999 = (arr == -9999.0)
            if is_neg9999.any():
                nodata_types.add("-9999")
                nodata_mask |= is_neg9999
        elif np.issubdtype(arr.dtype, np.integer):
            is_neg9999 = (arr == -9999)
            if is_neg9999.any():
                nodata_types.add("-9999")
                nodata_mask |= is_neg9999
            is_neg32768 = (arr == -32768)
            if is_neg32768.any():
                nodata_types.add("-32768")
                nodata_mask |= is_neg32768

        nodata_count = int(nodata_mask.sum())
        nodata_pct = nodata_count / total_px
        if nodata_pct > 0.25:
            excessive_nodata_files.append({
                "rel_path": rec.rel_path,
                "nodata_pct": round(nodata_pct * 100, 1),
            })

        valid = arr[~nodata_mask]
        if valid.size == 0:
            continue

        # Subsample for global distribution statistics
        step = max(1, valid.size // 500)
        sample_px = valid.ravel()[::step]
        all_valid_pixels.extend(sample_px.tolist())

        # Determine scale
        v_min, v_max, v_mean = float(valid.min()), float(valid.max()), float(valid.mean())
        if np.issubdtype(arr.dtype, np.floating):
            if v_min < 0 and v_mean < 0 and v_max < 25.0:
                scales_detected.add("decibel_dB")
                # Lower clipping check (e.g. constant noise floor clamping)
                lower_count = (valid == v_min).sum()
                if lower_count / valid.size > 0.05 and v_min < -20.0:
                    clipped_files.append({
                        "rel_path": rec.rel_path,
                        "type": "lower_clip_floor",
                        "value": round(v_min, 2),
                        "fraction": round(float(lower_count / valid.size), 3),
                    })
            elif v_min >= 0.0:
                scales_detected.add("linear_power")
        else:
            scales_detected.add("raw_digital_number")
            # Saturation check for uint8/uint16
            sat_val = 255 if arr.dtype == np.uint8 else (65535 if arr.dtype == np.uint16 else None)
            if sat_val is not None:
                sat_count = (valid == sat_val).sum()
                if sat_count / valid.size > 0.02:
                    clipped_files.append({
                        "rel_path": rec.rel_path,
                        "type": "saturation",
                        "value": sat_val,
                        "fraction": round(float(sat_count / valid.size), 3),
                    })

    stats = {}
    if all_valid_pixels:
        arr_all = np.array(all_valid_pixels, dtype=np.float32)
        stats = {
            "min": round(float(arr_all.min()), 2),
            "max": round(float(arr_all.max()), 2),
            "mean": round(float(arr_all.mean()), 2),
            "std": round(float(arr_all.std()), 2),
            "p10": round(float(np.percentile(arr_all, 10)), 2),
            "p50": round(float(np.percentile(arr_all, 50)), 2),
            "p90": round(float(np.percentile(arr_all, 90)), 2),
        }

    return {
        "scales_detected": sorted(scales_detected),
        "mixed_calibration": len(scales_detected) > 1,
        "nodata_types": sorted(nodata_types),
        "excessive_nodata_files": excessive_nodata_files[:10],
        "clipped_files": clipped_files[:10],
        "stats": stats,
    }


# --------------------------------------------------------------------------- #
# 3. Geospatial CRS & Affine Transform Integrity
# --------------------------------------------------------------------------- #

def _parse_geotiff_metadata(path):
    """Extract EPSG code, pixel scale, tiepoint origin, and bounding box."""
    with Image.open(path) as im:
        tags = getattr(im, "tag_v2", getattr(im, "tag", {}))
        width, height = im.width, im.height

    pixel_scale = tags.get(33550)  # (scale_x, scale_y, scale_z)
    tiepoint = tags.get(33922)     # (i, j, k, x, y, z)
    geo_keys = tags.get(34735)     # GeoKeyDirectoryTag

    if not pixel_scale or not tiepoint:
        return None

    sx, sy = float(pixel_scale[0]), float(pixel_scale[1])
    ox, oy = float(tiepoint[3]), float(tiepoint[4])

    min_x = ox
    max_x = ox + width * sx
    max_y = oy
    min_y = oy - height * sy

    # Parse EPSG code from GeoKeyDirectoryTag
    epsg = None
    if geo_keys and len(geo_keys) >= 4:
        num_keys = geo_keys[3]
        for idx in range(num_keys):
            offset = 4 + idx * 4
            if offset + 4 <= len(geo_keys):
                key_id, loc, count, val = geo_keys[offset : offset + 4]
                # Key 3072 is ProjectedCSTypeGeoKey, Key 2048 is GeographicTypeGeoKey
                if key_id in (3072, 2048) and loc == 0:
                    epsg = val
                    break

    return {
        "epsg": epsg,
        "pixel_scale": (round(sx, 4), round(sy, 4)),
        "bounds": [round(min_x, 4), round(min_y, 4), round(max_x, 4), round(max_y, 4)],
        "width": width,
        "height": height,
    }


def audit_geospatial_integrity(records, sample_size=100):
    """Audit GeoTIFF geospatial metadata, CRS consistency, and pixel scale."""
    georeferenced = {}
    unreferenced = []
    epsg_counts = defaultdict(int)
    scale_counts = defaultdict(int)

    sample = records[:min(sample_size, len(records))]
    for rec in sample:
        try:
            meta = _parse_geotiff_metadata(rec.path)
            if meta:
                georeferenced[rec.rel_path] = meta
                epsg_label = f"EPSG:{meta['epsg']}" if meta["epsg"] else "Unknown CRS"
                epsg_counts[epsg_label] += 1
                scale_str = f"{meta['pixel_scale'][0]}m x {meta['pixel_scale'][1]}m"
                scale_counts[scale_str] += 1
            else:
                unreferenced.append(rec.rel_path)
        except Exception:
            unreferenced.append(rec.rel_path)

    mixed_crs = len(epsg_counts) > 1

    return {
        "georeferenced_count": len(georeferenced),
        "unreferenced_count": len(unreferenced),
        "epsg_counts": dict(epsg_counts),
        "mixed_crs": mixed_crs,
        "pixel_scales": dict(scale_counts),
        "bounds_map": {rel: meta["bounds"] for rel, meta in georeferenced.items()},
    }


# --------------------------------------------------------------------------- #
# 4. Annotation Mask & GeoJSON Vector Alignment
# --------------------------------------------------------------------------- #

def audit_annotation_alignment(records, root, bounds_map=None):
    """Audit paired segmentation masks and GeoJSON vector annotations."""
    root = Path(root)
    img_by_base = {}
    for rec in records:
        base = os.path.splitext(Path(rec.rel_path).name)[0]
        # normalize base name (remove _VV/_VH/etc)
        base_clean = POL_PATTERN.sub("", base).rstrip("_-")
        img_by_base[base_clean] = rec
        img_by_base[base] = rec

    # Search for mask and vector files
    mask_files = list(root.rglob("*mask*.*")) + list(root.rglob("*label*.*"))
    geojson_files = list(root.rglob("*.geojson")) + list(root.rglob("*.json"))

    dimension_mismatches = []
    paired_masks = 0
    paired_geojsons = 0
    out_of_bounds = []

    # Check raster masks
    for mf in mask_files:
        if mf.suffix.lower() not in {".png", ".tif", ".tiff", ".bmp"}:
            continue
        m_base = os.path.splitext(mf.name)[0]
        m_clean = re.sub(r"(?:_mask|_label)", "", m_base, flags=re.IGNORECASE)
        matching_rec = img_by_base.get(m_clean)
        if matching_rec:
            paired_masks += 1
            try:
                with Image.open(mf) as m_im, Image.open(matching_rec.path) as i_im:
                    if (m_im.width, m_im.height) != (i_im.width, i_im.height):
                        dimension_mismatches.append({
                            "image": matching_rec.rel_path,
                            "mask": str(mf.relative_to(root)).replace("\\", "/"),
                            "image_size": [i_im.width, i_im.height],
                            "mask_size": [m_im.width, m_im.height],
                        })
            except Exception:
                pass

    # Check GeoJSON vectors
    for gf in geojson_files:
        if gf.name in {"config.json", "report.json", "metadata.json"}:
            continue
        g_base = os.path.splitext(gf.name)[0]
        g_clean = POL_PATTERN.sub("", g_base).rstrip("_-")
        matching_rec = img_by_base.get(g_clean)
        if matching_rec:
            paired_geojsons += 1
            if bounds_map and matching_rec.rel_path in bounds_map:
                b_min_x, b_min_y, b_max_x, b_max_y = bounds_map[matching_rec.rel_path]
                try:
                    with open(gf, "r", encoding="utf-8") as handle:
                        geo_data = json.load(handle)
                    coords = _extract_geojson_coordinates(geo_data)
                    for x, y in coords:
                        if not (b_min_x <= x <= b_max_x and b_min_y <= y <= b_max_y):
                            out_of_bounds.append({
                                "image": matching_rec.rel_path,
                                "geojson": str(gf.relative_to(root)).replace("\\", "/"),
                                "point": [round(x, 4), round(y, 4)],
                                "bounds": bounds_map[matching_rec.rel_path],
                            })
                            break
                except Exception:
                    pass

    return {
        "paired_masks": paired_masks,
        "paired_geojsons": paired_geojsons,
        "dimension_mismatches": dimension_mismatches,
        "out_of_bounds_annotations": out_of_bounds,
    }


def _extract_geojson_coordinates(data):
    """Recursively extract all [x, y] coordinates from a GeoJSON structure."""
    coords = []
    if isinstance(data, dict):
        if data.get("type") in ("Point", "MultiPoint", "LineString", "MultiLineString", "Polygon", "MultiPolygon"):
            c = data.get("coordinates", [])
            _flatten_coords(c, coords)
        elif "features" in data:
            for feat in data["features"]:
                coords.extend(_extract_geojson_coordinates(feat))
        elif "geometry" in data:
            coords.extend(_extract_geojson_coordinates(data["geometry"]))
    return coords


def _flatten_coords(val, out):
    if isinstance(val, (list, tuple)):
        if len(val) >= 2 and isinstance(val[0], (int, float)) and isinstance(val[1], (int, float)):
            out.append((float(val[0]), float(val[1])))
        else:
            for item in val:
                _flatten_coords(item, out)


# --------------------------------------------------------------------------- #
# 5. Spatial Autocorrelation & Train/Test Spatial Split Leakage
# --------------------------------------------------------------------------- #

def audit_spatial_leakage(records, splits, bounds_map=None):
    """Detect spatial overlap and spatial autocorrelation leakage across splits.

    Spatial autocorrelation occurs when adjacent or overlapping scenes are randomly
    assigned to train and test splits, causing test metrics to overfit local textures.
    """
    if not splits or len(splits) < 2:
        return {"leakage_risk": "none", "overlapping_pairs": [], "adjacent_pairs": []}

    split_records = defaultdict(list)
    for rec in records:
        parts = rec.rel_path.split("/")
        if len(parts) >= 2:
            split_records[parts[0]].append(rec)

    split_names = list(split_records.keys())
    if len(split_names) < 2:
        return {"leakage_risk": "none", "overlapping_pairs": [], "adjacent_pairs": []}

    # Derive bounding boxes: from bounds_map or from filename grid coordinates
    computed_bounds = {}
    if bounds_map:
        computed_bounds.update(bounds_map)

    for rec in records:
        if rec.rel_path not in computed_bounds:
            m = TILE_GRID_PATTERN.search(rec.rel_path)
            if m:
                x = int(m.group("x") or m.group("col") or 0)
                y = int(m.group("y") or m.group("row") or 0)
                # Unit grid tile bounding box
                computed_bounds[rec.rel_path] = [float(x), float(y), float(x + 1), float(y + 1)]

    overlapping_pairs = []
    adjacent_pairs = []

    # Check pairwise splits (e.g. train vs test or train vs val)
    train_recs = split_records.get("train", [])
    eval_recs = split_records.get("test", []) or split_records.get("val", [])

    for r_train in train_recs:
        b1 = computed_bounds.get(r_train.rel_path)
        if not b1:
            continue
        for r_eval in eval_recs:
            b2 = computed_bounds.get(r_eval.rel_path)
            if not b2:
                continue

            # Check overlap
            overlap_x = max(0.0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
            overlap_y = max(0.0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
            area = overlap_x * overlap_y
            if area > 0:
                overlapping_pairs.append({
                    "train": r_train.rel_path,
                    "eval": r_eval.rel_path,
                    "overlap_area": round(area, 2),
                })
            else:
                # Check adjacency / distance
                dist_x = max(0.0, max(b1[0], b2[0]) - min(b1[2], b2[2]))
                dist_y = max(0.0, max(b1[1], b2[1]) - min(b1[3], b2[3]))
                dist = math.hypot(dist_x, dist_y)
                # If touching or distance < threshold (0 or < 500m)
                if dist == 0.0 or (dist <= 1.0 and abs(b1[2] - b1[0]) <= 2.0):
                    adjacent_pairs.append({
                        "train": r_train.rel_path,
                        "eval": r_eval.rel_path,
                        "distance": round(dist, 2),
                    })

    risk = "none"
    if overlapping_pairs:
        risk = "critical"
    elif adjacent_pairs:
        risk = "high"

    return {
        "leakage_risk": risk,
        "overlapping_pairs": overlapping_pairs[:10],
        "adjacent_pairs": adjacent_pairs[:10],
    }


# --------------------------------------------------------------------------- #
# High-Level SAR Audit Orchestration
# --------------------------------------------------------------------------- #

def run_sar_audit(records, layout, root, sample_size=100):
    """Run all SAR and Remote Sensing integrity audits."""
    pols = audit_polarizations(records)
    calib = audit_radiometric_calibration(records, sample_size=sample_size)
    geo = audit_geospatial_integrity(records, sample_size=sample_size)
    annos = audit_annotation_alignment(records, root, bounds_map=geo.get("bounds_map"))
    spatial = audit_spatial_leakage(records, layout.splits, bounds_map=geo.get("bounds_map"))

    return {
        "polarizations": pols,
        "radiometry": calib,
        "geospatial": geo,
        "annotations": annos,
        "spatial_leakage": spatial,
    }


def compose_sar_alerts_and_recs(sar_audit):
    """Compose actionable alerts and recommendations from a SAR audit result."""
    alerts = []
    recs = []

    pols = sar_audit["polarizations"]
    calib = sar_audit["radiometry"]
    geo = sar_audit["geospatial"]
    annos = sar_audit["annotations"]
    spatial = sar_audit["spatial_leakage"]

    # 1. Polarizations
    if pols.get("orphan_scenes"):
        n_orphans = len(pols["orphan_scenes"])
        alerts.append({
            "level": "warn",
            "message": f"SAR polarization mismatch: {n_orphans} scene(s) have missing dual-pol pairs "
                       f"(e.g. {pols['orphan_scenes'][0]['scene']} missing "
                       f"{','.join(pols['orphan_scenes'][0]['missing'])}).",
        })
        recs.append("Ensure dual-pol SAR scenes have paired co-pol and cross-pol channels before training.")

    if pols.get("dimension_mismatches"):
        n_dim = len(pols["dimension_mismatches"])
        alerts.append({
            "level": "error",
            "message": f"SAR channel dimension mismatch: {n_dim} scene(s) have differing resolutions across polarizations.",
        })
        recs.append("Resample all SAR polarization channels of the same scene to identical spatial dimensions.")

    # 2. Calibration
    if calib.get("mixed_calibration"):
        scales = ", ".join(calib["scales_detected"])
        alerts.append({
            "level": "error",
            "message": f"Mixed SAR radiometric calibration: dataset contains mixed scales ({scales}) — models will suffer severe domain distortion.",
        })
        recs.append("Convert all SAR imagery into a consistent calibration domain (e.g. all in decibels dB or all in linear power).")

    if calib.get("excessive_nodata_files"):
        n_nodata = len(calib["excessive_nodata_files"])
        alerts.append({
            "level": "warn",
            "message": f"{n_nodata} SAR tile(s) contain >25% NoData pixels (orbital boundary masking).",
        })
        recs.append("Filter or mask border NoData pixels in SAR tiles to prevent loss functions overfitting zero/NaN borders.")

    if calib.get("clipped_files"):
        n_clipped = len(calib["clipped_files"])
        alerts.append({
            "level": "warn",
            "message": f"{n_clipped} SAR tile(s) exhibit radiometric lower-clipping or upper saturation.",
        })

    # 3. Geospatial CRS
    if geo.get("mixed_crs"):
        crs_list = ", ".join(geo["epsg_counts"].keys())
        alerts.append({
            "level": "error",
            "message": f"Mixed geospatial Coordinate Reference Systems: {crs_list} — spatial coordinates cannot be compared directly.",
        })
        recs.append("Reproject all geospatial rasters to a uniform CRS (e.g. UTM or EPSG:4326) prior to modeling.")

    # 4. Annotations
    if annos.get("dimension_mismatches"):
        n_mismatch = len(annos["dimension_mismatches"])
        alerts.append({
            "level": "error",
            "message": f"{n_mismatch} annotation mask(s) do not match image pixel dimensions.",
        })
        recs.append("Rescale annotation masks to exactly match the dimensions of their corresponding SAR images.")

    if annos.get("out_of_bounds_annotations"):
        n_oob = len(annos["out_of_bounds_annotations"])
        alerts.append({
            "level": "error",
            "message": f"{n_oob} vector GeoJSON annotation(s) contain coordinates outside their image's bounding box.",
        })
        recs.append("Verify GeoJSON vector projection matches the raster's CRS and bounds.")

    # 5. Spatial Split Leakage
    if spatial.get("leakage_risk") == "critical":
        n_ov = len(spatial["overlapping_pairs"])
        alerts.append({
            "level": "error",
            "message": f"Critical spatial leakage: {n_ov} pair(s) of tiles across train and test splits have overlapping geographical footprints!",
        })
        recs.append("Re-split spatial datasets using geographically disjoint regions (buffered spatial splits) to eliminate spatial leakage.")
    elif spatial.get("leakage_risk") == "high":
        n_adj = len(spatial["adjacent_pairs"])
        alerts.append({
            "level": "warn",
            "message": f"Spatial autocorrelation risk: {n_adj} adjacent tile(s) touch across train and test splits without a spatial buffer.",
        })
        recs.append("Implement a spatial buffer zone (minimum 1 km) between train and test/validation tiles to combat spatial autocorrelation.")

    return alerts, recs
