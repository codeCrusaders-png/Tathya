"""Tests for SAR and Remote Sensing dataset integrity auditing (tathya.sar)."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, TiffImagePlugin

from tathya.cli import main
from tathya.sar import (
    audit_annotation_alignment,
    audit_geospatial_integrity,
    audit_polarizations,
    audit_radiometric_calibration,
    audit_spatial_leakage,
    compose_sar_alerts_and_recs,
    is_sar_or_geotiff_dataset,
)
from tathya.scanning import ImageRecord


def _create_raster(path: Path, size=(32, 32), mode="L", fill=128, tiffinfo=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "F":
        arr = np.full(size, fill, dtype=np.float32)
        im = Image.fromarray(arr, mode="F")
    else:
        im = Image.new(mode, size, color=fill)

    if tiffinfo:
        im.save(path, tiffinfo=tiffinfo)
    else:
        im.save(path)
    return ImageRecord(path=str(path), rel_path=path.name, size_bytes=path.stat().st_size)


def test_is_sar_or_geotiff_detection(tmp_path):
    # Standard natural images -> False
    rec_standard = _create_raster(tmp_path / "dog.jpg", size=(20, 20), mode="RGB")
    assert not is_sar_or_geotiff_dataset([rec_standard], tmp_path)

    # Image with polarization in filename -> True
    rec_vv = _create_raster(tmp_path / "scene01_VV.png", size=(20, 20))
    assert is_sar_or_geotiff_dataset([rec_vv], tmp_path)

    # Float32 GeoTIFF -> True
    rec_float = _create_raster(tmp_path / "scene02.tif", size=(20, 20), mode="F", fill=-12.5)
    assert is_sar_or_geotiff_dataset([rec_float], tmp_path)


def test_audit_polarizations_and_dimension_parity(tmp_path):
    # Paired dual-pol scene 1: VV and VH both 32x32
    r_s1_vv = _create_raster(tmp_path / "scene01_VV.tif", size=(32, 32))
    r_s1_vh = _create_raster(tmp_path / "scene01_VH.tif", size=(32, 32))

    # Orphan scene 2: has VV but missing VH
    r_s2_vv = _create_raster(tmp_path / "scene02_VV.tif", size=(32, 32))

    # Dimension mismatched scene 3: VV is 32x32, VH is 64x64
    r_s3_vv = _create_raster(tmp_path / "scene03_VV.tif", size=(32, 32))
    r_s3_vh = _create_raster(tmp_path / "scene03_VH.tif", size=(64, 64))

    records = [r_s1_vv, r_s1_vh, r_s2_vv, r_s3_vv, r_s3_vh]
    pols = audit_polarizations(records)

    assert set(pols["detected_polarizations"]) == {"VV", "VH"}
    assert pols["dual_pol_pairs"] >= 2  # scene01 and scene03 have both
    assert len(pols["orphan_scenes"]) == 1
    assert pols["orphan_scenes"][0]["missing"] == ["VH"]

    assert len(pols["dimension_mismatches"]) == 1
    assert "scene03" in pols["dimension_mismatches"][0]["scene"]


def test_audit_radiometric_calibration_db_and_clipping(tmp_path):
    # Create SAR float32 backscatter tile in decibels with a noise-floor clip at -30 dB
    arr = np.random.normal(loc=-15.0, scale=4.0, size=(40, 40)).astype(np.float32)
    # Clip 10% of pixels to exactly -30 dB (lower clip noise floor)
    arr.ravel()[:160] = -30.0
    # Add some NaN NoData pixels
    arr[0, :10] = np.nan

    tif_path = tmp_path / "sar_db.tif"
    im = Image.fromarray(arr, mode="F")
    im.save(tif_path)
    rec = ImageRecord(path=str(tif_path), rel_path=tif_path.name, size_bytes=tif_path.stat().st_size)

    calib = audit_radiometric_calibration([rec])
    assert "decibel_dB" in calib["scales_detected"]
    assert "NaN" in calib["nodata_types"]
    assert len(calib["clipped_files"]) == 1
    assert calib["clipped_files"][0]["type"] == "lower_clip_floor"
    assert calib["clipped_files"][0]["value"] == -30.0
    assert calib["stats"]["mean"] < 0.0


def test_audit_radiometric_calibration_linear_and_saturation(tmp_path):
    # Linear power image (strictly >= 0)
    arr_linear = np.abs(np.random.exponential(scale=0.05, size=(30, 30))).astype(np.float32)
    p_lin = tmp_path / "sar_linear.tif"
    Image.fromarray(arr_linear, mode="F").save(p_lin)
    rec_lin = ImageRecord(path=str(p_lin), rel_path=p_lin.name, size_bytes=p_lin.stat().st_size)

    # Saturated uint8 image
    arr_sat = np.full((30, 30), 255, dtype=np.uint8)
    p_sat = tmp_path / "sar_sat.png"
    Image.fromarray(arr_sat).save(p_sat)
    rec_sat = ImageRecord(path=str(p_sat), rel_path=p_sat.name, size_bytes=p_sat.stat().st_size)

    calib = audit_radiometric_calibration([rec_lin, rec_sat])
    assert "linear_power" in calib["scales_detected"]
    assert "raw_digital_number" in calib["scales_detected"]
    assert calib["mixed_calibration"] is True
    assert any(c["type"] == "saturation" for c in calib["clipped_files"])


def test_audit_geospatial_integrity(tmp_path):
    # Build GeoTIFF with EPSG:32632 (UTM zone 32N) and 10m pixel scale
    info = TiffImagePlugin.ImageFileDirectory_v2()
    info[33550] = (10.0, 10.0, 0.0)  # 10m scale
    info[33922] = (0.0, 0.0, 0.0, 500000.0, 4500000.0, 0.0)  # origin
    info[34735] = (1, 1, 0, 1, 3072, 0, 1, 32632)  # EPSG:32632

    geo_path = tmp_path / "georeferenced.tif"
    rec_geo = _create_raster(geo_path, size=(50, 50), mode="F", fill=-12.0, tiffinfo=info)

    # Unreferenced raster
    plain_path = tmp_path / "unref.png"
    rec_unref = _create_raster(plain_path, size=(50, 50))

    geo_audit = audit_geospatial_integrity([rec_geo, rec_unref])
    assert geo_audit["georeferenced_count"] == 1
    assert geo_audit["unreferenced_count"] == 1
    assert "EPSG:32632" in geo_audit["epsg_counts"]
    assert "10.0m x 10.0m" in geo_audit["pixel_scales"]

    bounds = geo_audit["bounds_map"]["georeferenced.tif"]
    assert bounds[0] == 500000.0  # min_x
    assert bounds[2] == 500500.0  # max_x (500000 + 50 * 10)


def test_audit_annotation_alignment(tmp_path):
    img_path = tmp_path / "scene_01.png"
    _create_raster(img_path, size=(64, 64))
    rec = ImageRecord(path=str(img_path), rel_path="scene_01.png", size_bytes=img_path.stat().st_size)

    # Mask with matching dimension
    mask_good = tmp_path / "scene_01_mask.png"
    _create_raster(mask_good, size=(64, 64))

    # Mask with mismatching dimension
    img2_path = tmp_path / "scene_02.png"
    _create_raster(img2_path, size=(64, 64))
    rec2 = ImageRecord(path=str(img2_path), rel_path="scene_02.png", size_bytes=img2_path.stat().st_size)

    mask_bad = tmp_path / "scene_02_mask.png"
    _create_raster(mask_bad, size=(32, 32))

    # GeoJSON out-of-bounds
    geojson_path = tmp_path / "scene_01.geojson"
    geojson_data = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[999999.0, 999999.0], [999999.0, 1000000.0], [1000000.0, 999999.0]]],
            },
        }],
    }
    geojson_path.write_text(json.dumps(geojson_data), encoding="utf-8")

    bounds_map = {"scene_01.png": [500000.0, 4400000.0, 501000.0, 4401000.0]}
    annos = audit_annotation_alignment([rec, rec2], tmp_path, bounds_map=bounds_map)

    assert annos["paired_masks"] == 2
    assert len(annos["dimension_mismatches"]) == 1
    assert "scene_02" in annos["dimension_mismatches"][0]["image"]
    assert len(annos["out_of_bounds_annotations"]) == 1
    assert "scene_01" in annos["out_of_bounds_annotations"][0]["image"]


def test_audit_spatial_leakage_and_autocorrelation():
    # Split records
    r_train = ImageRecord(path="/data/train/tile_01.tif", rel_path="train/tile_01.tif", size_bytes=100)
    r_test_overlap = ImageRecord(path="/data/test/tile_02.tif", rel_path="test/tile_02.tif", size_bytes=100)
    r_test_adjacent = ImageRecord(path="/data/test/tile_03.tif", rel_path="test/tile_03.tif", size_bytes=100)

    # 1. Direct overlap
    bounds_overlap = {
        "train/tile_01.tif": [10.0, 10.0, 20.0, 20.0],
        "test/tile_02.tif": [15.0, 15.0, 25.0, 25.0],  # overlaps [15..20, 15..20]
    }
    splits = [{"name": "train"}, {"name": "test"}]
    res_ov = audit_spatial_leakage([r_train, r_test_overlap], splits, bounds_map=bounds_overlap)
    assert res_ov["leakage_risk"] == "critical"
    assert len(res_ov["overlapping_pairs"]) == 1

    # 2. Adjacent tiles (touching edge, autocorrelation leakage)
    bounds_adj = {
        "train/tile_01.tif": [0.0, 0.0, 1.0, 1.0],
        "test/tile_03.tif": [1.0, 0.0, 2.0, 1.0],  # touching at x=1.0
    }
    res_adj = audit_spatial_leakage([r_train, r_test_adjacent], splits, bounds_map=bounds_adj)
    assert res_adj["leakage_risk"] == "high"
    assert len(res_adj["adjacent_pairs"]) == 1


def test_sar_alerts_and_recommendations_composition():
    sar_audit = {
        "polarizations": {
            "orphan_scenes": [{"scene": "scene_01", "missing": ["VH"]}],
            "dimension_mismatches": [],
        },
        "radiometry": {
            "mixed_calibration": True,
            "scales_detected": ["decibel_dB", "linear_power"],
            "excessive_nodata_files": [{"rel_path": "tile.tif", "nodata_pct": 35.0}],
            "clipped_files": [],
        },
        "geospatial": {
            "mixed_crs": True,
            "epsg_counts": {"EPSG:4326": 5, "EPSG:32632": 5},
        },
        "annotations": {
            "dimension_mismatches": [{"image": "t1.png", "mask": "m1.png"}],
            "out_of_bounds_annotations": [],
        },
        "spatial_leakage": {
            "leakage_risk": "critical",
            "overlapping_pairs": [{"train": "tr.tif", "eval": "ts.tif"}],
            "adjacent_pairs": [],
        },
    }

    alerts, recs = compose_sar_alerts_and_recs(sar_audit)
    levels = [a["level"] for a in alerts]
    assert "error" in levels
    assert "warn" in levels
    assert any("polarization" in a["message"].lower() for a in alerts)
    assert any("mixed sar radiometric" in a["message"].lower() for a in alerts)
    assert any("spatial leakage" in a["message"].lower() for a in alerts)
    assert len(recs) >= 4


def test_cli_sar_e2e(tmp_path):
    dataset_dir = tmp_path / "sar_dataset"
    # Create train and test dual-pol images
    _create_raster(dataset_dir / "train" / "tile01_VV.tif", size=(24, 24), mode="F", fill=-14.0)
    _create_raster(dataset_dir / "train" / "tile01_VH.tif", size=(24, 24), mode="F", fill=-22.0)
    _create_raster(dataset_dir / "test" / "tile02_VV.tif", size=(24, 24), mode="F", fill=-13.5)
    _create_raster(dataset_dir / "test" / "tile02_VH.tif", size=(24, 24), mode="F", fill=-21.5)

    out_dir = tmp_path / "sar_report"
    code = main([
        str(dataset_dir),
        "--sar",
        "-o", str(out_dir),
        "--formats", "html", "md", "json",
        "--silent",
    ])
    assert code == 0

    # JSON report check
    json_path = out_dir / "report.json"
    assert json_path.exists()
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    assert "sar" in data
    assert "VV" in data["sar"]["polarizations"]["detected_polarizations"]
    assert "VH" in data["sar"]["polarizations"]["detected_polarizations"]
    assert "decibel_dB" in data["sar"]["radiometry"]["scales_detected"]

    # Markdown report check
    md_content = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "## SAR & Remote Sensing Integrity" in md_content
    assert "Polarizations" in md_content

    # HTML report check
    html_content = (out_dir / "report.html").read_text(encoding="utf-8")
    assert "SAR &amp; Remote Sensing Integrity" in html_content
