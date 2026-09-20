"""Tests for tathya.analysis."""
from PIL import Image

from tathya.analysis import (
    aggregate_geometry,
    gini_coefficient,
    read_metadata,
    read_pixel_stats,
    recommended_resize,
    summarize_channels,
)
from tathya.scanning import ImageRecord


def _create_image(path, size=(100, 200), color=(255, 0, 0)):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=color)
    img.save(path)
    return ImageRecord(path=str(path), rel_path=path.name, size_bytes=path.stat().st_size)


def test_read_metadata(tmp_path):
    img_path = tmp_path / "test.png"
    rec = _create_image(img_path, size=(150, 100))
    meta = read_metadata(rec)

    assert meta["width"] == 150
    assert meta["height"] == 100
    assert meta["mode"] == "RGB"
    assert meta["format"] == "PNG"
    assert meta["frames"] == 1


def test_read_pixel_stats(tmp_path):
    img_path = tmp_path / "red.png"
    rec = _create_image(img_path, size=(50, 50), color=(255, 0, 0))
    stats = read_pixel_stats(rec)

    assert "channel_means" in stats
    assert "brightness" in stats
    assert "colorfulness" in stats
    # Pure red has R channel mean close to 1.0, G and B close to 0.0
    r_mean, g_mean, b_mean = stats["channel_means"]
    assert r_mean > 0.95
    assert g_mean < 0.05
    assert b_mean < 0.05


def test_aggregate_geometry_and_recommended_resize():
    infos = [
        {"width": 100, "height": 100},
        {"width": 200, "height": 200},
        {"width": 300, "height": 300},
    ]
    geo = aggregate_geometry(infos)
    assert geo["width"]["median"] == 200
    assert geo["height"]["median"] == 200
    assert geo["aspect"]["median"] == 1.0

    resize = recommended_resize(geo["width"], geo["height"])
    assert resize == "208x208" or resize == "200x200" or resize.endswith("x" + resize.split("x")[0])


def test_gini_coefficient():
    # Perfectly balanced classes -> Gini is 0.0
    assert gini_coefficient([100, 100, 100]) == 0.0

    # Heavily imbalanced
    high_gini = gini_coefficient([1, 1, 1000])
    assert high_gini > 0.6


def test_summarize_channels(tmp_path):
    img1 = tmp_path / "img1.png"
    img2 = tmp_path / "img2.png"
    rec1 = _create_image(img1, color=(100, 150, 200))
    rec2 = _create_image(img2, color=(50, 80, 110))

    res1 = read_pixel_stats(rec1)
    res2 = read_pixel_stats(rec2)
    summary = summarize_channels([res1, res2])

    assert summary["n"] == 2
    assert len(summary["channel_means"]) == 3
    assert 0.0 <= summary["brightness_mean"] <= 1.0

