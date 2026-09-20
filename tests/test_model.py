"""Tests for tathya.model (trainability baseline probe)."""
from pathlib import Path

from PIL import Image

from tathya.model import run_baseline
from tathya.scanning import ImageRecord


def _create_image(path: Path, size=(32, 32), color="blue"):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=color)
    img.save(path)
    return ImageRecord(path=str(path), rel_path=path.name, size_bytes=path.stat().st_size)


def test_baseline_small_dataset(tmp_path):
    # 24 samples (12 class A, 12 class B) - tests PCA dynamic n_components sizing
    records, labels = [], []
    for i in range(12):
        rec_a = _create_image(tmp_path / "a" / f"img_{i}.png", color="red")
        records.append(rec_a)
        labels.append("class_a")
    for i in range(12):
        rec_b = _create_image(tmp_path / "b" / f"img_{i}.png", color="blue")
        records.append(rec_b)
        labels.append("class_b")

    result = run_baseline(records, labels)
    assert result is not None
    assert result["images_used"] == 24
    assert result["classes_used"] == 2
    assert result["cv_folds"] == 3
    assert 0.0 <= result["accuracy_mean"] <= 1.0
    assert result["majority_baseline"] == 0.5
    assert "PCA(" in result["feature_pipeline"]


def test_baseline_too_few_images(tmp_path):
    # Under minimum threshold of 20 images -> returns None
    records, labels = [], []
    for i in range(8):
        rec = _create_image(tmp_path / f"img_{i}.png")
        records.append(rec)
        labels.append("class_a" if i < 4 else "class_b")

    result = run_baseline(records, labels)
    assert result is None


def test_baseline_single_class(tmp_path):
    # 24 images all belonging to class_a -> returns None
    records, labels = [], []
    for i in range(24):
        rec = _create_image(tmp_path / f"img_{i}.png")
        records.append(rec)
        labels.append("class_a")

    result = run_baseline(records, labels)
    assert result is None


def test_baseline_handles_corrupted_images(tmp_path):
    # 26 images total, 2 corrupt -> 24 valid images should run baseline successfully
    records, labels = [], []
    for i in range(12):
        rec = _create_image(tmp_path / "a" / f"img_{i}.png", color="green")
        records.append(rec)
        labels.append("class_a")
    for i in range(12):
        rec = _create_image(tmp_path / "b" / f"img_{i}.png", color="yellow")
        records.append(rec)
        labels.append("class_b")

    # Add 2 corrupt files
    c1 = tmp_path / "a" / "corrupt1.png"
    c2 = tmp_path / "b" / "corrupt2.png"
    c1.write_bytes(b"corrupt image bytes")
    c2.write_bytes(b"corrupt image bytes")
    records.extend([
        ImageRecord(path=str(c1), rel_path=c1.name, size_bytes=c1.stat().st_size),
        ImageRecord(path=str(c2), rel_path=c2.name, size_bytes=c2.stat().st_size),
    ])
    labels.extend(["class_a", "class_b"])

    result = run_baseline(records, labels)
    assert result is not None
    assert result["images_used"] == 24


def test_baseline_majority_baseline_imbalance(tmp_path):
    # 18 class_a, 6 class_b -> majority baseline is 18/24 = 0.75
    records, labels = [], []
    for i in range(18):
        rec = _create_image(tmp_path / "a" / f"img_{i}.png", color="red")
        records.append(rec)
        labels.append("class_a")
    for i in range(6):
        rec = _create_image(tmp_path / "b" / f"img_{i}.png", color="blue")
        records.append(rec)
        labels.append("class_b")

    result = run_baseline(records, labels)
    assert result is not None
    assert result["majority_baseline"] == 0.75

