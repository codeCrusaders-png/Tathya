"""Tests for tathya.scanning."""
import csv
from pathlib import Path

from PIL import Image

from tathya.scanning import detect_layout, discover_images


def _create_image(path: Path, size=(10, 10), color="red"):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=color)
    img.save(path)


def test_discover_empty_directory(tmp_path):
    records, missed = discover_images(tmp_path)
    assert len(records) == 0
    assert missed == 0

    layout = detect_layout(records, tmp_path)
    assert layout.kind == "empty"


def test_discover_images_filters_noise(tmp_path):
    # Valid image
    _create_image(tmp_path / "cats" / "cat1.jpg")
    _create_image(tmp_path / "dogs" / "dog1.png")

    # Noise directories that should be skipped
    _create_image(tmp_path / ".git" / "dummy.png")
    _create_image(tmp_path / "__pycache__" / "cached.jpg")
    _create_image(tmp_path / "node_modules" / "asset.png")

    # Non-image files
    (tmp_path / "README.md").write_text("dataset info")
    (tmp_path / "config.json").write_text("{}")

    records, missed = discover_images(tmp_path)
    rel_paths = {rec.rel_path for rec in records}

    assert "cats/cat1.jpg" in rel_paths
    assert "dogs/dog1.png" in rel_paths
    assert len(records) == 2
    assert missed == 2  # README.md and config.json


def test_detect_class_folders_layout(tmp_path):
    _create_image(tmp_path / "apple" / "img1.jpg")
    _create_image(tmp_path / "apple" / "img2.jpg")
    _create_image(tmp_path / "banana" / "img3.jpg")

    records, _ = discover_images(tmp_path)
    layout = detect_layout(records, tmp_path)

    assert layout.kind == "class_folders"
    assert "apple" in layout.classes
    assert "banana" in layout.classes
    assert layout.class_of["apple/img1.jpg"] == "apple"
    assert layout.class_of["banana/img3.jpg"] == "banana"


def test_detect_split_dataset_layout(tmp_path):
    _create_image(tmp_path / "train" / "apple" / "img1.jpg")
    _create_image(tmp_path / "train" / "banana" / "img2.jpg")
    _create_image(tmp_path / "val" / "apple" / "img3.jpg")
    _create_image(tmp_path / "test" / "banana" / "img4.jpg")

    records, _ = discover_images(tmp_path)
    layout = detect_layout(records, tmp_path)

    assert layout.kind == "split_class_folders"
    split_names = [s["name"] for s in layout.splits]
    assert "train" in split_names
    assert "val" in split_names
    assert "test" in split_names


def test_detect_flat_with_labels_csv(tmp_path):
    _create_image(tmp_path / "img1.jpg")
    _create_image(tmp_path / "img2.jpg")

    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerow(["img1.jpg", "cat"])
        writer.writerow(["img2.jpg", "dog"])

    records, _ = discover_images(tmp_path)
    layout = detect_layout(records, tmp_path)

    assert layout.kind == "label_file"
    assert layout.class_of["img1.jpg"] == "cat"
    assert layout.class_of["img2.jpg"] == "dog"

