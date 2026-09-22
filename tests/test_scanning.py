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


def test_discover_images_excludes_report_and_custom_dirs(tmp_path):
    _create_image(tmp_path / "valid" / "photo.jpg")
    _create_image(tmp_path / "tathya_report" / "assets" / "chart_01.png")
    _create_image(tmp_path / "custom_output" / "assets" / "chart_02.png")

    records, _ = discover_images(tmp_path, exclude_dirs=[tmp_path / "custom_output"])
    rel_paths = {rec.rel_path for rec in records}

    assert "valid/photo.jpg" in rel_paths
    assert not any("tathya_report" in p for p in rel_paths)
    assert not any("custom_output" in p for p in rel_paths)
    assert len(records) == 1


def test_detect_partial_split_dataset_layout(tmp_path):
    # Only train and test, no val
    _create_image(tmp_path / "train" / "cat" / "c1.jpg")
    _create_image(tmp_path / "test" / "cat" / "c2.jpg")

    records, _ = discover_images(tmp_path)
    layout = detect_layout(records, tmp_path)

    assert layout.kind == "split_class_folders"
    split_names = {s["name"] for s in layout.splits}
    assert split_names == {"train", "test"}


def test_detect_nested_with_labels_csv_basenames(tmp_path):
    # Images in subfolder, CSV uses simple file basenames
    _create_image(tmp_path / "images" / "pic1.png")
    _create_image(tmp_path / "images" / "pic2.png")

    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "category"])
        writer.writerow(["pic1.png", "lion"])
        writer.writerow(["pic2.png", "tiger"])

    records, _ = discover_images(tmp_path)
    layout = detect_layout(records, tmp_path)

    assert layout.class_of.get("images/pic1.png") == "lion"
    assert layout.class_of.get("images/pic2.png") == "tiger"
    assert "lion" in layout.classes
    assert "tiger" in layout.classes


def test_sidecar_audit(tmp_path):
    _create_image(tmp_path / "img1.png")
    _create_image(tmp_path / "img2.png")
    _create_image(tmp_path / "unreferenced.png")

    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerow(["img1.png", "class_a"])
        writer.writerow(["img2.png", ""])  # empty label
        writer.writerow(["img1.png", "class_a"])  # duplicate entry
        writer.writerow(["ghost.png", "class_b"])  # missing on disk

    records, _ = discover_images(tmp_path)
    layout = detect_layout(records, tmp_path)

    audit = layout.sidecar_audit
    assert audit is not None
    assert "ghost.png" in audit["missing_on_disk"]
    assert "img2.png" in audit["null_or_empty_labels"]
    assert "img1.png" in audit["duplicate_entries"]
    assert "unreferenced.png" in audit["unreferenced_on_disk"]


def test_group_leakage_detection(tmp_path):
    # Subject patient01 has images in both train and test splits -> leakage!
    # Subject patient02 has images only in train -> no leakage.
    _create_image(tmp_path / "train" / "patient01_img1.png")
    _create_image(tmp_path / "test" / "patient01_img2.png")
    _create_image(tmp_path / "train" / "patient02_img1.png")

    records, _ = discover_images(tmp_path)
    layout = detect_layout(records, tmp_path)

    assert len(layout.group_leakage) == 1
    leak = layout.group_leakage[0]
    assert leak["group"] == "patient01"
    assert set(leak["splits"]) == {"train", "test"}
    assert leak["count"] == 2




