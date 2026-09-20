"""Tests for tathya CLI and report pipeline."""
import json
from pathlib import Path

from PIL import Image

from tathya.cli import main


def _create_image(path: Path, size=(40, 40), color="green"):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=color)
    img.save(path)


def test_cli_empty_directory(tmp_path):
    out_dir = tmp_path / "report"
    code = main([str(tmp_path), "-o", str(out_dir), "--silent"])
    assert code == 0
    assert (out_dir / "report.html").exists()
    assert (out_dir / "report.json").exists()

    with open(out_dir / "report.json", encoding="utf-8") as f:
        data = json.load(f)
    assert data["meta"]["image_files"] == 0
    assert data["layout"]["kind"] == "empty"


def test_cli_sample_dataset(tmp_path):
    dataset_dir = tmp_path / "dataset"
    _create_image(dataset_dir / "cat" / "cat1.jpg", color="orange")
    _create_image(dataset_dir / "cat" / "cat2.jpg", color="orange")
    _create_image(dataset_dir / "dog" / "dog1.png", color="brown")
    _create_image(dataset_dir / "dog" / "dog2.png", color="brown")

    out_dir = tmp_path / "report"
    code = main([
        str(dataset_dir),
        "-o", str(out_dir),
        "--formats", "html", "json", "md",
        "--silent",
    ])
    assert code == 0

    assert (out_dir / "report.html").exists()
    assert (out_dir / "report.md").exists()
    assert (out_dir / "report.json").exists()
    assert (out_dir / "assets").exists()

    with open(out_dir / "report.json", encoding="utf-8") as f:
        data = json.load(f)

    assert data["meta"]["image_files"] == 4
    assert data["layout"]["kind"] == "class_folders"
    assert "cat" in data["classes"]["counts"]
    assert "dog" in data["classes"]["counts"]


def test_cli_invalid_directory():
    code = main(["/non/existent/path/that/does/not/exist", "--silent"])
    assert code == 2

