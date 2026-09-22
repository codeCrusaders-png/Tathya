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


def test_cli_invalid_options(tmp_path):
    dataset_dir = tmp_path / "data"
    _create_image(dataset_dir / "item1.png")

    # Invalid workers (< 1)
    assert main([str(dataset_dir), "--workers", "0", "--silent"]) == 2
    # Invalid pixel-sample (< 0)
    assert main([str(dataset_dir), "--pixel-sample", "-1", "--silent"]) == 2
    # Invalid dup-threshold (< 0)
    assert main([str(dataset_dir), "--dup-threshold", "-2", "--silent"]) == 2
    # Invalid near-dup-cap (< 1)
    assert main([str(dataset_dir), "--near-dup-cap", "0", "--silent"]) == 2


def test_cli_markdown_with_plots(tmp_path):
    dataset_dir = tmp_path / "dataset"
    _create_image(dataset_dir / "cat" / "cat1.jpg")
    _create_image(dataset_dir / "cat" / "cat2.jpg")
    _create_image(dataset_dir / "dog" / "dog1.png")

    out_dir = tmp_path / "report"
    code = main([
        str(dataset_dir),
        "-o", str(out_dir),
        "--formats", "md", "json",
        "--silent",
    ])
    assert code == 0

    md_content = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "## Visualizations" in md_content
    assert "![Class balance](assets/chart_01.png)" in md_content

    # Verify JSON is strictly RFC 8259 compliant (rejecting bare NaN/Infinity constants)
    def _reject_constant(constant):
        raise ValueError(f"Illegal RFC 8259 JSON constant: {constant}")

    raw_json = (out_dir / "report.json").read_text(encoding="utf-8")
    parsed = json.loads(raw_json, parse_constant=_reject_constant)
    assert parsed["meta"]["image_files"] == 3


def test_cli_new_flags(tmp_path):
    dataset_dir = tmp_path / "data"
    _create_image(dataset_dir / "img1.png", size=(50, 50))
    _create_image(dataset_dir / "img2.png", size=(50, 50))

    # Invalid thumb-size (< 1)
    assert main([str(dataset_dir), "--thumb-size", "0", "--silent"]) == 2

    # Run with --full-res and --rotation-invariant
    out_dir = tmp_path / "report_flags"
    code = main([
        str(dataset_dir),
        "-o", str(out_dir),
        "--full-res",
        "--rotation-invariant",
        "--formats", "json",
        "--silent",
    ])
    assert code == 0

    with open(out_dir / "report.json", encoding="utf-8") as f:
        data = json.load(f)

    assert data["duplicates"]["rotation_invariant"] is True
    assert data["pixels"]["resolution_mode"] == "full"




