"""Tests for tathya.report (HTML and Markdown generation)."""
from tathya.report import render_html, render_markdown


def test_render_markdown_and_html_with_leakage():
    findings = {
        "meta": {
            "tool": "tathya",
            "tool_version": "1.0.0",
            "root": "/tmp/dataset",
            "dataset_name": "sample_ds",
            "generated_at": "2026-09-20 12:00 UTC",
            "elapsed": "1.2 s",
        },
        "layout": {
            "kind": "split_class_folders",
            "description": "Split structure",
            "splits": [
                {"name": "train", "count": 100, "n_classes": 2, "class_counts": {"a": 50, "b": 50}},
                {"name": "test", "count": 20, "n_classes": 2, "class_counts": {"a": 10, "b": 10}},
            ],
        },
        "classes": {
            "total": 2,
            "counts": {"a": 60, "b": 60},
        },
        "images": {
            "total": 120,
            "disk_bytes": 1024000,
            "corrupt": [],
        },
        "duplicates": {
            "exact_groups": 1,
            "exact_images": 2,
            "exact_bytes": 500,
            "exact": [["train/a/1.png", "test/a/1.png"]],
            "near_groups": 1,
            "near_images": 2,
            "near_bytes": 500,
            "near": [["train/a/1.png", "test/a/1.png"]],
            "leakage": [
                {"between": "test/train", "groups": 1, "images": 2},
            ],
        },
        "baseline": {
            "accuracy_mean": 0.85,
            "accuracy_std": 0.05,
            "cv_folds": 3,
            "images_used": 120,
            "classes_used": 2,
            "majority_baseline": 0.50,
            "feature_pipeline": "32x32 grayscale + PCA(64) + logistic regression",
            "note": "Quick sanity check",
        },
        "alerts": [
            {"level": "warn", "message": "Potential leakage found"},
        ],
        "recommendations": [
            "Re-split the dataset",
        ],
    }

    # Markdown test
    md = render_markdown(findings, asset_dir="assets")
    assert "# Tathya report — sample_ds" in md
    assert "- **Leakage risk:** 1 duplicate group(s) span `test/train` (2 images)." in md
    assert "- Majority class baseline: **50.0%**" in md
    assert "## Trainability baseline" in md

    # HTML test
    html = render_html(findings)
    assert "Tathya — sample_ds" in html
    assert "1 duplicate group(s) span <code>test/train</code> (2 images)" in html
    assert "majority baseline: <strong>50.0%</strong>" in html
    assert "Re-split the dataset" in html


def test_render_empty_minimal_findings():
    findings = {
        "meta": {"dataset_name": "empty_ds"},
        "layout": {"kind": "empty"},
        "images": {"total": 0, "disk_bytes": 0},
    }

    md = render_markdown(findings)
    assert "# Tathya report — empty_ds" in md

    html = render_html(findings)
    assert "empty_ds" in html
    assert "</html>" in html
