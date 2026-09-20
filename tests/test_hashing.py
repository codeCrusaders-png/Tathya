"""Tests for tathya.hashing."""
import shutil
from pathlib import Path

from PIL import Image

from tathya.hashing import compute_dhash, find_exact_duplicates, find_near_duplicates
from tathya.scanning import ImageRecord


def _create_image(path: Path, size=(30, 30), color="blue"):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=color)
    img.save(path)
    return ImageRecord(path=str(path), rel_path=path.name, size_bytes=path.stat().st_size)


def test_find_exact_duplicates(tmp_path):
    orig = tmp_path / "orig.png"
    copy1 = tmp_path / "copy1.png"
    diff = tmp_path / "different.png"

    rec_orig = _create_image(orig, color="green")
    shutil.copyfile(orig, copy1)
    rec_copy1 = ImageRecord(path=str(copy1), rel_path=copy1.name, size_bytes=copy1.stat().st_size)
    rec_diff = _create_image(diff, color="yellow")

    records = [rec_orig, rec_copy1, rec_diff]
    dup_groups = find_exact_duplicates(records)

    assert len(dup_groups) == 1
    assert len(dup_groups[0]) == 2
    paths = {r.rel_path for r in dup_groups[0]}
    assert paths == {"orig.png", "copy1.png"}


def test_dhash_and_near_duplicates(tmp_path):
    img1_path = tmp_path / "img1.png"
    img2_path = tmp_path / "img2.png"
    diff_path = tmp_path / "diff.png"

    # Slightly different sizes/encodings of identical content
    rec1 = _create_image(img1_path, size=(64, 64), color="red")
    rec2 = _create_image(img2_path, size=(32, 32), color="red")
    rec_diff = _create_image(diff_path, size=(32, 32), color="blue")

    h1 = compute_dhash(str(img1_path))
    h2 = compute_dhash(str(img2_path))
    assert isinstance(h1, int)
    assert isinstance(h2, int)

    # Near duplicates should group rec1 and rec2 together
    records = [rec1, rec2, rec_diff]
    groups = find_near_duplicates(records, threshold=6)
    assert len(groups) >= 1
    near_rel = {r.rel_path for r in groups[0]}
    assert "img1.png" in near_rel
    assert "img2.png" in near_rel

