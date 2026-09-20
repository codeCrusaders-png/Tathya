#!/usr/bin/env python3
"""Generate small synthetic (but realistic-looking) image datasets for demoing
tathya.  Run:

    python make_sample_dataset.py [--out sample_data] [--seed 7]

Creates three variants inside the output folder:

  sample_data/
    fruits_veggies/   folder-per-class classification dataset with
                      duplicates, corrupt files and outlier exposures
    split_dataset/    train/val/test splits (with a planted train->test
                      duplicate to demonstrate leakage detection)
    flat_mixed/       flat unlabelled-looking folder + labels.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------- #
# Synthetic image generation
# --------------------------------------------------------------------------- #

PALETTES = {
    "apple":  {"bg": (208, 222, 205), "fg": [(206, 54, 54), (152, 210, 120), (118, 178, 58)]},
    "banana": {"bg": (232, 232, 205), "fg": [(252, 220, 66), (255, 240, 150), (172, 132, 42)]},
    "carrot": {"bg": (232, 214, 202), "fg": [(248, 142, 42), (252, 182, 92), (164, 92, 32)]},
    "grape":  {"bg": (212, 198, 228), "fg": [(124, 62, 182), (184, 122, 224), (92, 62, 152)]},
    "lemon":  {"bg": (226, 226, 200), "fg": [(255, 240, 84), (254, 250, 182), (204, 182, 52)]},
    "kiwi":   {"bg": (198, 214, 198), "fg": [(106, 148, 60), (146, 186, 92), (120, 178, 72)]},
}


def synth_image(rng, width, height, palette, gray=False, exposure=1.0):
    """A plausible little photograph-ish image: gradient + soft colour blobs."""
    y, x = np.mgrid[:height, :width]
    background = np.asarray(palette["bg"], np.float32)
    img = np.zeros((height, width, 3), np.float32) + background
    img[..., 0] *= 0.75 + 0.5 * (y / max(height, 1))
    img[..., 1] *= 0.78 + 0.45 * (x / max(width, 1))
    img[..., 2] *= 0.85 + 0.30 * ((x + y) / (max(width + height, 1)))

    for _ in range(rng.randint(4, 8)):
        cx, cy = rng.uniform(0, width), rng.uniform(0, height)
        radius = rng.uniform(0.08, 0.26) * max(width, height)
        distance2 = ((x - cx) ** 2 + (y - cy) ** 2) / max(radius, 1.0) ** 2
        blob = np.exp(-distance2)[..., None]
        color = np.asarray(palette["fg"][int(rng.randint(0, len(palette["fg"])))], np.float32)
        img += blob * color * rng.uniform(0.45, 1.05)

    img = np.clip(img * exposure, 0, 255)
    if gray:
        gray_values = img.mean(axis=2, keepdims=True)
        img = np.concatenate([gray_values, gray_values, gray_values], axis=2)
    img += rng.normal(0.0, 7.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def save_image(rng, folder, name, palette, gray=False, exposure=1.0,
               ext="jpg", size=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if size is None:
        w = int(rng.randint(72, 272))
        h = int(rng.randint(72, 272))
    else:
        w, h = size
    array = synth_image(rng, w, h, palette, gray=gray, exposure=exposure)
    image = Image.fromarray(array)
    path = folder / f"{name}.{ext}"
    if ext.lower() == "jpg":
        image.save(path, quality=int(rng.randint(78, 95)))
    else:
        image.save(path)
    return path


def write_corrupt(base_dir, name):
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / f"{name}.png").write_bytes(b"\x89PNG\r\n\x1a\nCORRUPT\x00\xffNOTAPNG")
    (base_dir / f"{name}.jpg").write_bytes(b"\xff\xd8\xff not a real jpeg at all")


def near_duplicate_of(rng, source_path, target_path, shift_brightness=1.06):
    """Re-encode a copy with slightly altered brightness - a classic 'shadcned'
    near duplicate that shares no bytes with the original."""
    with Image.open(source_path) as im:
        array = np.asarray(im.convert("RGB"), np.float32) * shift_brightness
        target = np.clip(array, 0, 255).astype(np.uint8)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(target).save(target_path, quality=80)
    return target_path
# --------------------------------------------------------------------------- #
# Dataset builders
# --------------------------------------------------------------------------- #

FRUIT_CLASS_COUNTS = [("apple", 24), ("banana", 20), ("carrot", 16),
                      ("grape", 10), ("lemon", 6), ("kiwi", 2)]


def build_fruits_veggies(out_dir, rng):
    root = Path(out_dir) / "fruits_veggies"
    generated = []
    for class_name, count in FRUIT_CLASS_COUNTS:
        palette = PALETTES[class_name]
        for i in range(count):
            ext = "jpg" if rng.random() > 0.25 else rng.choice(["png", "bmp", "webp"])
            gray = (class_name == "banana" and i % 9 == 1)
            exposure = 1.0
            if i == 3 and class_name in ("apple", "carrot"):
                exposure = 0.16          # very dark outlier
            if i == 5 and class_name in ("grape", "lemon"):
                exposure = 3.30          # blown-out bright outlier
            size = None
            if i == 7 and class_name == "apple":
                size = (28, 24)          # tiny image (< 32 px)
            if i == 2 and class_name == "carrot":
                size = (420, 96)         # extreme panorama aspect ratio
            path = save_image(rng, root / class_name, f"{class_name}_{i:03d}",
                              palette, gray=gray, exposure=exposure, ext=ext, size=size)
            generated.append(path)

    # exact duplicates (identical bytes)
    a = next(root.glob("apple/apple_000.*"))
    b = a.with_name("apple_000_copy.png")
    b.write_bytes(a.read_bytes())
    generated.append(b)
    c = next(root.glob("grape/grape_001.*"))
    d = c.with_name("grape_001_twin.jpg")
    d.write_bytes(c.read_bytes())
    generated.append(d)

    # near-duplicates (re-encoded with slight brightness change)
    src = root / "apple" / "apple_001.jpg"
    near_duplicate_of(rng, src, root / "apple" / "apple_001_near.png")
    near_duplicate_of(rng, src, root / "kiwi" / "kiwi_001_from_apple.jpg")

    src2 = root / "lemon" / "lemon_000.jpg"
    near_duplicate_of(rng, src2, root / "lemon" / "lemon_000_near.png")

    # corrupt files
    write_corrupt(root, "broken_image")
    write_corrupt(root / "banana", "rotten")

    # noise files that should be ignored
    (root / "README.txt").write_text("synthetic demo dataset\n")
    (root / "Thumbs.db").write_bytes(b"windows thumbs file, not an image")
    return root


def build_split_dataset(out_dir, rng):
    root = Path(out_dir) / "split_dataset"
    for split, per_class in (("train", 10), ("val", 4), ("test", 5)):
        for class_name in ("apple", "banana", "carrot"):
            if split == "val" and class_name == "carrot":
                continue          # deliberately missing class -> alert
            palette = PALETTES[class_name]
            for i in range(per_class):
                save_image(rng, root / split / class_name,
                           f"{class_name}_{i:03d}", palette, ext="jpg")

    # Terrible-but-easy-way to demonstrate leakage: the exact same photo in
    # train and test, plus a near-duplicate pair across splits.
    train_src = root / "train" / "apple" / "apple_000.jpg"
    (root / "test" / "apple").mkdir(parents=True, exist_ok=True)
    (root / "test" / "apple" / "apple_dup_in_test.jpg").write_bytes(train_src.read_bytes())
    near_duplicate_of(rng, root / "train" / "banana" / "banana_000.jpg",
                      root / "test" / "banana" / "banana_shadow.jpg")
    return root


def build_flat_mixed(out_dir, rng):
    root = Path(out_dir) / "flat_mixed"
    labels = []
    for i in range(10):
        class_name = rng.choice(["flower", "leaf", "rock"])
        palette = PALETTES.get(class_name, PALETTES["apple"])
        if class_name == "flower":
            palette = PALETTES["apple"] if rng.random() < 0.5 else PALETTES["grape"]
        path = save_image(rng, root, f"IMG_{i:03d}", palette, ext="jpg")
        labels.append((path.name, class_name))
    with open(root / "labels.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "label"])
        writer.writerows(labels)
    return root


def main():
    parser = argparse.ArgumentParser(description="Generate demo datasets.")
    parser.add_argument("--out", default="sample_data")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    for name, builder in (
        ("fruits_veggies", build_fruits_veggies),
        ("split_dataset", build_split_dataset),
        ("flat_mixed", build_flat_mixed),
    ):
        root = builder(out, rng)
        count = sum(1 for _ in root.rglob("*")
                    if _.is_file() and _.suffix.lower() in
                    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"})
        print(f"  created {root}  ({count} image files)")

    print("Done. Run the detective with, e.g.:")
    print(f'  python dataset_detective.py "{out / "fruits_veggies"}"')
    print(f'  python dataset_detective.py "{out / "split_dataset"}"')
    print(f'  python dataset_detective.py "{out / "flat_mixed"}"')


if __name__ == "__main__":
    main()
