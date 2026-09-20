"""Duplicate detection: exact byte hashing + perceptual (dHash) near-duplicates."""
from __future__ import annotations

import numpy as np
from PIL import Image

from .utils import sha256_file

#: Default Hamming distance below which two dHashes are considered near-duplicates.
DEFAULT_DUP_THRESHOLD = 6


def compute_dhash(path, size=(9, 8)):
    """Compute a 64-bit difference hash (dHash) for an image file.

    The image is resized to 9x8 grayscale (aspect ratio ignored, like the
    classic dHash implementation) and each bit records whether the right
    neighbour pixel is brighter than the left one.
    """
    with Image.open(path) as im:
        gray = im.convert("L").resize(size, Image.Resampling.BILINEAR)
    arr = np.asarray(gray, dtype=np.int16)
    diff = (arr[:, 1:] > arr[:, :-1]).ravel()
    value = 0
    for bit in diff:
        value = (value << 1) | int(bit)
    return value


def find_exact_duplicates(records):
    """Group records with identical file bytes (sha256).

    Returns groups sorted from largest to smallest, each group being a list of
    :class:`~dataset_detective.scanning.ImageRecord`.
    """
    buckets = {}
    for rec in records:
        buckets.setdefault(sha256_file(rec.path), []).append(rec)
    groups = [recs for recs in buckets.values() if len(recs) > 1]
    groups.sort(key=lambda group: (-len(group), group[0].rel_path))
    return groups


class _DisjointSet:
    """Union-find for grouping near-duplicates in near-linear time."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _deterministic_sample(records, max_images):
    n = len(records)
    if n <= max_images:
        return list(records)
    step = n / max_images
    indexes = sorted({min(int(i * step), n - 1) for i in range(max_images)})
    return [records[i] for i in indexes]


def find_near_duplicates(records, threshold=DEFAULT_DUP_THRESHOLD, max_images=20000):
    """Group perceptually similar images using 64-bit dHash fingerprints.

    A fast approximate method: images are bucketed on four overlapping 16-bit
    slices of their hash and any pair sharing a slice whose Hamming distance is
    ``<= threshold`` is unioned.  For datasets larger than ``max_images`` a
    deterministic sample is analysed.

    Returns a list of groups (each a list of :class:`ImageRecord`), sorted from
    largest to smallest.
    """
    if len(records) > max_images:
        sampled = _deterministic_sample(records, max_images)
    else:
        sampled = list(records)

    hashes = [compute_dhash(rec.path) for rec in sampled]
    tables = [dict() for _ in range(4)]
    dsu = _DisjointSet(len(sampled))

    for i, h in enumerate(hashes):
        slices = (
            (h >> 48) & 0xFFFF,
            (h >> 32) & 0xFFFF,
            (h >> 16) & 0xFFFF,
            h & 0xFFFF,
        )
        for table, key in zip(tables, slices):
            for j in table.get(key, ()):
                if dsu.find(i) != dsu.find(j):
                    if (h ^ hashes[j]).bit_count() <= threshold:
                        dsu.union(i, j)
            table.setdefault(key, []).append(i)

    buckets = {}
    for i, rec in enumerate(sampled):
        buckets.setdefault(dsu.find(i), []).append(rec)
    groups = [recs for recs in buckets.values() if len(recs) > 1]
    groups.sort(key=lambda group: (-len(group), group[0].rel_path))
    return groups
