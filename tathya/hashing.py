"""Duplicate detection: exact byte hashing + perceptual (dHash) near-duplicates."""
from __future__ import annotations

import numpy as np
from PIL import Image

from .utils import sha256_file

#: Default Hamming distance below which two dHashes are considered near-duplicates.
DEFAULT_DUP_THRESHOLD = 6


def _dhash_from_image(im, size=(9, 8)):
    gray = im.convert("L").resize(size, Image.Resampling.BILINEAR)
    arr = np.asarray(gray, dtype=np.int16)
    diff = (arr[:, 1:] > arr[:, :-1]).ravel()
    value = 0
    for bit in diff:
        value = (value << 1) | int(bit)
    return value


def compute_dhash(path, size=(9, 8)):
    """Compute a 64-bit difference hash (dHash) for an image file.

    The image is resized to 9x8 grayscale (aspect ratio ignored, like the
    classic dHash implementation) and each bit records whether the right
    neighbour pixel is brighter than the left one.

    Returns None if the image cannot be opened or decoded.
    """
    try:
        with Image.open(path) as im:
            return _dhash_from_image(im, size)
    except Exception:
        return None


def compute_multirotation_dhash(path, size=(9, 8)):
    """Compute difference hashes across 4 orthogonal rotations (0, 90, 180, 270) and horizontal flip."""
    try:
        with Image.open(path) as im:
            im_0 = im.convert("L")
            im_90 = im_0.transpose(Image.Transpose.ROTATE_90)
            im_180 = im_0.transpose(Image.Transpose.ROTATE_180)
            im_270 = im_0.transpose(Image.Transpose.ROTATE_270)
            im_flip = im_0.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            return (
                _dhash_from_image(im_0, size),
                _dhash_from_image(im_90, size),
                _dhash_from_image(im_180, size),
                _dhash_from_image(im_270, size),
                _dhash_from_image(im_flip, size),
            )
    except Exception:
        return None



def find_exact_duplicates(records):
    """Group records with identical file bytes (sha256).

    Returns groups sorted from largest to smallest, each group being a list of
    :class:`~tathya.scanning.ImageRecord`.
    """
    buckets = {}
    for rec in records:
        try:
            digest = sha256_file(rec.path)
            buckets.setdefault(digest, []).append(rec)
        except OSError:
            continue
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


def find_near_duplicates(
    records,
    threshold=DEFAULT_DUP_THRESHOLD,
    max_images=20000,
    rotation_invariant=False,
):
    """Group perceptually similar images using 64-bit dHash fingerprints.

    When ``rotation_invariant=True``, detects duplicates across 90/180/270 degree
    rotations and horizontal flips. For datasets larger than ``max_images`` a
    deterministic sample is analysed.

    Returns a list of groups (each a list of :class:`ImageRecord`), sorted from
    largest to smallest.
    """
    if len(records) > max_images:
        sampled = _deterministic_sample(records, max_images)
    else:
        sampled = list(records)

    valid_sampled, hashes_list = [], []
    for rec in sampled:
        if rotation_invariant:
            rot_h = compute_multirotation_dhash(rec.path)
            if rot_h is not None:
                valid_sampled.append(rec)
                hashes_list.append(rot_h)
        else:
            h = compute_dhash(rec.path)
            if h is not None:
                valid_sampled.append(rec)
                hashes_list.append((h,))

    if not hashes_list:
        return []

    tables = [dict() for _ in range(4)]
    dsu = _DisjointSet(len(valid_sampled))

    for i, h_tuple in enumerate(hashes_list):
        for h in h_tuple:
            slices = (
                (h >> 48) & 0xFFFF,
                (h >> 32) & 0xFFFF,
                (h >> 16) & 0xFFFF,
                h & 0xFFFF,
            )
            for table, key in zip(tables, slices):
                for j in table.get(key, ()):
                    if dsu.find(i) != dsu.find(j):
                        min_dist = min(
                            (ha ^ hb).bit_count()
                            for ha in hashes_list[i]
                            for hb in hashes_list[j]
                        )
                        if min_dist <= threshold:
                            dsu.union(i, j)
                table.setdefault(key, []).append(i)

    buckets = {}
    for i, rec in enumerate(valid_sampled):
        buckets.setdefault(dsu.find(i), []).append(rec)
    groups = [recs for recs in buckets.values() if len(recs) > 1]
    groups.sort(key=lambda group: (-len(group), group[0].rel_path))
    return groups



def find_split_leakage(exact_groups, near_groups, splits):
    """Detect duplicate image clusters that span across different dataset splits.

    Unifies exact and near duplicate groups using connected components so
    images present in both are not double-counted.

    Parameters
    ----------
    exact_groups : list of list of ImageRecord
        Groups of exact byte-level duplicates.
    near_groups : list of list of ImageRecord
        Groups of perceptual near-duplicates.
    splits : list of dict or list of str
        Layout splits list, where each entry has a 'name' key or is a split string.

    Returns
    -------
    list of dict
        List of leakage summaries sorted by split name interaction:
        [{'between': 'test/train', 'groups': 1, 'images': 2}, ...]
    """
    if not splits:
        return []

    split_names_set = {
        s["name"] if isinstance(s, dict) and "name" in s else str(s)
        for s in splits
    }
    if not split_names_set:
        return []

    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    all_groups = list(exact_groups or []) + list(near_groups or [])
    if not all_groups:
        return []

    for group in all_groups:
        if not group:
            continue
        first = group[0].rel_path
        for rec in group[1:]:
            union(first, rec.rel_path)

    clusters = {}
    for group in all_groups:
        for rec in group:
            root = find(rec.rel_path)
            clusters.setdefault(root, set()).add(rec.rel_path)

    leakage_map = {}
    for cluster in clusters.values():
        splits_in_cluster = set()
        for rel_path in cluster:
            parts = rel_path.replace("\\", "/").split("/")
            for p in parts[:-1]:
                if p in split_names_set:
                    splits_in_cluster.add(p)
                    break

        if len(splits_in_cluster) > 1:
            between = "/".join(sorted(splits_in_cluster))
            if between not in leakage_map:
                leakage_map[between] = {"between": between, "groups": 0, "images": 0}
            leakage_map[between]["groups"] += 1
            leakage_map[between]["images"] += len(cluster)

    return sorted(leakage_map.values(), key=lambda item: item["between"])

