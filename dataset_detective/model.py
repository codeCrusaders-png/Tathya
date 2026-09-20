"""Optional 'can I train on this?' baseline using scikit-learn (if installed)."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from PIL import Image


def run_baseline(records, labels, seed=7, per_class=400, max_images=20000, thumb=32):
    """Train a tiny PCA + logistic-regression baseline and return CV metrics.

    Returns ``None`` (silently) when scikit-learn is not installed, the dataset
    is too small, or the number of classes is unreasonably large.
    """
    try:
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.pipeline import make_pipeline
    except Exception:
        return None

    per_class_records = defaultdict(list)
    for rec, label in zip(records, labels):
        per_class_records[label].append(rec)

    chosen, chosen_labels = [], []
    for label, recs in per_class_records.items():
        if len(recs) < 2:
            continue
        take = min(per_class, len(recs))
        chosen.extend(recs[:take])
        chosen_labels.extend([label] * take)

    class_count = len({lab for lab in chosen_labels})
    if len(chosen) < 20 or class_count < 2 or class_count > 200:
        return None

    features = []
    for rec in chosen:
        with Image.open(rec.path) as im:
            gray = im.convert("L").resize((thumb, thumb), Image.Resampling.BILINEAR)
        features.append(np.asarray(gray, dtype=np.float32).reshape(-1) / 255.0)

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(chosen_labels)
    if len(x) > max_images:
        rng = np.random.RandomState(seed)
        index = rng.choice(len(x), max_images, replace=False)
        x, y = x[index], y[index]

    pipeline = make_pipeline(
        PCA(n_components=min(64, x.shape[0] - 1)),
        LogisticRegression(max_iter=2000),
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    try:
        scores = cross_val_score(pipeline, x, y, cv=cv, scoring="accuracy", n_jobs=1)
    except Exception:
        return None

    return {
        "images_used": int(len(x)),
        "classes_used": class_count,
        "feature_pipeline": f"{thumb}x{thumb} grayscale + PCA(64) + logistic regression",
        "cv_folds": 3,
        "accuracy_mean": round(float(scores.mean()), 4),
        "accuracy_std": round(float(scores.std()), 4),
        "note": "Quick sanity baseline only — not comparable to a real trained model.",
    }
