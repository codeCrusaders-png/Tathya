"""Matplotlib chart generation (rendered to base64 data-URIs for self-contained reports)."""
from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DPI = 110
C_BAR = "#4c78a8"
C_ACCENT = "#e07b39"
C_GRID = "#d8d8d8"


def _as_png_data_uri(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=C_GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def class_balance_chart(class_counts, max_bars=40):
    """Horizontal bar chart of images per class."""
    items = sorted(class_counts.items(), key=lambda kv: kv[1])
    if len(items) > max_bars:
        kept = items[-max_bars:]
        others = sum(v for _, v in items[:-max_bars])
        kept = [("other classes", others)] + kept
        truncated = True
    else:
        kept = items
        truncated = False
    names = [name for name, _ in kept]
    values = [value for _, value in kept]

    height = max(2.6, 0.34 * len(names))
    fig, ax = plt.subplots(figsize=(9.0, min(height, 14.0)))
    ax.barh(names, values, color=C_BAR)
    _style_axes(ax)
    ax.set_xlabel("number of images")
    total = sum(class_counts.values())
    if truncated:
        ax.set_title(f"Images per class (top {max_bars} of {len(items)} shown, total {total:,})")
    else:
        ax.set_title(f"Images per class (total {total:,})")
    fig.tight_layout()
    return _as_png_data_uri(fig)


def split_heatmap(split_rows, class_names):
    """Heatmap of class counts per split (log10 scale)."""
    split_names = [row["name"] for row in split_rows]
    matrix = np.zeros((len(split_names), len(class_names)), dtype=np.float64)
    for i, row in enumerate(split_rows):
        for j, cls in enumerate(class_names):
            matrix[i, j] = row["class_counts"].get(cls, 0.0)
    matrix = np.log10(matrix + 1.0)

    fig, ax = plt.subplots(figsize=(max(6.0, 0.30 * len(class_names)), 2.6))
    image = ax.imshow(matrix, aspect="auto", cmap="YlOrBr")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    ax.set_yticks(range(len(split_names)))
    ax.set_yticklabels(split_names, fontsize=9)
    ax.set_title("Class coverage per split (log10 counts)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _as_png_data_uri(fig)


def _histogram(values, title, xlabel, color, bins=50):
    values = [v for v in values if v is not None]
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.hist(values, bins=bins, color=color, edgecolor="none", alpha=0.9)
    _style_axes(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("images")
    ax.set_title(title)
    fig.tight_layout()
    return _as_png_data_uri(fig)


def brightness_chart(values):
    return _histogram(values, "Brightness distribution (0 = black, 1 = white)",
                      "mean grayscale brightness", "#9aa7c7")


def colorfulness_chart(values):
    return _histogram(values,
                      "Colorfulness distribution (Hasler & Süsstrunk metric)",
                      "colorfulness score", "#c98d5f")


def dimension_scatter(infos, max_points=6000):
    """Scatter of width vs height."""
    rng = np.random.RandomState(0)
    if len(infos) > max_points:
        sample = rng.choice(len(infos), max_points, replace=False)
    else:
        sample = np.arange(len(infos))
    widths = np.array([infos[i]["width"] for i in sample])
    heights = np.array([infos[i]["height"] for i in sample])

    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    ax.scatter(widths, heights, s=9, alpha=0.35, color=C_BAR, edgecolors="none")
    limit = max(widths.max(), heights.max())
    ax.plot([0, limit], [0, limit], color=C_ACCENT, lw=1.2, ls="--", label="square")
    ax.set_xlabel("width (px)")
    ax.set_ylabel("height (px)")
    ax.set_title(f"Image dimensions ({len(sample):,} sampled)")
    ax.set_xlim(0, limit * 1.02)
    ax.set_ylim(0, limit * 1.02)
    ax.legend(loc="lower right", frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    return _as_png_data_uri(fig)


def aspect_chart(values):
    clipped = [min(max(v, 0.10), 4.0) for v in values]
    return _histogram(clipped, "Aspect ratio distribution (w / h)",
                      "aspect ratio (clipped to 0.1–4.0)", "#7fae8f")


def channel_corr_heatmap(channel_names, corr_matrix):
    corr = np.asarray(corr_matrix)
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(channel_names)))
    ax.set_yticks(range(len(channel_names)))
    ax.set_xticklabels(channel_names)
    ax.set_yticklabels(channel_names)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(corr[i, j]) > 0.5 else "black", fontsize=9)
    ax.set_title("Mean channel correlation")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _as_png_data_uri(fig)


def format_chart(counts):
    items = sorted(counts.items(), key=lambda kv: -kv[1])[:8]
    names = [name for name, _ in items]
    values = [value for _, value in items]
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.bar(names, values, color=C_BAR)
    _style_axes(ax)
    ax.set_xlabel("format")
    ax.set_ylabel("images")
    ax.set_title("File format distribution")
    fig.tight_layout()
    return _as_png_data_uri(fig)


def duplicates_chart(group_sizes, kind):
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    if group_sizes:
        ax.bar(range(len(group_sizes)), group_sizes[:25], color=C_ACCENT)
        ax.set_xlabel(f"duplicate group (top {min(len(group_sizes), 25)})")
    ax.set_ylabel("images")
    ax.set_title(f"{kind} duplicate groups by size")
    _style_axes(ax)
    fig.tight_layout()
    return _as_png_data_uri(fig)


def build_plots(plot_inputs):
    """Return ``[{"title", "caption", "data_url"}, ...]`` for a report."""
    charts = []
    class_counts = plot_inputs.get("class_counts") or {}
    if class_counts:
        charts.append({
            "title": "Class balance",
            "caption": "Horizontal bar chart of the number of images per class.",
            "data_url": class_balance_chart(class_counts),
        })
    split_rows = plot_inputs.get("split_rows") or []
    class_names = plot_inputs.get("class_names") or []
    if split_rows and class_names:
        charts.append({
            "title": "Split coverage",
            "caption": "How many images of each class live in each train/val/test split "
                       "(log10 scale so empty cells are visible).",
            "data_url": split_heatmap(split_rows, class_names),
        })
    brightness = plot_inputs.get("brightness") or []
    if brightness:
        charts.append({
            "title": "Brightness",
            "caption": "Distribution of mean grayscale brightness across sampled images.",
            "data_url": brightness_chart(brightness),
        })
    colorfulness = plot_inputs.get("colorfulness") or []
    if colorfulness:
        charts.append({
            "title": "Colorfulness",
            "caption": "Distribution of the colorfulness metric across sampled images.",
            "data_url": colorfulness_chart(colorfulness),
        })
    infos = plot_inputs.get("infos") or []
    if infos:
        charts.append({
            "title": "Dimensions",
            "caption": "Width vs height for sampled images; the dashed line marks square "
                       "aspect ratio.",
            "data_url": dimension_scatter(infos),
        })
    aspects = plot_inputs.get("aspects") or []
    if aspects:
        charts.append({
            "title": "Aspect ratios",
            "caption": "Distribution of width/height ratios.",
            "data_url": aspect_chart(aspects),
        })
    corr = plot_inputs.get("channel_corr")
    if corr:
        charts.append({
            "title": "Channel correlation",
            "caption": "Mean RGB channel correlation averaged over the pixel sample.",
            "data_url": channel_corr_heatmap(["R", "G", "B"], corr),
        })
    formats = plot_inputs.get("formats") or {}
    if formats:
        charts.append({
            "title": "Formats",
            "caption": "Number of images per file format.",
            "data_url": format_chart(formats),
        })
    dup_exact = plot_inputs.get("dup_exact_sizes") or []
    dup_near = plot_inputs.get("dup_near_sizes") or []
    if dup_exact:
        charts.append({
            "title": "Exact duplicates",
            "caption": "Size of each exact-duplicate group (images sharing identical bytes).",
            "data_url": duplicates_chart(dup_exact, "exact"),
        })
    if dup_near:
        charts.append({
            "title": "Near-duplicates",
            "caption": "Size of each perceptual near-duplicate group (dHash).",
            "data_url": duplicates_chart(dup_near, "near"),
        })
    return charts
    return _histogram(values,
                      "Colorfulness distribution (Hasler & Süsstrunk metric)",
                      "colorfulness score", "#c98d5f")
