"""Report rendering: Markdown and self-contained HTML."""
from __future__ import annotations

import html as _html

from .utils import human_size

HUMAN_LAYOUT = {
    "class_folders": "Classification folders (labels from folder names)",
    "split_class_folders": "Train/val/test split with class sub-folders",
    "split_flat": "Train/val/test split without class sub-folders",
    "flat": "Flat / unlabelled images",
    "label_file": "Flat images with a label sidecar file",
    "irregular": "Irregular / nested structure",
    "empty": "Empty",
}


def _esc(value):
    return _html.escape(str(value))


def _pct(fraction, digits=1):
    try:
        return f"{float(fraction) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_count(count):
    return f"{count:,}"


# --------------------------------------------------------------------------- #
# Markdown report
# --------------------------------------------------------------------------- #

def render_markdown(findings, asset_dir="assets"):
    meta = findings.get("meta", {})
    layout = findings.get("layout", {})
    classes = findings.get("classes", {})
    images = findings.get("images", {})

    lines = []
    label = meta.get("dataset_name", "dataset")
    lines.append(f"# Dataset Detective report — {label}")
    lines.append("")
    lines.append(f"> Generated {meta.get('generated_at', '')} by "
                 f"dataset_detective v{meta.get('tool_version', '?')}  ·  "
                 f"analysed in {meta.get('elapsed', '')}.")
    lines.append("")

    alerts = findings.get("alerts", [])
    if alerts:
        lines.append("## Alerts")
        lines.append("")
        for alert in alerts:
            lines.append(f"- **[{alert['level'].upper()}]** {alert['message']}")
        lines.append("")

    lines.append("## At a glance")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Dataset root | `{meta.get('root', '')}` |")
    lines.append(f"| Layout | {HUMAN_LAYOUT.get(layout.get('kind'), layout.get('kind', 'unknown'))} |")
    lines.append(f"| Total images | {_fmt_count(images.get('total', 0))} |")
    lines.append(f"| Labels | {layout.get('label_source', 'none') or 'none'} |")
    lines.append(f"| Classes | {_fmt_count(classes.get('total', 0))} |")
    lines.append(f"| Disk used by images | {human_size(images.get('disk_bytes', 0))} |")
    lines.append("")

    return _build_markdown_details(lines, findings)


def _build_markdown_details(lines, findings):
    layout = findings.get("layout", {})
    classes = findings.get("classes", {})
    images = findings.get("images", {})
    geometry = findings.get("geometry", {})
    pixels = findings.get("pixels", {})
    duplicates = findings.get("duplicates", {})
    baseline = findings.get("baseline")

    # --- structure & labels ------------------------------------------------
    lines.append("## Structure & labels")
    lines.append("")
    lines.append(f"**{HUMAN_LAYOUT.get(layout.get('kind'), layout.get('kind', 'unknown'))}** — "
                 f"{layout.get('description', '')}")
    if layout.get("label_file"):
        lines.append(f"- Label file: `{layout['label_file']}`")
    lines.append(f"- Label source: `{layout.get('label_source', 'none')}`")
    for note in layout.get("notes", []):
        lines.append(f"- Note: {note}")
    lines.append("")

    splits = layout.get("splits") or []
    if splits:
        lines.append("### Splits")
        lines.append("")
        lines.append("| Split | Images | Classes |")
        lines.append("|---|---|---|")
        for row in splits:
            lines.append(f"| {row['name']} | {_fmt_count(row['count'])} | "
                         f"{_fmt_count(row['n_classes'])} |")
        lines.append("")

    # --- class distribution ------------------------------------------------
    if classes.get("counts"):
        lines.append("## Class distribution")
        lines.append("")
        lines.append("| Class | Images | Share |")
        lines.append("|---|---|---|")
        total = classes.get("total", sum(classes.get("counts", {}).values()))
        for name, count in sorted(classes["counts"].items(), key=lambda kv: -kv[1]):
            share = _pct(count / total) if total else "—"
            lines.append(f"| `{name}` | {_fmt_count(count)} | {share} |")
        lines.append("")
        balance = classes.get("balance", {})
        if balance:
            lines.append(f"- Imbalance ratio (max/min): **{balance.get('imbalance_ratio', 'n/a')}x** "
                         f"· Gini coefficient: **{balance.get('gini', 'n/a')}**")
            if balance.get("small_classes"):
                lines.append(f"- Small classes (< 1% of data): "
                             f"{', '.join(balance['small_classes'])}")
        lines.append("")
# --- image health ------------------------------------------------------
    corrupt = images.get("corrupt", [])
    if corrupt:
        lines.append("## Image health")
        lines.append("")
        lines.append(f"{len(corrupt)} images could **not** be opened or are corrupt:")
        for item in corrupt[:15]:
            lines.append(f"- `{item['rel_path']}` — {item['error'][:160]}")
        if len(corrupt) > 15:
            lines.append(f"- _… and {len(corrupt) - 15} more_")
        lines.append("")

    # --- geometry ----------------------------------------------------------
    if geometry.get("width"):
        lines.append("## Image geometry")
        lines.append("")
        lines.append("| Dimension | Min | Median | Mean (σ) | Max |")
        lines.append("|---|---|---|---|---|")
        for key, label in (("width", "Width"), ("height", "Height"), ("aspect", "Aspect")):
            block = geometry.get(key) or {}
            if not block:
                continue
            lines.append(f"| {label} | {block['min']} | {block['median']} | "
                         f"{block['mean']} ({block['std']}) | {block['max']} |")
        if geometry.get("recommended"):
            lines.append("")
            lines.append(f"**Suggested training input size:** `{geometry['recommended']}`")
        lines.append("")

    # --- pixel statistics --------------------------------------------------
    if pixels.get("n"):
        lines.append("## Pixel statistics")
        lines.append("")
        lines.append(f"Sampled {pixels['n']} images.")
        lines.append("")
        remap = {"R": "Red", "G": "Green", "B": "Blue"}
        lines.append("| Channel | Mean (0–1) | Std (0–1) |")
        lines.append("|---|---|---|")
        for name, mean, std in zip(pixels["channel_names"],
                                   pixels["channel_means"],
                                   pixels["channel_stds"]):
            lines.append(f"| {remap.get(name, name)} | {mean:.3f} | {std:.3f} |")
        lines.append("")
        lines.append(f"- Brightness: **{pixels['brightness_mean']:.2f}** "
                     f"± {pixels['brightness_std']:.3f}")
        lines.append(f"- Colorfulness: **{pixels['colorfulness_mean']:.1f}** "
                     f"± {pixels['colorfulness_std']:.1f}")
        lines.append(f"- Dark-exposed fraction: {_pct(pixels['dark_frac_mean'])} · "
                     f"Bright-exposed fraction: {_pct(pixels['bright_frac_mean'])}")
        lines.append(f"- Low-contrast images: {_pct(pixels['low_contrast_frac'])}")
        lines.append("")
# --- duplicates --------------------------------------------------------
    if duplicates.get("exact_groups") or duplicates.get("near_groups"):
        lines.append("## Duplicate analysis")
        lines.append("")
        if duplicates.get("exact_groups"):
            lines.append(f"- **Exact duplicates:** {duplicates['exact_groups']} groups, "
                         f"{duplicates['exact_images']} images, "
                         f"~{human_size(duplicates.get('exact_bytes', 0))} waste")
            for group in duplicates.get("exact", [])[:8]:
                lines.append(f"  - `{group[0]}` + {len(group) - 1} more")
        if duplicates.get("near_groups"):
            note = " (sampled)" if duplicates.get("near_sampled") else ""
            lines.append(f"- **Near-duplicates (dHash ≤ {duplicates.get('threshold', 6)}):** "
                         f"{duplicates['near_groups']} groups, {duplicates['near_images']} images, "
                         f"~{human_size(duplicates.get('near_bytes', 0))} waste{note}")
            for group in duplicates.get("near", [])[:8]:
                lines.append(f"  - `{group[0]}` + {len(group) - 1} more")
        for leak in duplicates.get("leakage", []):
            lines.append(f"- **Leakage risk:** {leak['groups']} near-duplicate group(s) span "
                         f"`{leak['between']}` ({leak['images']} images).")
        lines.append("")

    # --- baseline ----------------------------------------------------------
    if baseline:
        lines.append("## Trainability baseline")
        lines.append("")
        lines.append(f"- CV accuracy: **{baseline['accuracy_mean'] * 100:.1f}% ± "
                     f"{baseline['accuracy_std'] * 100:.1f}%** "
                     f"(3-fold, {baseline['images_used']} images, "
                     f"{baseline['classes_used']} classes)")
        lines.append(f"- Pipeline: {baseline['feature_pipeline']}")
        lines.append(f"- {baseline.get('note', '')}")
        lines.append("")

    recs = findings.get("recommendations", [])
    if recs:
        lines.append("## Recommendations")
        lines.append("")
        for i, rec in enumerate(recs, start=1):
            lines.append(f"{i}. {rec}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("_Generated by dataset_detective. Near-duplicate detection uses a perceptual "
                 "dHash and is approximate; pixel statistics are computed on down-scaled thumbnails._")
    lines.append("")
    return "\n".join(lines)
# --------------------------------------------------------------------------- #
# HTML report (self-contained: CSS inline, charts embedded as base64 data URIs)
# --------------------------------------------------------------------------- #

_CSS = """
:root { --ink:#1f2937; --muted:#6b7280; --accent:#2563eb; --ok:#16a34a;
        --warn:#d97706; --err:#dc2626; --card:#ffffff; --bg:#f3f4f6; --line:#e5e7eb; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
.container { max-width:1080px; margin:0 auto; padding:24px 20px 48px; }
header.hero { background:linear-gradient(120deg,#0b1f3a,#1d3a6b); color:#fff;
              border-radius:14px; padding:22px 26px; margin-bottom:22px; }
header.hero h1 { margin:0 0 6px; font-size:22px; letter-spacing:.2px; }
header.hero p { margin:0; opacity:.85; font-size:13px; }
h2 { font-size:17px; margin:28px 0 10px; }
h3 { font-size:14px; margin:14px 0 6px; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px;
        padding:12px 14px; }
.card .k { font-size:11.5px; color:var(--muted); text-transform:uppercase; letter-spacing:.4px; }
.card .v { font-size:20px; font-weight:650; margin-top:4px; line-height:1.05; }
.card .s { font-size:11.5px; color:var(--muted); }
.alert { border:1px solid var(--line); border-left:5px solid var(--warn);
         border-radius:8px; padding:10px 14px; margin:8px 0; background:var(--card); }
.alert.error { border-left-color:var(--err); }
.alert.warn { border-left-color:var(--warn); }
.alert.info { border-left-color:var(--accent); }
.alert.ok { border-left-color:var(--ok); }
.alert .tag { font-weight:700; font-size:11px; }
table { border-collapse:collapse; width:100%; font-size:12.5px; margin:8px 0 4px; }
th, td { border:1px solid var(--line); padding:6px 9px; text-align:left; }
th { background:#eef1f6; font-weight:600; }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
tr:nth-child(even) td { background:#fafbfc; }
details { margin:6px 0 2px; }
summary { cursor:pointer; font-size:12.5px; color:var(--muted); }
.chart { background:var(--card); border:1px solid var(--line); border-radius:10px;
         padding:12px; margin:10px 0; }
.chart img { width:100%; height:auto; display:block; }
.chart .cap { font-size:12px; color:var(--muted); margin-top:8px; }
code { font-family:ui-monospace,SFMono-Regular,Consolas,Menlo,monospace; font-size:12px;
       background:#f2f3f5; border:1px solid var(--line); border-radius:5px; padding:1px 5px; }
.recs { margin:6px 0; }
footer.foot { color:var(--muted); font-size:12px; margin-top:34px; border-top:1px solid var(--line);
              padding:10px 0; }
@media print { .container { max-width:none; } }
"""
def render_html(findings):
    lines = []
    meta = findings.get("meta", {})
    layout = findings.get("layout", {})
    classes = findings.get("classes", {})
    images = findings.get("images", {})
    geometry = findings.get("geometry", {})
    pixels = findings.get("pixels", {})
    duplicates = findings.get("duplicates", {})
    baseline = findings.get("baseline")
    plots = findings.get("plots", [])
    alerts = findings.get("alerts", [])
    recs = findings.get("recommendations", [])

    lines.append("<!DOCTYPE html>")
    lines.append('<html lang="en"><head><meta charset="utf-8">')
    lines.append(f"<title>Dataset Detective — {_esc(meta.get('dataset_name', 'report'))}</title>")
    lines.append(f"<style>{_CSS}</style></head><body>")
    lines.append('<div class="container">')

    # Hero ------------------------------------------------------------------
    lines.append('<header class="hero">')
    lines.append(f"<h1>Dataset Detective — {_esc(meta.get('dataset_name', 'report'))}</h1>")
    lines.append(f"<p>Full report generated {_esc(meta.get('generated_at', ''))} by "
                 f"dataset_detective v{_esc(meta.get('tool_version', '?'))} · "
                 f"analysed in {_esc(meta.get('elapsed', ''))} · "
                 f"root <code>{_esc(meta.get('root', ''))}</code></p>")
    lines.append("</header>")

    # Alerts ----------------------------------------------------------------
    lines.append("<h2>⚑ Alerts &amp; warnings</h2>")
    if alerts:
        for alert in alerts:
            lines.append(f'<div class="alert {_esc(alert["level"])}">'
                         f'<span class="tag">[{_esc(alert["level"].upper())}]</span> '
                         f'{_esc(alert["message"])}</div>')
    else:
        lines.append('<div class="alert ok"><span class="tag">[OK]</span> '
                     'No blocking issues found.</div>')

    # At a glance cards -----------------------------------------------------
    cards = [
        ("Total images", _fmt_count(images.get("total", 0))),
        ("Classes", _fmt_count(classes.get("total", 0))
         if classes.get("total") else "n/a (unlabelled)"),
        ("Disk used", human_size(images.get("disk_bytes", 0))),
        ("Duplicates", _fmt_count(duplicates.get("exact_images", 0) +
                                  duplicates.get("near_images", 0))),
        ("Corrupt", _fmt_count(len(images.get("corrupt", [])))),
    ]
    if baseline:
        cards.append(("Baseline CV",
                      f"{baseline['accuracy_mean'] * 100:.1f}% ± "
                      f"{baseline['accuracy_std'] * 100:.1f}%"))
    lines.append('<h2>At a glance</h2><div class="cards">')
    for label, value in cards:
        lines.append(f'<div class="card"><div class="k">{_esc(label)}</div>'
                     f'<div class="v">{_esc(value)}</div></div>')
    lines.append("</div>")

    # Structure -------------------------------------------------------------
    lines.append("<h2>Structure &amp; labels</h2>")
    lines.append(f"<p><strong>{_esc(HUMAN_LAYOUT.get(layout.get('kind'), layout.get('kind', 'unknown')))}</strong> — "
                 f"{_esc(layout.get('description', ''))}</p>")
    lines.append(f"<p>Label source: <code>{_esc(layout.get('label_source', 'none'))}</code>")
    if layout.get("label_file"):
        lines.append(f" · label file: <code>{_esc(layout['label_file'])}</code>")
    lines.append("</p>")
    for note in layout.get("notes", []):
        lines.append(f'<div class="alert info"><span class="tag">[NOTE]</span> {_esc(note)}</div>')

    splits = layout.get("splits") or []
    if splits:
        lines.append("<h3>Splits</h3><table><tr><th>Split</th>"
                     "<th class='num'>Images</th><th class='num'>Classes</th></tr>")
        for row in splits:
            lines.append(f"<tr><td>{_esc(row['name'])}</td>"
                         f"<td class='num'>{_fmt_count(row['count'])}</td>"
                         f"<td class='num'>{_fmt_count(row['n_classes'])}</td></tr>")
        lines.append("</table>")
# Class distribution -----------------------------------------------------
    counts = classes.get("counts") or {}
    if counts:
        lines.append("<h2>Class distribution</h2>")
        balance = classes.get("balance", {})
        if balance:
            ratio = balance.get("imbalance_ratio", "n/a")
            lines.append(f"<p>Imbalance ratio (max/min): <strong>{_esc(ratio)}×</strong> · "
                         f"Gini coefficient: <strong>{_esc(balance.get('gini', 'n/a'))}</strong></p>")
            if balance.get("small_classes"):
                small = ", ".join(_esc(name) for name in balance["small_classes"])
                lines.append(f"<p>Small classes (&lt; 1% of data or &lt; 10 images): {small}</p>")
        total = classes.get("total", sum(counts.values()))
        lines.append("<details><summary>Show per-class table</summary><table>"
                     "<tr><th>Class</th><th class='num'>Images</th>"
                     "<th class='num'>Share</th></tr>")
        for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            share = _pct(count / total) if total else "—"
            lines.append(f"<tr><td><code>{_esc(name)}</code></td>"
                         f"<td class='num'>{_fmt_count(count)}</td>"
                         f"<td class='num'>{share}</td></tr>")
        lines.append("</table></details>")

    # Image health ---------------------------------------------------------
    corrupt = images.get("corrupt", [])
    formats = findings.get("formats", {})
    modes = findings.get("modes", {})
    if corrupt or formats or modes:
        lines.append("<h2>Image health</h2>")
        if corrupt:
            lines.append(f'<div class="alert error"><span class="tag">[ERROR]</span> '
                         f"{len(corrupt)} image(s) could not be opened or are corrupt.</div>")
            lines.append("<details><summary>Show corrupt files</summary><table>"
                         "<tr><th>Path</th><th>Error</th></tr>")
            for item in corrupt[:30]:
                lines.append(f"<tr><td><code>{_esc(item['rel_path'])}</code></td>"
                             f"<td>{_esc(item['error'][:200])}</td></tr>")
            if len(corrupt) > 30:
                lines.append(f"<tr><td colspan='2'>… and {len(corrupt) - 30} more</td></tr>")
            lines.append("</table></details>")
        if formats:
            parts = ", ".join(f"{_esc(k)}: <strong>{v}</strong>"
                              for k, v in sorted(formats.items(), key=lambda kv: -kv[1]))
            lines.append(f"<p>Formats: {parts}</p>")
        if modes:
            parts = ", ".join(f"{_esc(k)}: <strong>{v}</strong>"
                              for k, v in sorted(modes.items(), key=lambda kv: -kv[1]))
            lines.append(f"<p>Colour modes: {parts}</p>")

    # Geometry -------------------------------------------------------------
    if geometry.get("width"):
        lines.append("<h2>Image geometry</h2><table><tr><th>Dimension</th>"
                     "<th class='num'>Min</th><th class='num'>P10</th>"
                     "<th class='num'>Median</th><th class='num'>Mean (σ)</th>"
                     "<th class='num'>P90</th><th class='num'>Max</th></tr>")
        for key, label in (("width", "Width (px)"), ("height", "Height (px)"),
                           ("aspect", "Aspect ratio")):
            block = geometry.get(key) or {}
            if not block:
                continue
            lines.append(f"<tr><td>{label}</td><td class='num'>{block['min']}</td>"
                         f"<td class='num'>{block['p10']}</td>"
                         f"<td class='num'>{block['median']}</td>"
                         f"<td class='num'>{block['mean']} (±{block['std']})</td>"
                         f"<td class='num'>{block['p90']}</td>"
                         f"<td class='num'>{block['max']}</td></tr>")
        lines.append("</table>")
        if geometry.get("recommended"):
            lines.append(f"<p>Suggested training input size: "
                         f"<code>{_esc(geometry['recommended'])}</code></p>")
# Pixel statistics -----------------------------------------------------
    if pixels.get("n"):
        lines.append("<h2>Pixel statistics</h2>")
        remap = {"R": "Red", "G": "Green", "B": "Blue"}
        lines.append("<table><tr><th>Channel</th><th class='num'>Mean (0–1)</th>"
                     "<th class='num'>Std (0–1)</th></tr>")
        for name, mean, std in zip(pixels["channel_names"],
                                   pixels["channel_means"],
                                   pixels["channel_stds"]):
            lines.append(f"<tr><td>{_esc(remap.get(name, name))}</td>"
                         f"<td class='num'>{mean:.3f}</td>"
                         f"<td class='num'>{std:.3f}</td></tr>")
        lines.append("</table>")
        lines.append("<ul>")
        lines.append(f"<li>Sampled <strong>{pixels['n']}</strong> images</li>")
        lines.append(f"<li>Brightness: <strong>{pixels['brightness_mean']:.2f}</strong> "
                     f"± {pixels['brightness_std']:.3f}</li>")
        lines.append(f"<li>Colorfulness: <strong>{pixels['colorfulness_mean']:.1f}</strong> "
                     f"± {pixels['colorfulness_std']:.1f}</li>")
        lines.append(f"<li>Dark-exposed: <strong>{_pct(pixels['dark_frac_mean'])}</strong> · "
                     f"bright-exposed: <strong>{_pct(pixels['bright_frac_mean'])}</strong></li>")
        lines.append(f"<li>Low-contrast images: "
                     f"<strong>{_pct(pixels['low_contrast_frac'])}</strong></li>")
        lines.append("</ul>")

    # Duplicates -------------------------------------------------------------
    if duplicates.get("exact_groups") or duplicates.get("near_groups"):
        lines.append("<h2>Duplicate analysis</h2>")
        if duplicates.get("exact_groups"):
            lines.append(f'<div class="alert warn"><span class="tag">[WARN]</span> '
                         f"<strong>{duplicates['exact_groups']}</strong> exact-duplicate group(s) "
                         f"covering <strong>{duplicates['exact_images']}</strong> images "
                         f"(~{human_size(duplicates.get('exact_bytes', 0))} wasted).</div>")
            lines.append("<details><summary>Show exact duplicate groups</summary>")
            for group in duplicates.get("exact", []):
                shown = ", ".join(f"<code>{_esc(p)}</code>" for p in group[:6])
                more = f" <em>+{len(group) - 6} more</em>" if len(group) > 6 else ""
                lines.append(f"<div class='recs'>{shown}{more}</div>")
            lines.append("</details>")
        if duplicates.get("near_groups"):
            sample_note = " (analysed on a sample)" if duplicates.get("near_sampled") else ""
            lines.append(f'<div class="alert warn"><span class="tag">[WARN]</span> '
                         f"<strong>{duplicates['near_groups']}</strong> near-duplicate group(s) "
                         f"(dHash ≤ {duplicates.get('threshold', 6)}) covering "
                         f"<strong>{duplicates['near_images']}</strong> images "
                         f"(~{human_size(duplicates.get('near_bytes', 0))} wasted){sample_note}.</div>")
            lines.append("<details><summary>Show near-duplicate groups</summary>")
            for group in duplicates.get("near", []):
                shown = ", ".join(f"<code>{_esc(p)}</code>" for p in group[:6])
                more = f" <em>+{len(group) - 6} more</em>" if len(group) > 6 else ""
                lines.append(f"<div>{shown}{more}</div>")
            lines.append("</details>")
        for leak in duplicates.get("leakage", []):
            lines.append(f'<div class="alert error"><span class="tag">[LEAK]</span> '
                         f"{leak['groups']} near-duplicate group(s) span "
                         f"<code>{_esc(leak['between'])}</code> "
                         f"({leak['images']} images) — validation could be optimistic.</div>")

    # Baseline ----------------------------------------------------------------
    if baseline:
        lines.append("<h2>Trainability baseline</h2>")
        lines.append(f"<p>Cross-validation accuracy: "
                     f"<strong>{baseline['accuracy_mean'] * 100:.1f}%</strong>"
                     f" ± {baseline['accuracy_std'] * 100:.1f}% "
                     f"(3-fold, {baseline['images_used']} images, "
                     f"{baseline['classes_used']} classes) on "
                     f"{_esc(baseline['feature_pipeline'])}.</p>")
        lines.append(f"<p>{_esc(baseline.get('note', ''))}</p>")

    # Charts ------------------------------------------------------------------
    if plots:
        lines.append("<h2>Charts</h2>")
        for plot in plots:
            lines.append(f'<div class="chart"><img src="{plot["data_url"]}" '
                         f'alt="{_esc(plot["title"])}">'
                         f'<div class="cap"><strong>{_esc(plot["title"])}.</strong> '
                         f'{_esc(plot["caption"])}</div></div>')

    # Recommendations ------------------------------------------------------------
    if recs:
        lines.append("<h2>Recommendations</h2><ol>")
        for rec in recs:
            lines.append(f"<li>{_esc(rec)}</li>")
        lines.append("</ol>")

    lines.append("<footer class='foot'>Generated by <strong>dataset_detective</strong> from "
                 "folder-wise structure inspection, sha256 / dHash duplicate checks and thumbnail "
                 "pixel statistics. Near-duplicate groups are approximate; verify edge cases "
                 "before deleting anything.</footer>")
    lines.append("</div><!-- /container -->")
    lines.append("</body></html>")
    return "\n".join(lines)
