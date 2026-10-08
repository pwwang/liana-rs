#!/usr/bin/env python3
"""Render the manuscript's figure set from `bench/results.json`.

    python3 paper/make_figures.py [bench/results.json] [paper/figures/]

F1  memory law   — peak RSS vs `n_perms × n_lrs`, three arms, log-log, with the
                   recorded law and the 50k re-fit, and the consensus-resource
                   run (4,620 LRs) annotated.
F2  wall         — grouped bars, wall clock, 10k/50k/100k at 1,000 permutations.
F3  peak RSS     — the same matrix as F2, peak resident memory instead of wall.
F4  threads      — wall vs thread count, three arms, with the patched arm's
                   box-state spread shown rather than averaged away.

Every plotted number is read from `bench/results.json` at render time — the same
labels `paper/make_numbers.py` freezes — so a figure cannot drift from the text.
Each panel is written standalone (for the companion HTML) and the four panels
are composed as one 2×2 figure (a+b+c+d) for a single-column App Note.  Panel
letters are bracketed and sit above the axes frame; bar value labels sit centred
on their bar and clear of its top — `check_placement` measures both on every
save, so the placement is asserted, not eyeballed.

Style (project convention): Arial first with DejaVu Sans last in the fallback
chain, no text below 7 pt, 180 mm wide, 300 dpi, PDF + PNG, Okabe-Ito palette.
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from matplotlib.transforms import Bbox

REPO = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO / "bench/results.json"
OUT = REPO / "paper/figures"

# Okabe-Ito (colour-blind safe). One colour per arm, in every panel.
OKABE_ITO = {
    "rust": "#0072B2",       # blue
    "release": "#D55E00",    # vermillion
    "patched": "#009E73",    # bluish green
    "grid": "#BBBBBB",
    "law": "#000000",
    "faint": "0.35",
}
LABELS = {"rust": "liana-rs", "release": "LIANA+ 2.0.0 (release)",
          "patched": "LIANA+ 2.0.0 + chunked patch"}
ARMS = ("rust", "release", "patched")

WIDTH_MM = 180.0
MM_PER_IN = 25.4
DPI = 300

# W12: the panel letters sit in the margin above the axes' top-left corner — bracketed,
# outside the plotting area — and the bar value labels sit wholly above their bar tops.
# Both gaps are in points; `check_placement` measures the rendered result on every save.
# W14: the author asked for a visibly clear gap between the tag and the plotting area, so
# the letter pad is 6.0 pt (was 1.5); `check_placement` now asserts the measured gap equals
# this value, not merely a non-negative one. The bar-label gap is unchanged.
LETTER_PAD_PT = 6.0
BAR_LABEL_PAD_PT = 1.5

# The Python arms' `n_jobs=4` rows are the frozen suite's; the rest of their
# thread points are recorded reference logs from a different session.
SUITE_JOBS = 4


def mm(value: float) -> float:
    """A millimetre figure dimension, snapped to a whole pixel at `DPI`.

    A 180 mm figure is 2125.98 px at 300 dpi; asking for the unrounded size gets
    a 2125 px canvas (matplotlib truncates), so the rounded canvas is the one
    that lands closest to 180.0 mm.
    """
    return round(value / MM_PER_IN * DPI) / DPI


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Arimo", "Nimbus Sans",
                            "Liberation Sans", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8,
        "axes.linewidth": 0.6,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": None,          # keep the figure exactly 180 mm wide
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# --------------------------------------------------------------------------
# the manifest, addressed the way make_numbers.py addresses it

def need(doc: dict, label: str, field: str = "rss_kb"):
    for row in doc["measurements"]:
        if row["label"] == label:
            if row.get(field) is None:
                raise SystemExit(f"make_figures: MISSING NUMBER — bench/results.json "
                                 f"has no `{label}.{field}`")
            return row[field]
    raise SystemExit(f"make_figures: MISSING NUMBER — bench/results.json has no "
                     f"measurement `{label}`")


def gb(rss_kb: float) -> float:
    """GB = 1000 MB over the manifest's own MB (kB / 1024) — the convention the
    manuscript's text uses, so 23,746.9 MB reads 23.7 GB and 11,041.6 MB 11.0 GB."""
    return rss_kb / 1024 / 1000


def law_gb(intercept_mb: float, slope_kb: float, perms_lrs: float) -> float:
    return (intercept_mb * 1024 + slope_kb * perms_lrs) / 1024 / 1000


def sig3(value: float) -> str:
    return f"{value:.3g}"


def panel_letter(ax, letter: str) -> None:
    """The panel's bracketed letter — `(a)` — bold, in the margin above the axes' top-left
    corner (W12: the author asked for bracketed labels outside the plotting area).

    `xytext` is an offset in points from the axes' top-left corner (`LETTER_PAD_PT`, 6 pt
    since W14), so the text bbox sits wholly above the axes frame; `check_placement`
    measures that on every save and fails the render if a letter drifts back inside the
    frame, off the recorded gap, or is clipped at the figure edge.
    """
    ax.annotate(f"({letter})", xy=(0.0, 1.0), xycoords="axes fraction",
                xytext=(0.0, LETTER_PAD_PT), textcoords="offset points",
                ha="left", va="bottom", fontsize=9, fontweight="bold",
                gid=f"letter:{letter}")


def footer(ax, text: str) -> None:
    ax.text(0.012, 0.035, text, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7, color=OKABE_ITO["faint"])


# --------------------------------------------------------------------------
# panels

def draw_f1(ax, doc: dict, *, compact: bool = False) -> None:
    """F1 — peak RSS vs n_perms × n_lrs, log-log.

    `compact` is the 2×2 cell of the composed figure. There the figure legend
    carries the three arm colours, so the fits are labelled without their
    numbers (the caption carries those), which keeps the legend clear of the
    consensus annotation in the narrower panel.
    """
    for arm in ARMS:
        points = []
        for n_perms in (10, 100, 1000):
            for n_lrs in (200, 1000, 2000):
                label = f"t2_ra_50k_p{n_perms}_lrs{n_lrs}_{arm}"
                points.append((n_perms * n_lrs, gb(need(doc, label))))
        xs, ys = zip(*sorted(points))
        ax.plot(xs, ys, "o", color=OKABE_ITO[arm], markeredgecolor="white",
                markeredgewidth=0.3, zorder=3,
                label=None if compact else LABELS[arm])

    # the consensus-resource run (t5): 1,000 perms × 4,620 LRs, same 50k data
    consensus_x = 1000 * 4620
    for arm in ARMS:
        ax.plot([consensus_x],
                [gb(need(doc, f"t5_ra_50k_p1000_lrs4620_{arm}"))], "*",
                color=OKABE_ITO[arm], markersize=9, markeredgecolor="white",
                markeredgewidth=0.4, zorder=4)

    # Both fits run to the consensus point, the configuration they predict.
    xs = [1.5e3, consensus_x]
    fit = doc["law"]["fit_50k"]
    ax.plot(xs, [law_gb(fit["recorded_intercept_mb"], fit["recorded_slope"], x)
                 for x in xs],
            "--", color=OKABE_ITO["law"], linewidth=0.9, zorder=2,
            label=("law fit at 10k" if compact else
                   f"law fit at 10k: {fit['recorded_intercept_mb']} MB + "
                   f"{fit['recorded_slope']} KB × perms × LRs"))
    ax.plot(xs, [law_gb(fit["intercept_mb"], fit["slope_kb_per_perm_lr"], x)
                 for x in xs],
            "-", color=OKABE_ITO["law"], linewidth=0.9, alpha=0.55, zorder=2,
            label=("re-fit at 50k" if compact else
                   f"re-fit at 50k: {fit['intercept_mb']} MB + "
                   f"{fit['slope_kb_per_perm_lr']:.2f} KB × perms × LRs"))

    head = "consensus resource (4,620 LRs):"
    body = (f"release {gb(need(doc, 't5_ra_50k_p1000_lrs4620_release')):.1f} GB, "
            f"engine {gb(need(doc, 't5_ra_50k_p1000_lrs4620_rust')):.2f} GB")
    if compact:
        # two short lines at the panel's top right; the annotation sits clear of
        # the fits, which reach f≈0.84 of the axis at the right edge
        ax.annotate(f"{head}\n{body}", xy=(0.97, 0.99), xycoords="axes fraction",
                    fontsize=7, ha="right", va="top")
    else:
        ax.annotate(f"{head} {body}", xy=(0.97, 0.93), xycoords="axes fraction",
                    fontsize=7, ha="right", va="bottom")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.5e3, 6e6)
    ax.set_ylim(0.2, 60)
    ax.set_xticks([1e4, 1e5, 1e6])
    ax.set_yticks([0.25, 1, 4, 16, 32])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("permutations × LR pairs  ($n_{perms} \\times n_{lrs}$)")
    ax.set_ylabel("peak RSS (GB)")
    ax.grid(True, which="major", color=OKABE_ITO["grid"], linewidth=0.35, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.995), handlelength=1.7,
              borderpad=0.1, labelspacing=0.3)
    panel_letter(ax, "a")


def draw_bars(ax, doc: dict, metric: str, ylabel: str, *, letter: str,
              legend: bool, headroom: float = 1.30) -> None:
    """The 10k/50k/100k × p1000 grouped bars: `wall_s` (F2) or `rss_kb` (F3).

    `metric` is the manifest field to plot; `rss_kb` is converted to the
    manuscript's GB convention. `legend` is off in the composed 2×2, where the
    figure legend names the arms; `headroom` clears the value labels of the
    tallest bar (the RSS panel needs more, its release bars fill the panel).
    """
    datasets = (10000, 50000, 100000)
    width = 0.26
    for index, dataset in enumerate(datasets):
        for arm_index, arm in enumerate(ARMS):
            label = f"t1_ra_{dataset // 1000}k_p1000_{arm}"
            value = need(doc, label, metric)
            if metric == "rss_kb":
                value = gb(value)
            bars = ax.bar(index + (arm_index - 1) * width, value, width * 0.9,
                          color=OKABE_ITO[arm], edgecolor="white", linewidth=0.4,
                          zorder=3, label=LABELS[arm] if index == 0 else None)
            bar = bars[0]
            key = f"{dataset // 1000}k:{arm}"
            bar.set_gid(f"bar:{key}")
            # Rotated 90°, rotation_mode left at its default: ha/va then align the *rotated*
            # bbox, so ha="center" centres the number on the bar and the point offset in
            # xytext lifts it wholly clear of the bar top.  (`rotation_mode="anchor"` aligns
            # the unrotated box instead — that put the number half through the bar top and
            # half a glyph-height left of the bar centre.)
            ax.annotate(sig3(value), xy=(bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0.0, BAR_LABEL_PAD_PT), textcoords="offset points",
                        ha="center", va="bottom", rotation=90, fontsize=7,
                        gid=f"barvalue:{key}")

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([f"{d // 1000}k" for d in datasets])
    ax.set_xlabel("cells ($n_{obs}$)")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", color=OKABE_ITO["grid"], linewidth=0.35, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(0, ax.get_ylim()[1] * headroom)
    if legend:
        ax.legend(loc="upper left", bbox_to_anchor=(0.045, 1.0), ncols=3,
                  columnspacing=1.1, handlelength=1.4, borderpad=0.1)
    panel_letter(ax, letter)


def draw_f3(ax, doc: dict, *, compact: bool = False) -> None:
    """F4 — wall vs threads, three arms, over the points that exist."""
    rust_y = [need(doc, f"t3_ra_50k_p1000_t{t}_rust", "wall_s")
              for t in (1, 4, 8, 32)]
    ax.plot([1, 4, 8, 32], rust_y, "-o", color=OKABE_ITO["rust"],
            markeredgecolor="white", markeredgewidth=0.4, zorder=4,
            label=f"{LABELS['rust']} (RAYON_NUM_THREADS)")

    # The reference's thread points: j1 and j16 are the recorded thread-scaling
    # references (a different session); j4 is the frozen suite's own row.
    reference = {SUITE_JOBS: need(doc, "t1_ra_50k_p1000_release", "wall_s")}
    for ref in doc["recorded_references"]:
        if "thread-scaling reference" in (ref.get("note") or ""):
            jobs = int(ref["description"].rsplit("j", 1)[1])
            reference[jobs] = ref["wall_s"]
    xs = sorted(reference)
    ax.plot(xs, [reference[j] for j in xs], "--", color=OKABE_ITO["release"],
            linewidth=1.0, zorder=3, label=f"{LABELS['release']} (n_jobs)")
    _markers(ax, [(j, reference[j], j != SUITE_JOBS) for j in xs], "release")

    # The patched arm's wall is box-state dependent; all three of its j4 runs
    # are shown, so the spread is visible rather than averaged away.
    patched = [(need(doc, "t1_ra_50k_p1000_patched", "wall_s"), False)]
    for ref in doc["recorded_references"]:
        if "patched" in ref["description"] and "50k" in ref["description"] \
                and "p1000" in ref["description"]:
            patched.append((ref["wall_s"], True))
    ys = [value for value, _ in patched]
    # wrapped after the arm name: the label then fits a quarter-width panel too
    patched_label = (f"{LABELS['patched']}\n(n_jobs=4, spread over 3 box states)"
                     if compact else
                     f"{LABELS['patched']} (n_jobs=4, spread over 3 box states)")
    ax.plot([SUITE_JOBS, SUITE_JOBS], [min(ys), max(ys)], "-",
            color=OKABE_ITO["patched"], linewidth=2.4, alpha=0.35,
            solid_capstyle="butt", zorder=2, label=patched_label)
    _markers(ax, [(SUITE_JOBS, value, recorded) for value, recorded in patched],
             "patched")

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 4, 8, 16, 32]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0.8, 40)
    ax.set_ylim(0, 36)
    ax.set_xlabel("threads")
    ax.set_ylabel("wall (s)")
    ax.grid(True, axis="y", color=OKABE_ITO["grid"], linewidth=0.35, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", handlelength=1.7, borderpad=0.1, labelspacing=0.3)
    footer(ax, "open squares: recorded reference logs (a different session)")
    panel_letter(ax, "d")


def _markers(ax, points, arm: str) -> None:
    for x, y, recorded in points:
        ax.plot([x], [y], "s", color=OKABE_ITO[arm],
                markerfacecolor="white" if recorded else OKABE_ITO[arm],
                markeredgecolor=OKABE_ITO[arm], markeredgewidth=0.8,
                markersize=4, zorder=5)


# --------------------------------------------------------------------------
# composition

def margins(fig, *, top=0.97, bottom=0.14, hspace=None, wspace=None) -> None:
    fig.subplots_adjust(left=0.085, right=0.985, top=top, bottom=bottom,
                        **({"hspace": hspace} if hspace is not None else {}),
                        **({"wspace": wspace} if wspace is not None else {}))


def _tagged(artist, prefix: str) -> bool:
    return isinstance(artist, Text) and (artist.get_gid() or "").startswith(prefix)


def _rect_px(ax, rect) -> Bbox:
    """A data-space rectangle (a bar) as its display-pixel bbox."""
    x0, y0 = ax.transData.transform((rect.get_x(), rect.get_y()))
    x1, y1 = ax.transData.transform((rect.get_x() + rect.get_width(),
                                     rect.get_y() + rect.get_height()))
    return Bbox.from_extents(x0, y0, x1, y1)


def _legends(fig) -> list:
    out = list(fig.legends)
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            out.append(legend)
    return out


def check_placement(fig, name: str) -> None:
    """W12: measure and prove the two placement rules on the figure about to be written.

    Panel letters — the text bbox must lie entirely outside the axes bbox, above its top
    edge, with the measured gap equal to the letter's own `LETTER_PAD_PT` (W14: 6.0 pt); it
    must stay inside the figure canvas, so nothing is clipped, and clear of every legend and
    of the other panels' labels.  Bar value labels — `ha` must be `center`, the rotated bbox centred on
    the bar's x within 2 px, its bottom edge at or above the bar's top edge (gap =
    `BAR_LABEL_PAD_PT`), and clear of every bar and every other label in the panel.

    The canvas is drawn at `DPI` first, so the pixels measured are the pixels saved.  Every
    measurement is printed — `ops/logs/` reports quote them — and a violation raises, so a
    render that drifts fails instead of shipping.
    """
    fig.set_dpi(DPI)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    frame = fig.bbox
    letters = fig.findobj(lambda a: _tagged(a, "letter:"))
    values = fig.findobj(lambda a: _tagged(a, "barvalue:"))
    print(f"{name}: placement — {len(letters)} panel letter(s), "
          f"{len(values)} bar value label(s)")

    for text in letters:
        letter = text.get_gid().split(":", 1)[1]
        ax = text.axes
        tb = text.get_window_extent(renderer)
        ab = ax.get_window_extent(renderer)
        gap_px = tb.y0 - ab.y1
        assert text.get_fontweight() == "bold", f"{name}: letter {letter} is not bold"
        assert text.get_fontsize() >= 7.0, f"{name}: letter {letter} is under 7 pt"
        assert gap_px >= 0.0, (
            f"{name}: letter {letter} crosses the axes top by {-gap_px:.2f} px")
        expect_px = LETTER_PAD_PT * DPI / 72
        assert abs(gap_px - expect_px) <= 0.5, (
            f"{name}: letter {letter} gap is {gap_px:.2f} px, recorded expectation "
            f"{expect_px:.2f} px ({LETTER_PAD_PT:g} pt)")
        assert frame.contains(tb.x0, tb.y0) and frame.contains(tb.x1, tb.y1), (
            f"{name}: letter {letter} leaves the figure canvas and would be clipped")
        for legend in _legends(fig):
            assert not tb.overlaps(legend.get_window_extent(renderer)), (
                f"{name}: letter {letter} touches a legend")
        for other in fig.axes:
            if other is not ax:
                assert not tb.overlaps(other.get_tightbbox(renderer)), (
                    f"{name}: letter {letter} touches a neighbouring panel's labels")
        print(f"  {name}: letter ({letter}) — {gap_px:.2f} px ({gap_px * 72 / DPI:.2f} pt) "
              f"above the axes top, {frame.y1 - tb.y1:.2f} px below the figure top, "
              f"{text.get_fontsize():g} pt bold")

    for text in values:
        key = text.get_gid().split(":", 1)[1]
        ax = text.axes
        bar = _rect_px(ax, next(r for r in ax.patches if r.get_gid() == f"bar:{key}"))
        tb = text.get_window_extent(renderer)
        centre_dx = (tb.x0 + tb.x1) / 2 - (bar.x0 + bar.x1) / 2
        gap_px = tb.y0 - bar.y1
        assert text.get_ha() == "center", f"{name}: bar label {key} is not ha=center"
        assert text.get_fontsize() >= 7.0, f"{name}: bar label {key} is under 7 pt"
        assert abs(centre_dx) <= 2.0, (
            f"{name}: bar label {key} is {centre_dx:+.2f} px off the bar centre (limit 2)")
        assert gap_px >= 0.0, (
            f"{name}: bar label {key} crosses its bar top by {-gap_px:.2f} px")
        for other in ax.patches:
            assert not tb.overlaps(_rect_px(ax, other)), (
                f"{name}: bar label {key} overlaps a bar")
        for other in values:
            if other is not text:
                assert not tb.overlaps(other.get_window_extent(renderer)), (
                    f"{name}: bar label {key} overlaps another value label")
        print(f"  {name}: bar {key} — centre dx {centre_dx:+.2f} px, {gap_px:.2f} px "
              f"({gap_px * 72 / DPI:.2f} pt) above its bar top, {text.get_fontsize():g} pt")


def save(fig, out: pathlib.Path, name: str) -> list[pathlib.Path]:
    check_placement(fig, name)
    written = []
    for suffix in ("pdf", "png"):
        path = out / f"{name}.{suffix}"
        fig.savefig(path, dpi=DPI)
        written.append(path)
    plt.close(fig)
    return written


def standalone(doc: dict, out: pathlib.Path) -> list[pathlib.Path]:
    # the top margins carry the out-of-axes panel letter (a 9 pt line plus its 6 pt gap)
    written = []
    fig, ax = plt.subplots(figsize=(mm(WIDTH_MM), mm(118)))
    draw_f1(ax, doc)
    margins(fig, top=0.935, bottom=0.13)
    written += save(fig, out, "fig1_memory_law")

    fig, ax = plt.subplots(figsize=(mm(WIDTH_MM), mm(96)))
    draw_bars(ax, doc, "wall_s", "wall (s)", letter="b", legend=True)
    margins(fig, top=0.935, bottom=0.17)
    written += save(fig, out, "fig2_wall")

    fig, ax = plt.subplots(figsize=(mm(WIDTH_MM), mm(96)))
    draw_bars(ax, doc, "rss_kb", "peak RSS (GB)", letter="c", legend=True,
              headroom=1.45)
    margins(fig, top=0.935, bottom=0.17)
    written += save(fig, out, "fig3_rss")

    fig, ax = plt.subplots(figsize=(mm(WIDTH_MM), mm(105)))
    draw_f3(ax, doc)
    margins(fig, top=0.935, bottom=0.15)
    written += save(fig, out, "fig4_thread_scaling")
    return written


def combined(doc: dict, out: pathlib.Path) -> list[pathlib.Path]:
    """a+b+c+d in a 2×2 grid, one column wide — the App Note's single figure.

    a (memory law) over c (peak RSS) is the memory column; b (wall clock) over
    d (thread scaling) is the time column. The arm colours are shared by every
    panel, so one figure legend at the foot names them and b and c carry none.
    """
    fig = plt.figure(figsize=(mm(WIDTH_MM), mm(165)))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])
    draw_f1(ax_a, doc, compact=True)
    draw_bars(ax_b, doc, "wall_s", "wall (s)", letter="b", legend=False)
    draw_bars(ax_c, doc, "rss_kb", "peak RSS (GB)", letter="c", legend=False)
    draw_f3(ax_d, doc, compact=True)
    margins(fig, top=0.96, bottom=0.105, hspace=0.30, wspace=0.28)
    handles, labels = ax_b.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.004),
               ncols=3, frameon=False, fontsize=7, handlelength=1.4,
               columnspacing=1.4)
    return save(fig, out, "fig_all")


def captions(doc: dict) -> dict:
    fit = doc["law"]["fit_50k"]
    return {
        "fig1_memory_law": {
            "letter": "a",
            "caption": (
                "The memory law. Peak resident memory of `rank_aggregate` against "
                "the size of the permutation null, `n_perms × n_lrs`, for the three "
                "implementations (50k cells, log-log). The engine is flat within "
                "each resource column and grows only with the LR table it carries; "
                "the reference is linear in `n_perms × n_lrs` with a cell-count "
                f"term in its intercept (re-fit at 50k: {fit['intercept_mb']} MB, "
                f"{fit['slope_kb_per_perm_lr']:.2f} KB per perm × LR) that the "
                f"recorded 10k law ({fit['recorded_intercept_mb']} MB + "
                f"{fit['recorded_slope']} KB) carries as "
                f"{fit['recorded_intercept_mb']} MB. Stars: the consensus-resource "
                "run (4,620 LR pairs, 50k × 1,000 perms). Memory is quoted in GB "
                "= 1000 MB over the manifest's MB (kB / 1024)."),
            "source_keys": ["law", "law_grid", "t5_consensus_4620"],
        },
        "fig2_wall": {
            "letter": "b",
            "caption": (
                "Wall clock at 10k/50k/100k cells (2,000 LR pairs, 1,000 "
                "permutations, 4 threads). Wall is the in-process measurement "
                "(see `bench/RESULTS.md`, *Wall bases*)."),
            "source_keys": ["wall_rss_matrix"],
        },
        "fig3_rss": {
            "letter": "c",
            "caption": (
                "Peak resident memory at 10k/50k/100k cells (same configuration). "
                "The reference's memory is set by the permutation null, not the "
                "cell count, so it moves only when the resource does; the engine's "
                "is set by the block. Memory is quoted in GB = 1000 MB over the "
                "manifest's MB (kB / 1024), so 50k release reads "
                f"{gb(need(doc, 't1_ra_50k_p1000_release', 'rss_kb')):.1f} GB."),
            "source_keys": ["wall_rss_matrix"],
        },
        "fig4_thread_scaling": {
            "letter": "d",
            "caption": (
                "Thread scaling of wall clock at 50k × 2,000 LRs × 1,000 perms. The "
                "engine runs on `RAYON_NUM_THREADS`; the Python arms on joblib "
                "`n_jobs`, with their Numba kernels at 32 threads in every run "
                "recorded here, so the engine's 4-thread point is the conservative "
                "one. Open squares are the recorded reference logs (a different "
                "session); filled squares the frozen suite's own rows. The patched "
                "arm's wall is box-state dependent, so all three of its `n_jobs=4` "
                "measurements are shown rather than one."),
            "source_keys": ["thread_scaling", "patched_wall_spread",
                            "recorded_references"],
        },
        "fig_all": {
            "letter": "a + b + c + d",
            "caption": ("Panels a–d: the memory law, wall clock and peak memory at "
                        "10k/50k/100k cells, and thread scaling. Full captions for "
                        "the individual panels are in `paper/figures/captions.json`."),
            "source_keys": ["fig1_memory_law", "fig2_wall", "fig3_rss",
                            "fig4_thread_scaling"],
        },
    }


def _short(path: pathlib.Path) -> pathlib.Path:
    """Repo-relative when it is inside the repo, as-is otherwise (the output
    directory is a free argument, not necessarily under `paper/`)."""
    try:
        return path.relative_to(REPO)
    except ValueError:
        return path


def main(argv: list[str]) -> int:
    results = pathlib.Path(argv[1]) if len(argv) > 1 else RESULTS
    out = pathlib.Path(argv[2]) if len(argv) > 2 else OUT
    out.mkdir(parents=True, exist_ok=True)
    style()
    doc = json.loads(results.read_text())
    written = standalone(doc, out) + combined(doc, out)
    (out / "captions.json").write_text(json.dumps(captions(doc), indent=2) + "\n")
    for path in written:
        print(_short(path))
    print(_short(out / "captions.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
