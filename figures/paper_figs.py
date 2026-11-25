#!/usr/bin/env python3
# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)

"""The data figures: fig-waste, fig-overhead-same-recall, fig-adaptive-savings-both, fig-shard-tail and
fig-verdicts. Values are read by data.py from RESULTS (default: the shipped results); PDFs go to OUT
(default: $REPRO/work/figures).

Style: column (3.33 in) or full (7.0 in) width, 7 pt text, TrueType fonts embedded (pdf.fonttype 42) for PDF/A.
No titles or footnotes inside the figures.

Usage: python3 paper_figs.py [name ...]      (default: all five)
Each figure prints the values it plots, so a run can be compared with the shipped results.
"""
import os
import sys

import matplotlib
matplotlib.use("pdf")
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "DejaVu Sans", "font.size": 7,
    "axes.labelsize": 7, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "lines.linewidth": 1.2,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "legend.frameon": False, "figure.dpi": 300,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.02, "figure.constrained_layout.w_pad": 0.02,
})
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

import data as D                  # noqa: E402

COL, FULL = 3.33, 7.0
BLUE, ORANGE = "#2a78d6", "#eb6834"
OUT = os.path.abspath(os.environ.get("OUT", os.path.join(D.REPRO, "work", "figures")))
MAIN_DS = (("t2i", "T2I 10M"), ("msmarco", "MS MARCO 8.8M"))


def finish(fig, name):
    for ax in fig.axes:
        ax.grid(axis="y", color="#e6e8eb", zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)
    print(f"-> {os.path.join(OUT, name + '.pdf')}")


# ---------------------------------------------------------------- fig-waste
WASTE_PARTS = (("oracle", "#1b4f8a", "white", "necessary (oracle)"),       # dark -> light, so the three parts stay
               ("confirm", "#5b9bd5", "white", "extra search"),           # apart in grayscale print
               ("empty", "#f2a65a", "#0b0b0b", "no contribution"))


def fig_waste():
    """Where fixed ef spends its distance computations at the ef that reaches a recall@20 of 0.95 at each S:
    necessary (arrival-time oracle), extra search on shards that contributed, and shards that contributed nothing.
    100% stacked, one panel per dataset; the matched ef is printed below each bar."""
    fig, axs = plt.subplots(1, 2, figsize=(FULL, 2.0), sharey=True, squeeze=False)
    for ax, (ds, dname) in zip(axs[0], MAIN_DS):
        Ss = list(D.SS)
        w = [D.waste(S, ds) for S in Ss]
        x = np.arange(len(Ss))
        bot = np.zeros(len(Ss))
        shares = {}
        for key, c, tc, lab in WASTE_PARTS:
            v = np.array([wi[key] / wi["total"] * 100 for wi in w])
            shares[key] = v
            ax.bar(x, v, bottom=bot, width=0.62, color=c, label=lab, zorder=3)
            for xi, (vi, bi) in enumerate(zip(v, bot)):
                if vi > 6:
                    ax.text(xi, bi + vi / 2, f"{vi:.0f}", ha="center", va="center", fontsize=5.8, color=tc)
            bot += v
        ax.set_xticks(x, [f"S={s}\nef={wi['ef']}" for s, wi in zip(Ss, w)])
        if ax.get_subplotspec().colspan.start == 0:
            ax.set_ylabel("share of distance computations (%)")
        ax.set_ylim(0, 100)
        ax.set_title(dname, fontsize=7, pad=3)
        for i, S in enumerate(Ss):
            print(f"waste {dname} S={S} ef={w[i]['ef']}: fixed ndis {w[i]['total']:,.1f}, oracle {shares['oracle'][i]:.2f}%, "
                  f"extra {shares['confirm'][i]:.2f}%, no contribution {shares['empty'][i]:.2f}%")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=3, handlelength=1.1, columnspacing=1.0)
    finish(fig, "fig-waste")


# ---------------------------------------------------------------- fig-overhead-same-recall
OVH_R, OVH_EF = 0.95, 200
OVH_YLIM, OVH_YTICKS = (0.5, 60), [1, 2, 5, 10, 20, 50]
OVH_FLOOR = OVH_YLIM[0]


def _ovh_bars(ax, x, series, width, fs=5.5, rot=90):
    """series = [(values, color, label)] -- ratio to fixed S=1 as log-axis bars, value written above each bar"""
    n = len(series)
    for j, (v, col, lab) in enumerate(series):
        off = (j - (n - 1) / 2) * width
        v = np.array(v)
        ax.bar(x + off, v - OVH_FLOOR, bottom=OVH_FLOOR, width=width * 0.9, color=col, label=lab, zorder=3)
        for xi, a in zip(x, v):
            ax.text(xi + off, a * 1.06, f"{a:.2f}" if a < 1 else (f"{a:.1f}" if round(a, 1) < 10 else f"{a:.0f}"),
                    ha="center", va="bottom", fontsize=fs, color="#0b0b0b", zorder=5, rotation=rot)
    ax.set_yscale("log")
    ax.set_ylim(*OVH_YLIM)
    ax.set_yticks(OVH_YTICKS, [f"{t:g}" for t in OVH_YTICKS])
    ax.minorticks_off()


def fig_overhead_same_recall(R=OVH_R):
    """Sharding overhead of fixed ef at the same recall R: cost at S / cost of fixed ef at S=1, CPU time and distance
    computations, without adaptive termination. The ef that reaches R at each S (a matched cell, not an
    interpolation) is printed, not drawn: fig-waste shows it below its bars."""
    Ss = list(D.SS)
    x = np.arange(len(Ss))
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.8), sharey=True)
    for c, (ds, name) in enumerate(MAIN_DS):
        v = D.overhead(ds, R, "fixed", OVH_EF)
        nd, cpu, knob = [v["R"][S][0] for S in Ss], [v["R"][S][1] for S in Ss], [v["R"][S][2] for S in Ss]
        ax = axes[c]
        _ovh_bars(ax, x, [(cpu, ORANGE, "fixed $ef$: CPU time"), (nd, BLUE, "fixed $ef$: distance computations")], 0.38)
        ax.set_title(name, fontsize=6.5, pad=3)
        ax.set_xticks(x, [f"{s}" for s in Ss], fontsize=6)
        print(f"overhead {name} same recall {R:.2f}: (S, ndis x, CPU x, ef) "
              f"{[(S, round(a, 3), round(b, 3), round(e, 1)) for S, a, b, e in zip(Ss, nd, cpu, knob)]}")
    axes[0].set_ylabel("cost relative to $S$=1", fontsize=6.5)
    fig.supxlabel("$S$", fontsize=6.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=2, handlelength=1.4, columnspacing=1.0, fontsize=5.8)
    finish(fig, "fig-overhead-same-recall")


# ---------------------------------------------------------------- fig-adaptive-savings-both
def fig_adaptive_savings_both(Rs=(0.95,)):
    """Share of the work adaptive termination saves against fixed ef at the same S. Left column: same recall R
    (matched cells), CPU time and distance computations; right column: same ef (200, adaptive with the calibrated
    gammas), CPU time only (the two metrics agree within two points there). Rows: datasets."""
    Ss = list(D.SS)
    x = np.arange(len(Ss))
    cols = [("R", R, f"same recall = {R:.2f}") for R in Rs] + [("ef", Rs[-1], f"same $ef$ = {OVH_EF}")]
    fig, axes = plt.subplots(2, len(cols), figsize=(COL, 2.3), sharex=True, sharey=True)
    for rr, (ds, name) in enumerate(MAIN_DS):
        for cc, (key, R, title) in enumerate(cols):
            v = D.savings(ds, R, OVH_EF)
            ax = axes[rr, cc]
            bars = (((-0.18, 1, ORANGE, "CPU time"), (0.18, 0, BLUE, "distance computations")) if key == "R"
                    else ((0.0, 1, ORANGE, "CPU time"),))
            for off, i, col, lab in bars:
                vals = [100 * (1 - v[key][S][i]) for S in Ss]
                ax.bar(x + off, vals, width=0.34 if key == "R" else 0.5, color=col, label=lab, zorder=3)
                for xi, a in zip(x, vals):
                    ax.text(xi + off, a + 1.5, f"{a:.0f}", ha="center", va="bottom", fontsize=4.8)
            ax.set_ylim(0, 80)
            ax.set_yticks([0, 20, 40, 60])
            if rr == 0:
                ax.set_title(title, fontsize=6.5, pad=2)
            print(f"savings {name} {title.replace('$', '')}: (S, ndis saved %, CPU saved %) "
                  f"{[(S, round(100 * (1 - v[key][S][0]), 1), round(100 * (1 - v[key][S][1]), 1)) for S in Ss]}")
    for ax in axes[1]:
        ax.set_xticks(x, [f"{s}" for s in Ss], fontsize=6)
    for ax, (_, name) in zip(axes[:, 0], MAIN_DS):
        ax.set_ylabel(name, fontsize=6.5)
    fig.supylabel("saved (%)", fontsize=7)
    fig.supxlabel("$S$", fontsize=6.5)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=2, handlelength=1.1, columnspacing=1.0, fontsize=6)
    finish(fig, "fig-adaptive-savings-both")


# ---------------------------------------------------------------- fig-shard-tail
def fig_shard_tail():
    """CPU time saved per query against fixed ef at the same S and ef. Columns: datasets. Bars: average query, summed
    over shards (total) or at the slowest shard; dots: the same quantity over the costliest 1% of queries (200 on T2I,
    65 on MS MARCO). Vertical lines: paired-bootstrap 95% intervals. The distance-computation version is printed, not
    drawn (the same order within 2.6 points)."""
    get = D.shard_tail_reader()
    cells = D.TAIL_CELLS
    x = np.arange(len(cells))
    fig, axes = plt.subplots(1, len(MAIN_DS), figsize=(COL, 1.55), sharey=True)
    for c, (ds, name) in enumerate(MAIN_DS):
        for metric in ("cpu", "ndis"):
            vals = [get(ds, S, ef, metric) for S, ef in cells]
            for stat in ("mean", "worst1"):
                for agg in ("total", "slowest"):
                    v = [vi[agg, stat][:3] for vi in vals]
                    print(f"shard-tail {name} {metric} {stat} {agg}: "
                          + ", ".join(f"S={S} ef={ef} {p:.1f} [{lo:.1f}, {hi:.1f}]" for (S, ef), (p, lo, hi) in zip(cells, v)))
            if metric != "cpu":
                continue
            ax = axes[c]
            for off, agg, col, lab in ((-0.19, "total", BLUE, "total (all shards)"),
                                       (0.19, "slowest", ORANGE, "slowest shard")):
                raw = [vi[agg, "mean"] for vi in vals]
                v = np.array([q[:3] for q in raw])
                w = np.array([vi[agg, "worst1"][:3] for vi in vals])
                labels = [str(q[3]) if len(q) > 3 else f"{q[0]:.0f}".replace("-0", "0") for q in raw]
                ax.bar(x + off, v[:, 0], width=0.36, color=col, label=lab, zorder=3)
                ax.errorbar(x + off, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="none",
                            ecolor="#0b0b0b", elinewidth=0.6, capsize=1.2, zorder=4)
                ax.errorbar(x + off, w[:, 0], yerr=[w[:, 0] - w[:, 1], w[:, 2] - w[:, 0]], fmt="o", ms=2.2,
                            mfc="white", mec="#0b0b0b", mew=0.6, ecolor="#0b0b0b", elinewidth=0.5, capsize=0, zorder=5)
                for xi, (pt, lo, hi), text in zip(x, v, labels):
                    ax.text(xi + off, max(hi, 0) + 0.8, text, ha="center", va="bottom", fontsize=4.6, color="#0b0b0b")
        ax = axes[c]
        ax.axhline(0, color="#0b0b0b", lw=0.6, zorder=4)
        ax.axvline(2.5, color="#c9ccd1", lw=0.6, zorder=1)
        ax.set_title(name, fontsize=6.5, pad=3)
        ax.set_xticks(x, [f"{ef}\n$S$={S}" if ef == 200 else f"{ef}" for S, ef in cells], fontsize=5.5)
    axes[0].set_ylim(0, 82)
    axes[0].set_yticks([0, 25, 50, 75])
    axes[0].set_ylabel("CPU time saved (%)", fontsize=6.5)
    fig.supxlabel("$ef$", fontsize=6.5)
    h, l = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], ls="none", marker="o", ms=2.2, mfc="white", mec="#0b0b0b", mew=0.6))
    l.append("costliest 1% of queries")
    fig.legend(h, l, loc="outside upper center", ncol=3, handlelength=1.1, columnspacing=0.9, fontsize=5.8)
    finish(fig, "fig-shard-tail")


# ---------------------------------------------------------------- fig-verdicts
VERDICT_DS = (("t2i", "T2I 10M"), ("msmarco", "MS MARCO"), ("glove", "GloVe"), ("sift", "SIFT"),
              ("deep", "DEEP\n100M"), ("t2i100m", "T2I\n100M"))
TOL_PP = -1.0


def fig_verdicts():
    """Paired-bootstrap tolerance verdicts of adaptive termination (verdicts/bootstrap.json, arm rule): one point per
    configuration, ordered dataset -> S -> ef limit (100, 200, 300). Point = merged recall@20 of adaptive minus fixed
    ef (pp), vertical line = 95% interval, dashed line = the 1 pp tolerance. Filled blue = within, open orange =
    undecided (a configuration judged exceeds, absent from the shipped results, would be drawn as a black cross).
    Tick = S, text below = dataset."""
    rows = D.verdict_rows("rule")
    fig, ax = plt.subplots(figsize=(COL, 1.5))
    x, sx, sl, groups, ys = 0.0, [], [], [], []
    seen = set()
    for ds, name in VERDICT_DS:
        rs = sorted((r for r in rows if r["ds"] == ds), key=lambda r: (r["S"], r["ef"]))
        if not rs:
            continue
        x0 = x
        for S in sorted({r["S"] for r in rs}):
            xs = x
            for r in (r for r in rs if r["S"] == S):
                verdict = r["verdict"]
                if verdict not in ("within", "undecided", "exceeds"):
                    raise ValueError(f"verdicts/bootstrap.json: unknown verdict {verdict!r} in {r}")
                seen.add(verdict)
                col = {"within": BLUE, "undecided": ORANGE, "exceeds": "#0b0b0b"}[verdict]
                ax.plot([x, x], r["delta_ci"], color=col, lw=0.8, zorder=3)
                if verdict == "exceeds":
                    ax.plot(x, r["delta_pp"], "x", ms=2.6, color=col, mew=0.8, zorder=4)
                else:
                    ax.plot(x, r["delta_pp"], "o", ms=2.3, mfc="white" if verdict == "undecided" else col, mec=col,
                            mew=0.7, zorder=4)
                ys += list(r["delta_ci"])
                print(f"verdict {ds} S={S} ef={r['ef']}: {r['delta_pp']:+.3f} pp "
                      f"[{r['delta_ci'][0]:+.3f}, {r['delta_ci'][1]:+.3f}] {verdict}")
                x += 1
            sx.append((xs + x - 1) / 2)
            sl.append(str(S))
        groups.append(((x0 + x - 1) / 2, name))
        x += 1.6
    unknown = sorted({r["ds"] for r in rows} - {d for d, _ in VERDICT_DS})
    if unknown:
        raise ValueError(f"verdicts/bootstrap.json: dataset(s) {unknown} have no place in fig-verdicts")
    n = {v: sum(r["verdict"] == v for r in rows) for v in ("within", "undecided", "exceeds")}
    print(f"verdicts: {len(rows)} configurations, within {n['within']}, undecided {n['undecided']}, exceeds {n['exceeds']}")
    ax.axhline(TOL_PP, color="#0b0b0b", lw=0.7, ls=(0, (3, 2)), zorder=2)
    ax.set_xlim(-1, x - 0.6)
    lo, hi = min([-1.2] + [y - 0.05 for y in ys]), max([-0.25] + [y + 0.05 for y in ys])
    ax.set_ylim(lo, hi)
    ax.set_yticks([t for t in np.arange(-2.0, 0.01, 0.2).round(1) if lo <= t <= hi - 0.1])
    ax.set_ylabel("merged recall, adaptive\nminus fixed $ef$ (pp)", fontsize=6.3)
    ax.set_xticks(sx, sl, fontsize=4.8)
    ax.tick_params(axis="x", length=1.5, pad=1.2)
    for gx, name in groups:
        ax.text(gx, -0.11, name, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=5.6,
                linespacing=0.95)
    ax.plot([], [], "o", ms=2.3, color=BLUE, label="within")
    ax.plot([], [], "o", ms=2.3, mfc="white", mec=ORANGE, mew=0.7, label="undecided")
    if "exceeds" in seen:
        ax.plot([], [], "x", ms=2.6, color="#0b0b0b", mew=0.8, label="exceeds")
    ax.plot([], [], color="#0b0b0b", lw=0.7, ls=(0, (3, 2)), label="tolerance")
    ax.legend(loc="upper right", ncol=4 if "exceeds" in seen else 3, handlelength=1.2, columnspacing=0.9,
              fontsize=5.8, borderaxespad=0.2)
    finish(fig, "fig-verdicts")


FIGS = {"fig-waste": fig_waste, "fig-overhead-same-recall": fig_overhead_same_recall,
        "fig-adaptive-savings-both": fig_adaptive_savings_both, "fig-shard-tail": fig_shard_tail,
        "fig-verdicts": fig_verdicts}

if __name__ == "__main__":
    names = sys.argv[1:] or list(FIGS)
    bad = [n for n in names if n not in FIGS]
    if bad:
        sys.exit(f"unknown figure(s) {bad}; choose from {list(FIGS)}")
    print(f"RESULTS = {D.RESULTS}\nOUT = {OUT}")
    try:
        for n in names:
            FIGS[n]()
    except (FileNotFoundError, ValueError, KeyError) as e:
        sys.exit(f"!! {n}: {type(e).__name__}: {e}")
