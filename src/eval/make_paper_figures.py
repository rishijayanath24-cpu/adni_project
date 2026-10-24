"""
Regenerate every paper figure from the committed result tables.

Reads only paper/tables/*.csv (no data access, no model fitting), so every
figure is reproducible without ADNI. Writes PNGs next to the manuscript.

Every figure is drawn at its final printed width (6.3 in, the text width of
the A4 template) and placed at that width, so the type sizes set here are the
type sizes in the paper. Two-panel figures are single images with the panels
side by side and labelled (a) and (b).

Run:  python -m src.eval.make_paper_figures
"""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "paper" / "tables"
OUT = ROOT / "paper"

# Okabe-Ito colour-blind-safe palette
BLUE, ORANGE, GREEN, RED, PURPLE, GREY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#6B7280")
ORANGE_TXT = "#B87800"          # darker orange for text on white
W = 6.3                          # final text width in inches

plt.rcParams.update({
    "font.family": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
})

# Short display names. The RL policy appears at two cost weights in the paper,
# so its weight is always stated.
LABEL = {
    "cheap_only": "cheap-only", "fixed_MRI": "fixed MRI",
    "fixed_MRI+FDG": "fixed MRI+FDG", "acquire_all": "acquire-all",
    "random_rate0.5": "random", "umos_mri_to_amyloid": "uncertainty cascade [1]",
    "promuse_full_ladder_sharedpred": "staged evidential [2]",
    "dqn_lambda0.05": "RL test-ordering [3], w = 0.05",
    "dqn_lambda0.01": "RL test-ordering [3], w = 0.01",
    "ours_MRI": "MRI only (ours)", "ours_MRI+FDG": "MRI+FDG (ours)",
}


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / f"{name}.csv")


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("wrote", name)


def _box(ax, x0, y0, x1, y1, text, ec, fc="white", fs=9, bold=False, lw=1.3):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.0,rounding_size=1.0",
                                fc=fc, ec=ec, lw=lw))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.2)


def _arrow(ax, p, q, color="#444444", lw=1.2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, lw=lw,
                                 color=color, shrinkA=0, shrinkB=0))


def _panel_title(ax, letter, text):
    ax.set_title(f"({letter}) {text}", loc="left", fontsize=9.5)


# --- Figure 1: evaluation pipeline (left to right) ------------------------------------
def fig_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(W, 3.55))
    ax.set_xlim(-0.5, 100.5)
    ax.set_ylim(-9, 60)
    ax.axis("off")
    ax.grid(False)
    fs = 8.2
    _box(ax, 0, 38, 17, 58, "ADNI\n(n = 3,762),\nfiltered to\nMCI at\nbaseline", BLUE, fs=fs)
    _box(ax, 0, 14, 17, 30, "MCI cohort\nn = 1,566", BLUE, fs=8.8, bold=True)
    _box(ax, 21, 37, 40, 58, "Always available\n(zero cost):\ndemographics,\neducation, APOE,\nMMSE, CDR-SB", BLUE, fs=fs)
    _box(ax, 21, 13, 40, 31, "Acquirable blocks\n(policy-gated):\nMRI, FDG-PET,\namyloid-PET, CSF", ORANGE, fs=fs)
    _box(ax, 21, -1, 40, 9, "Cost regimes:\nprice-like, flat,\nburden", ORANGE, fs=7.6, lw=1.1)
    _box(ax, 44, 8, 61, 52, "Subject-level\nsplit, stratified\non outcome and\nfield strength,\nmade before\nany pre-\nprocessing\nstatistic", BLUE, fs=fs)
    _box(ax, 65, 49, 84, 58, "Train / validation:\n5-fold CV", BLUE, fs=fs)
    _box(ax, 65, 28, 84, 45, "Acquisition policy\nchooses blocks per\nsubject: cheap-only,\nacquire-all, fixed,\nrandom, 3 published", ORANGE, fs=7.8)
    _box(ax, 65, 14, 84, 24, "Shared logistic-\nregression bank\n(refit per fold)", GREEN, fs=fs)
    _box(ax, 65, -1, 84, 9, "Held-out test set,\nscored at the end", BLUE, fs=7.8)
    _box(ax, 88, 8, 100, 46, "Held-out\ntest\nevaluation:\nconversion\nat 12, 24\nand 36\nmonths", GREEN, fs=fs, bold=True)
    _arrow(ax, (8.5, 38), (8.5, 30))
    _arrow(ax, (17, 25), (21, 47))
    _arrow(ax, (17, 20), (21, 22))
    _arrow(ax, (30.5, 13), (30.5, 9))
    _arrow(ax, (40, 47), (44, 40))
    _arrow(ax, (40, 22), (44, 24))
    _arrow(ax, (40, 4.5), (44, 12))
    _arrow(ax, (61, 46), (65, 53.5))
    _arrow(ax, (74.5, 49), (74.5, 45))
    _arrow(ax, (74.5, 28), (74.5, 24))
    _arrow(ax, (61, 12), (65, 4.5))
    _arrow(ax, (84, 19), (88, 24))
    _arrow(ax, (84, 4.5), (88, 12))
    handles = [Line2D([0], [0], color=c, lw=3) for c in (BLUE, ORANGE, GREEN)]
    ax.legend(handles, ["Cohort / data stage", "Acquisition decision", "Modelling / outcome"],
              loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04),
              fontsize=8.2, handlelength=1.6, columnspacing=1.6)
    save(fig, "fig_pipeline_block_diagram")


# --- Figure 2: proposed algorithm flow ------------------------------------------------
def fig_algorithm_flow() -> None:
    fig, ax = plt.subplots(figsize=(W, 4.7))
    ax.set_xlim(-2, 105)
    ax.set_ylim(-1, 76)
    ax.axis("off")
    ax.grid(False)
    fs = 8.6
    cols = [(0, 21), (26, 48), (53, 75), (81, 103)]
    ax.text(0, 72.5, "Training (per outer fold; subject-level, leakage-free)", fontsize=10,
            fontweight="bold", color=BLUE, va="center")
    _box(ax, cols[0][0], 50, cols[0][1], 66, "Fold training rows\n$D_{tr}$: development\nsubjects at\nhorizon $h$", BLUE, fs=fs)
    _box(ax, cols[0][0], 29, cols[0][1], 45, "Pool $P$: labelled\nsubjects lacking\none or more\nmodalities", BLUE, fs=fs)
    _box(ax, cols[1][0], 36, cols[1][1], 60, "Pooled state data\n$T_S=D_{tr}\\cup P_S$\nwhere $P_S$ holds\nevery block in $S$", ORANGE, fs=fs)
    _box(ax, cols[2][0], 36, cols[2][1], 60, "Fit imputer and\nscaler on $T_S$ only;\nselect $C$ by inner\nCV on $D_{tr}$", GREEN, fs=fs)
    _box(ax, cols[3][0], 36, cols[3][1], 60, "Logistic\nregression $f_S$\nfor state\n$S=\\mathrm{CHEAP}\\cup A$", GREEN, fs=fs)
    _arrow(ax, (21, 58), (26, 52))
    _arrow(ax, (21, 37), (26, 44))
    _arrow(ax, (48, 48), (53, 48))
    _arrow(ax, (75, 48), (81, 48))
    ax.text(0, 25, "Inference (one patient $x$)", fontsize=10, fontweight="bold",
            color=ORANGE_TXT, va="center")
    _box(ax, cols[0][0], 1, cols[0][1], 20, "Always-available\ncovariates $x_{cheap}$\n(age, APOE4,\nMMSE, …)", BLUE, fs=fs)
    _box(ax, cols[1][0], 1, cols[1][1], 20, "Acquire the same\nset for everyone:\n$A$ = {MRI} or\n$A$ = {MRI, FDG}", ORANGE, fs=8.0, bold=True)
    _box(ax, cols[2][0], 1, cols[2][1], 20, "Features: 10\nsignature MRI\nmeasures, ventricles,\nICV (+ FDG if\nFDG $\\in A$)", ORANGE, fs=7.8)
    _box(ax, cols[3][0], 1, cols[3][1], 20, "", GREEN, fs=fs)
    ax.text(92, 14.8, "Risk $\\hat p=f_S(x)$", ha="center", va="center", fontsize=fs)
    ax.text(92, 6.5, "cost $\\kappa=\\sum_{b\\in A}c_b$", ha="center", va="center", fontsize=fs)
    _arrow(ax, (21, 10.5), (26, 10.5))
    _arrow(ax, (48, 10.5), (53, 10.5))
    _arrow(ax, (75, 10.5), (81, 10.5))
    _arrow(ax, (92, 36), (92, 20), color=GREEN)
    ax.text(94, 28, "$f_S$", fontsize=10, color=GREEN, va="center")
    save(fig, "fig_algorithm_flow")


# --- Figure 3: cost-accuracy frontier ----------------------------------------------------
def fig_frontier() -> None:
    d = read("summary_across_horizons").set_index("policy")
    # The random point is the 50-seed mean at rate 0.5, not a single draw.
    sw = {h: read(f"random_sweep_{h}mo") for h in (12, 24, 36)}
    sw = {h: t[t.rate == 0.5].iloc[0] for h, t in sw.items()}
    d.loc["random_rate0.5", "cost"] = float(np.mean([r.cost_price_like for r in sw.values()]))
    for h, r in sw.items():
        d.loc["random_rate0.5", str(h)] = r.auc_mean
    pts = {  # label: (policy key, kind)
        "cheap": ("cheap_only", "ref"), "fix-M": ("fixed_MRI", "ref"),
        "fix-MF": ("fixed_MRI+FDG", "ref"), "rand": ("random_rate0.5", "ref"),
        "all": ("acquire_all", "ref"),
        "[1]": ("umos_mri_to_amyloid", "pub"),
        "[2]": ("promuse_full_ladder_sharedpred", "pub"),
        "[3]": ("dqn_lambda0.01", "pub"),
        "ours-M": ("ours_MRI", "ours"), "ours-MF": ("ours_MRI+FDG", "ours"),
    }
    # label positions in data coordinates (x = cost, y = AUC), chosen by eye per panel;
    # every label is joined to its point by a short leader line
    txt = {
        "12": {"cheap": (1.6, 0.8135), "fix-M": (2.2, 0.8335), "fix-MF": (0.9, 0.8445),
               "rand": (5.6, 0.8395), "all": (9.5, 0.8665), "[1]": (3.0, 0.8585),
               "[2]": (5.5, 0.8585), "[3]": (3.9, 0.8395), "ours-M": (0.0, 0.8665),
               "ours-MF": (3.0, 0.8775)},
        "24": {"cheap": (1.6, 0.8075), "fix-M": (2.2, 0.857), "fix-MF": (1.7, 0.8725),
               "rand": (7.2, 0.866), "all": (9.5, 0.886), "[1]": (5.4, 0.8955),
               "[2]": (7.2, 0.884), "[3]": (3.1, 0.8625), "ours-M": (-0.2, 0.8935),
               "ours-MF": (1.6, 0.9005)},
        "36": {"cheap": (1.6, 0.7905), "fix-M": (2.2, 0.8445), "fix-MF": (1.9, 0.8955),
               "rand": (7.2, 0.869), "all": (9.5, 0.8915), "[1]": (3.0, 0.8525),
               "[2]": (6.4, 0.8555), "[3]": (6.3, 0.8815), "ours-M": (-0.2, 0.8705),
               "ours-MF": (0.2, 0.8795)},
    }
    ylims = {"12": (0.808, 0.882), "24": (0.800, 0.906), "36": (0.785, 0.906)}
    style = {"ref": (BLUE, "o"), "pub": (ORANGE, "s"), "ours": (GREEN, "^")}
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.75))
    for ax, h in zip(axes, ("12", "24", "36")):
        for lab, (key, kind) in pts.items():
            c, m = style[kind]
            x, y = d.loc[key, "cost"], d.loc[key, h]
            ax.scatter(x, y, color=c, marker=m, s=26, zorder=3, edgecolor="white", lw=0.5)
            tx, ty = txt[h][lab]
            ax.annotate(lab, (x, y), xytext=(tx, ty), textcoords="data", fontsize=7.2,
                        ha="center", va="center",
                        arrowprops=dict(arrowstyle="-", color=GREY, lw=0.5, shrinkA=2, shrinkB=3))
        ax.axhline(d.loc["acquire_all", h], color=GREY, ls="--", lw=1)
        ax.set_title(f"{h}-month horizon", fontsize=9)
        ax.set_xlim(-1.5, 11.0)
        ax.set_ylim(*ylims[h])
        ax.set_xlabel("Cost (relative units)", fontsize=8.5)
        ax.tick_params(labelsize=7.8)
    axes[0].set_ylabel("AUC (cross-validated)")
    handles = [
        Line2D([0], [0], marker="o", color="w", mfc=BLUE, ms=7, label="Reference baselines"),
        Line2D([0], [0], marker="s", color="w", mfc=ORANGE, ms=7, label="Published adaptive policies"),
        Line2D([0], [0], marker="^", color="w", mfc=GREEN, ms=7, label="SPFA (ours)"),
        Line2D([0], [0], color=GREY, ls="--", lw=1.2, label="acquire-all reference"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.02), handlelength=1.6, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=0.6)
    save(fig, "fig1_cost_accuracy_frontier")


# --- Figure: cost-matched random-acquisition frontier -------------------------------------
def fig_random_sweep() -> None:
    kinds = {  # policy key -> (marker colour, marker, label)
        "fixed_MRI": (BLUE, "o", "fixed rules"), "fixed_MRI+FDG": (BLUE, "o", None),
        "umos_mri_to_amyloid": (ORANGE, "s", "published adaptive policies"),
        "promuse_full_ladder_sharedpred": (ORANGE, "s", None),
        "dqn_lambda0.01": (ORANGE, "s", None), "dqn_lambda0.05": (ORANGE, "s", None),
        "ours_MRI": (GREEN, "^", "SPFA (ours)"), "ours_MRI+FDG": (GREEN, "^", None),
    }
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.75))
    for ax, h in zip(axes, (12, 24, 36)):
        sw = read(f"random_sweep_{h}mo")
        m = read(f"random_matched_{h}mo")
        m = m[m.regime == "price_like"].set_index("policy")
        ax.fill_between(sw.cost_price_like, sw.auc_ci_lo, sw.auc_ci_hi, color=GREY, alpha=0.18,
                        lw=0, label="random, 95% CI (subjects)")
        ax.plot(sw.cost_price_like, sw.auc_mean, color=GREY, lw=1.4, label="random, mean of 50 seeds")
        for key, (c, mk, lab) in kinds.items():
            ax.scatter(m.loc[key, "policy_cost"], m.loc[key, "policy_auc"], color=c, marker=mk, s=26,
                       zorder=3, edgecolor="white", lw=0.5, label=lab)
        ax.set_title(f"{h}-month horizon", fontsize=9)
        ax.set_xlabel("Cost (relative price-like units)", fontsize=8.5)
        ax.tick_params(labelsize=7.8)
    axes[0].set_ylabel("AUC (cross-validated)")
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=3, frameon=False, fontsize=7.6,
               bbox_to_anchor=(0.5, -0.04), handlelength=1.6, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=0.6)
    save(fig, "fig_random_sweep")


# --- panels ------------------------------------------------------------------------------
def _p_acquisition(ax):
    g = read("policy_benchmark_24mo").set_index("policy")[["acq_MRI", "acq_FDG", "acq_AMYLOID", "acq_CSF"]]
    order = ["fixed_MRI", "fixed_MRI+FDG", "umos_mri_to_amyloid", "dqn_lambda0.05", "dqn_lambda0.01",
             "promuse_full_ladder_sharedpred", "random_rate0.5", "acquire_all"]
    g = g.loc[order]
    left = np.zeros(len(g))
    for col, c, n in zip(g.columns, (BLUE, ORANGE, RED, GREEN), ("MRI", "FDG", "amyloid", "CSF")):
        ax.barh(range(len(g)), g[col], left=left, color=c, label=n, height=0.62)
        left += g[col].to_numpy()
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels([LABEL[p] for p in order])
    ax.invert_yaxis()
    ax.set_xlabel("Modalities acquired per patient")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.42, -0.17), frameon=False,
              handlelength=1.2, columnspacing=1.0)
    _panel_title(ax, "a", "Modalities ordered (24 months)")


def _p_domination(ax):
    d = read("domination_by_regime")
    regs = [("price_like", "price-like", BLUE, ""), ("flat", "flat", ORANGE, "//"),
            ("burden", "burden", GREEN, "xx")]
    hs = [12, 24, 36]
    w = 0.26
    for k, (reg, lab, c, hatch) in enumerate(regs):
        n = [int(d[d.horizon == h][f"dominated_{reg}"].sum()) for h in hs]
        bars = ax.bar(np.arange(3) + (k - 1) * w, n, w, color=c, hatch=hatch, edgecolor="white", label=lab)
        for b_, v in zip(bars, n):
            ax.text(b_.get_x() + b_.get_width() / 2, v + 0.12, str(v), ha="center", va="bottom", fontsize=7.6)
    ax.set_ylim(0, 9.4)
    ax.set_yticks(range(0, 10, 3))
    ax.set_xticks(range(3))
    ax.set_xticklabels(["12 months", "24 months", "36 months"])
    ax.set_ylabel("Configurations dominated (of 9)")
    ax.legend(title="Cost regime", loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=7.4,
              title_fontsize=7.4, frameon=False, handlelength=1.3, columnspacing=0.9)
    _panel_title(ax, "b", "Dominated by a fixed rule")


def fig4() -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 3.0), gridspec_kw={"width_ratios": [1.25, 1]})
    _p_acquisition(a)
    _p_domination(b)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig4_acquisition_domination")


def _p_seed(ax):
    d = read("robustness_seed_24mo")
    order = ["cheap_only", "fixed_MRI", "fixed_MRI+FDG", "umos_mri_to_amyloid",
             "dqn_lambda0.05", "promuse_full_ladder_sharedpred", "random_rate0.5", "acquire_all"]
    for i, p in enumerate(order):
        s = d[d.policy == p].auc
        ax.scatter(s, [i] * len(s), color=BLUE, s=16, zorder=3, alpha=0.85)
        ax.errorbar(s.mean(), i, xerr=s.std(ddof=1), color=ORANGE, capsize=3, lw=1.5, zorder=4)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([LABEL[p] for p in order])
    ax.invert_yaxis()
    ax.set_xlabel("AUC (dots: seeds 0, 7, 42; bar: mean ± SD)")
    _panel_title(ax, "a", "Seed robustness")


def _p_predictor(ax):
    # Same cohort, seed and MRI block as the gradient-boosting run.
    s = read("robustness_seed_24mo")
    lr = s[s.seed == 42].set_index("policy")
    gb = read("robustness_predictor_24mo").set_index("policy")
    pols = [p for p in ["cheap_only", "fixed_MRI", "fixed_MRI+FDG", "umos_mri_to_amyloid",
                        "dqn_lambda0.05", "promuse_full_ladder_sharedpred", "random_rate0.5",
                        "acquire_all"] if p in lr.index and p in gb.index]
    short = {"umos_mri_to_amyloid": "cascade [1]", "promuse_full_ladder_sharedpred": "evidential [2]",
             "dqn_lambda0.05": "RL [3], w = 0.05"}
    x = np.arange(len(pols))
    ax.bar(x - 0.2, lr.loc[pols, "auc"], 0.4, color=BLUE, label="logistic regression")
    ax.bar(x + 0.2, gb.loc[pols, "auc"], 0.4, color=ORANGE, label="gradient boosting", hatch="//",
           edgecolor="white")
    ax.set_ylim(0.75, 0.93)
    ax.set_xticks(x)
    ax.set_xticklabels([short.get(p, LABEL[p]) for p in pols], rotation=40, ha="right")
    ax.set_ylabel("AUC")
    ax.legend(loc="upper left", ncol=1)
    _panel_title(ax, "b", "Predictor swap (same 366 subjects)")


def fig5() -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 3.05), gridspec_kw={"width_ratios": [1.05, 1]})
    _p_seed(a)
    _p_predictor(b)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig5_robustness")


def _p_dca(ax):
    d = read("decision_curve_24mo")
    s0 = d[d.policy == "cheap_only"]
    ax.plot(s0.threshold, s0.treat_all, color="k", lw=1.0, ls=":", label="treat all")
    ax.axhline(0, color=GREY, lw=1, label="treat none")
    show = {"ours_MRI+FDG": (GREEN, "-", "^"), "fixed_MRI+FDG": (BLUE, "--", "o"),
            "random_rate0.5": (ORANGE, "-.", "s"), "acquire_all": (RED, ":", "D"),
            "umos_mri_to_amyloid": (PURPLE, (0, (5, 1)), "v"),
            "promuse_full_ladder_sharedpred": ("#56B4E9", (0, (1, 1)), "P")}
    for p, (c, ls, m) in show.items():
        s = d[d.policy == p]
        ax.plot(s.threshold, s.model, color=c, lw=1.3, ls=ls, marker=m, ms=3.5, label=LABEL[p])
    ax.set_ylim(-0.03, 0.17)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=7.0, handlelength=2.0,
              columnspacing=0.8, frameon=False)
    _panel_title(ax, "a", "Decision-curve analysis")


def _p_calibration(ax):
    b = read("calibration_bins_24mo")
    show = {"ours_MRI+FDG": (GREEN, "^"), "fixed_MRI+FDG": (BLUE, "o"),
            "random_rate0.5": (ORANGE, "s"), "acquire_all": (RED, "D")}
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for p, (c, m) in show.items():
        s = b[b.policy == p]
        ax.plot(s.mean_predicted, s.observed, "-", marker=m, color=c, ms=4, lw=1.2, label=LABEL[p])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed conversion rate")
    ax.legend(loc="upper left", fontsize=7.0)
    _panel_title(ax, "b", "Reliability (development CV)")


def fig6() -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 3.3), gridspec_kw={"width_ratios": [1.25, 1]})
    _p_dca(a)
    _p_calibration(b)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig6_clinical_utility")


def _p_ablation(ax):
    d = read("ablation_24mo")
    styles = {("minimal", False): (GREY, "o", "3-feature MRI, complete-case"),
              ("minimal", True): (BLUE, "s", "3-feature MRI, pooled"),
              ("signature", False): (ORANGE, "^", "signature MRI, complete-case"),
              ("signature", True): (GREEN, "D", "signature MRI, pooled (ours)")}
    for k, ((feat, pooled), (c, m, lab)) in enumerate(styles.items()):
        s = d[(d.mri_features == feat) & (d.pooled == pooled)].sort_values("cost")
        ax.errorbar(s.cost + (k - 1.5) * 0.16, s.auc, yerr=[s.auc - s.lo, s.hi - s.auc], color=c,
                    marker=m, ms=4, lw=1.1, capsize=2, label=lab, alpha=0.95)
    ax.set_xlabel("Acquisition cost (price-like units)")
    ax.set_ylabel("Cross-validated AUC")
    ax.legend(loc="lower right", fontsize=7.2)
    _panel_title(ax, "a", "AUC by acquisition state")


def _p_attribution(ax):
    d = read("ablation_attribution_24mo")
    d = d[d.state.isin(["CHEAP+MRI", "CHEAP+MRI+FDG", "CHEAP+MRI+FDG+AMYLOID"])]
    states = list(d.state.unique())
    changes = ["rich MRI only", "pooling only", "both (ours)"]
    for i, st in enumerate(states):
        for j, (ch, c, m) in enumerate(zip(changes, (ORANGE, BLUE, GREEN), ("^", "s", "D"))):
            r = d[(d.state == st) & (d.change == ch)].iloc[0]
            ax.errorbar(r.delta_auc, i + (j - 1) * 0.25, xerr=[[r.delta_auc - r.lo], [r.hi - r.delta_auc]],
                        fmt=m, color=c, capsize=2.5, ms=4, label=ch if i == 0 else None)
    ax.axvline(0, color="k", lw=0.9)
    ax.set_yticks(range(len(states)))
    ax.set_yticklabels([s.replace("CHEAP+", "").replace("+AMYLOID", "+\namyloid") for s in states])
    ax.invert_yaxis()
    ax.set_xlabel("Change in AUC vs. 3-feature reference (95% CI)")
    ax.legend(loc="lower right", fontsize=7.4)
    _panel_title(ax, "b", "Attribution of the gain")


def fig7() -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 3.0), gridspec_kw={"width_ratios": [1.1, 1]})
    _p_ablation(a)
    _p_attribution(b)
    fig.tight_layout(w_pad=1.3)
    save(fig, "fig7_ablation")


# --- Figure 8: held-out test, one grouped forest plot ----------------------------------
def fig_final_test() -> None:
    order = ["cheap_only", "ours_MRI", "ours_MRI+FDG", "fixed_MRI+FDG", "acquire_all"]
    cost = {"cheap_only": "0", "ours_MRI": "1", "ours_MRI+FDG": "3", "fixed_MRI+FDG": "3",
            "acquire_all": "9.5"}
    hz = [(12, BLUE, "o", -0.25), (24, ORANGE, "s", 0.0), (36, GREEN, "D", 0.25)]
    fig, ax = plt.subplots(figsize=(W, 3.5))
    for h, c, m, off in hz:
        d = read(f"FINAL_TEST_{h}mo").set_index("model").loc[order]
        for i, (mname, r) in enumerate(d.iterrows()):
            ax.errorbar(r.auc, i + off, xerr=[[r.auc - r.lo], [r.hi - r.auc]], fmt=m, color=c,
                        capsize=2.5, ms=4.5, lw=1.2, label=f"{h} months" if i == 0 else None)
            lab = Decimal(str(r.auc)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
            ax.text(r.hi + 0.005, i + off, f"{lab}", va="center", fontsize=7.4, color=c)
    for i in range(len(order) - 1):
        ax.axhline(i + 0.5, color="#DDDDDD", lw=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{LABEL[m]} (cost {cost[m]})" for m in order])
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlim(0.65, 1.04)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Held-out test AUC (95% CI)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8, frameon=False,
              title="Prediction horizon", title_fontsize=8)
    save(fig, "fig_final_test_forest")


# --- Figure 9: representation attrition -------------------------------------------------
def _p_attrition(ax):
    d = read("representation_attrition_24mo")
    labels = ["MCI at\nbaseline", "+ 24-month\noutcome label", "+ all four\nmodalities"]
    x = np.arange(3)
    ax.bar(x, d.n, 0.62, color=BLUE, alpha=0.9, label="cohort size (n)")
    for xi, n in zip(x, d.n):
        ax.text(xi, 40, f"n = {n:,}", ha="center", va="bottom", color="white", fontweight="bold",
                fontsize=7.4)
    ax.set_ylim(0, 2050)
    ax.set_ylabel("Participants (n)", color=BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.tick_params(axis="y", colors=BLUE)
    ax2 = ax.twinx()
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax2.plot(x, d.pct_non_white, "-s", color=ORANGE, lw=2, ms=6, label="non-White (%)")
    for xi, pct, nn in zip(x, d.pct_non_white, d.n_non_white):
        ax2.annotate(f"{pct:.1f}%\n(n = {nn})", (xi, pct), xytext=(0, 9), textcoords="offset points",
                     ha="center", va="bottom", color=ORANGE_TXT, fontweight="bold", fontsize=7.6,
                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax2.set_ylim(0, 26)
    ax2.set_ylabel("Non-White (%)", color=ORANGE_TXT)
    ax2.tick_params(axis="y", colors=ORANGE_TXT)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7.4)
    _panel_title(ax, "a", "Attrition (24-month horizon)")


def _p_race(ax):
    d = read("fairness_acquisition_24mo")
    s = d[(d.dimension == "RACE") & (d.policy == "cheap_only")].sort_values("n", ascending=True)
    names = {"American Indian or Alaskan Native": "American Indian or\nAlaska Native",
             "Native Hawaiian or Other Pacific Islander": "Native Hawaiian or\nPacific Islander",
             "Black or African American": "Black or African\nAmerican",
             "More than one race": "More than\none race"}
    ax.barh(range(len(s)), s.n, color=[GREY if u else BLUE for u in s.underpowered], height=0.6)
    ax.set_xlim(0, 420)
    for i, n in enumerate(s.n):
        ax.text(n + 6, i, str(int(n)), va="center", fontsize=8)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels([names.get(g, g) for g in s.group], fontsize=7.8)
    ax.set_xlabel("Subjects (grey: underpowered)")
    _panel_title(ax, "b", "Race in the audit cohort")


def fig9() -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.95), gridspec_kw={"width_ratios": [1.1, 1]})
    _p_attrition(a)
    _p_race(b)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig9_attrition")


# --- Figure 10: acquisition by sex --------------------------------------------------------
def fig_sex_acquisition() -> None:
    d = read("fairness_acquisition_24mo")
    d = d[d.dimension == "SEX"]
    pols = ["umos_mri_to_amyloid", "promuse_full_ladder_sharedpred", "dqn_lambda0.05"]
    titles = ["uncertainty cascade [1]", "staged evidential [2]", "RL test-ordering [3], w = 0.05"]
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.6), sharey=True)
    for ax, p, ttl in zip(axes, pols, titles):
        s = d[d.policy == p].set_index("group")
        mods = ["acq_MRI", "acq_FDG", "acq_AMYLOID", "acq_CSF"]
        x = np.arange(len(mods))
        ax.bar(x - 0.2, s.loc["Male", mods], 0.4, color=BLUE, label="male")
        ax.bar(x + 0.2, s.loc["Female", mods], 0.4, color=ORANGE, label="female", hatch="//",
               edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(["MRI", "FDG", "amyloid", "CSF"], fontsize=8)
        ax.set_title(ttl, fontsize=9)
    axes[0].set_ylabel("Acquisition rate")
    axes[0].legend(loc="upper right", fontsize=7.6)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_sex_acquisition")


# --- Figure 11: horizon dependence ----------------------------------------------------------
def _p_horizon(ax):
    d = read("horizon_effect")
    d = d[~d.underpowered]
    cols = {"MRI": (BLUE, "o"), "FDG": (ORANGE, "s"), "AMYLOID": (RED, "^"), "CSF": (GREEN, "D")}
    for k, (blk, (c, m)) in enumerate(cols.items()):
        s = d[d.block == blk].sort_values("horizon")
        ax.errorbar(s.horizon.map({12: 0, 24: 1, 36: 2}).to_numpy() + (k - 1.5) * 0.12,
                    s.delta_auc, yerr=[s.delta_auc - s.delta_lo, s.delta_hi - s.delta_auc],
                    marker=m, color=c, capsize=2.5, lw=1.1, ms=4,
                    label={"AMYLOID": "amyloid"}.get(blk, blk))
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["12 mo", "24 mo", "36 mo"])
    ax.set_ylabel("Incremental AUC over cheap-only")
    ax.set_ylim(-0.02, 0.14)
    ax.legend(ncol=4, loc="upper center", fontsize=7.2, handlelength=1.4, columnspacing=0.8)
    _panel_title(ax, "a", "Fixed subject set")


def _p_modality(ax):
    d = read("modality_value")
    d = d[d.delta_auc.notna() & ~d.underpowered].copy()
    d["blk"] = d["subset"].str.replace("matched_", "", regex=False).str.replace(
        "MRI+FDG+AMYLOID+CSF", "all four", regex=False)
    blks = ["MRI", "FDG", "AMYLOID", "CSF", "all four"]
    w = 0.26
    for k, (h, c, hatch) in enumerate(zip((12, 24, 36), (BLUE, ORANGE, GREEN), ("", "//", "xx"))):
        s = d[d.horizon == h].set_index("blk").reindex(blks)
        ax.bar(np.arange(len(blks)) + (k - 1) * w, s.delta_auc, w, color=c, label=f"{h} mo",
               hatch=hatch, edgecolor="white",
               yerr=[(s.delta_auc - s.delta_lo).to_numpy(), (s.delta_hi - s.delta_auc).to_numpy()],
               capsize=1.8, error_kw={"lw": 0.8})
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks(range(len(blks)))
    ax.set_xticklabels(["MRI", "FDG", "amyloid", "CSF", "all four"], fontsize=8)
    ax.legend(loc="upper left", fontsize=7.4)
    _panel_title(ax, "b", "Matched subsets")


def fig11() -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.9), sharey=False)
    _p_horizon(a)
    _p_modality(b)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig11_horizon")


def main() -> None:
    for fn in (fig_pipeline, fig_algorithm_flow, fig_frontier, fig4, fig5, fig6, fig7,
               fig_final_test, fig9, fig_sex_acquisition, fig11, fig_random_sweep):
        fn()


if __name__ == "__main__":
    main()
