"""
Fairness audit of ACQUISITION decisions, not just of predictions.

This is the project's primary novelty claim, so it is worth being precise
about what is new. The existing AD fairness literature (e.g. Algorithmic
Fairness of ML Models for Alzheimer Disease Progression, JAMA Network Open
2023; FR-CCA 2025) audits whether a model PREDICTS equally well across
demographic groups. None of the acquisition policies audited here -- UMoS,
ProMUSE, the 2025 DQN -- reports whether the policy ORDERS TESTS equally
across groups.

Those are different questions. A policy can predict with equal AUC in every
subgroup while systematically sending one group for fewer scans, which in a
deployed system means unequal access to diagnostic workup. So two families
of metric are reported:

  ACQUISITION fairness -- per-subgroup rate of being sent for each modality,
      and the max gap between subgroups. This is the new part.
  OUTCOME fairness -- per-subgroup AUC / TPR / TNR / calibration, and
      whether accuracy is being bought for one group at another's expense.

A note on power, which matters more here than anywhere else in the project:
ADNI is documented to oversample White, non-Hispanic participants, so
minority subgroups are small (tens of subjects, single-digit converters).
Every table therefore carries an explicit `underpowered` flag, and gaps are
computed both across all subgroups and across adequately-powered ones only.
An unflagged gap computed against a subgroup of six people is not evidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.metrics import auc, sensitivity_at, specificity_at

MIN_SUBGROUP_N = 30
MIN_SUBGROUP_POS = 10


def collapse_race(series: pd.Series) -> pd.Series:
    """Collapse ADNI's PTRACCAT into analysis categories.

    ADNI records multi-select race as pipe-delimited combinations
    ("Black or African American|White"), producing ~20 raw categories, most
    with single-digit counts. Multi-select responses are mapped to
    "More than one race" rather than being assigned to one component, which
    would silently reclassify people.
    """
    def one(v):
        if not isinstance(v, str):
            return "Unknown"
        if "|" in v:
            return "More than one race"
        return v

    return series.map(one).fillna("Unknown")


def subgroup_acquisition(
    groups: pd.Series, acquired: list[list[str]], blocks: list[str]
) -> pd.DataFrame:
    """Per-subgroup rate of being sent for each modality block."""
    df = pd.DataFrame({"g": groups.to_numpy()})
    for b in blocks:
        df[b] = [b in blk for blk in acquired]
    df["n_blocks"] = [len(blk) for blk in acquired]

    rows = []
    for g, sub in df.groupby("g", dropna=False):
        row = {"group": g, "n": len(sub), "underpowered": len(sub) < MIN_SUBGROUP_N}
        for b in blocks:
            row[f"acq_{b}"] = round(float(sub[b].mean()), 4)
        row["mean_n_blocks"] = round(float(sub["n_blocks"].mean()), 4)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def subgroup_outcomes(
    groups: pd.Series, y: np.ndarray, p: np.ndarray, threshold: float = 0.5
) -> pd.DataFrame:
    """Per-subgroup discrimination, error rates and calibration-in-the-large."""
    df = pd.DataFrame({"g": groups.to_numpy(), "y": np.asarray(y).astype(int), "p": np.asarray(p)})
    rows = []
    for g, sub in df.groupby("g", dropna=False):
        n_pos = int(sub["y"].sum())
        rows.append(
            dict(
                group=g,
                n=len(sub),
                n_pos=n_pos,
                underpowered=(len(sub) < MIN_SUBGROUP_N) or (n_pos < MIN_SUBGROUP_POS),
                auc=round(auc(sub["y"], sub["p"]), 4),
                tpr=round(sensitivity_at(sub["y"], sub["p"], threshold), 4),
                tnr=round(specificity_at(sub["y"], sub["p"], threshold), 4),
                mean_predicted=round(float(sub["p"].mean()), 4),
                observed=round(float(sub["y"].mean()), 4),
            )
        )
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def gaps(table: pd.DataFrame, cols: list[str], powered_only: bool = True) -> dict:
    """Max pairwise gap per column, optionally restricted to powered subgroups."""
    t = table[~table["underpowered"]] if powered_only else table
    out = {"n_groups_used": int(len(t))}
    for c in cols:
        vals = pd.to_numeric(t[c], errors="coerce").dropna()
        out[f"{c}_gap"] = round(float(vals.max() - vals.min()), 4) if len(vals) >= 2 else np.nan
    return out


def bootstrap_gap_ci(
    groups: pd.Series,
    indicator: np.ndarray,
    group_a: str,
    group_b: str,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI for rate(group_a) - rate(group_b) on a binary indicator.

    A raw acquisition-rate difference is a point estimate on subgroups of a
    few hundred people; without an interval there is no way to tell a real
    disparity from sampling noise. Subjects are resampled within each group
    so the group sizes stay fixed, which is the relevant sampling model here
    (the cohort composition is not itself random).

    Returns (gap, lo, hi) where gap = rate(a) - rate(b).
    """
    rng = np.random.default_rng(seed)
    g = groups.to_numpy()
    ind = np.asarray(indicator, dtype=float)
    a = ind[g == group_a]
    b = ind[g == group_b]
    if len(a) == 0 or len(b) == 0:
        return (float("nan"),) * 3

    gap = float(a.mean() - b.mean())
    vals = []
    for _ in range(n_boot):
        sa = a[rng.integers(0, len(a), len(a))]
        sb = b[rng.integers(0, len(b), len(b))]
        vals.append(sa.mean() - sb.mean())
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (gap, float(lo), float(hi))


def acquisition_disparity(
    groups: pd.Series,
    acquired: list[list[str]],
    blocks: list[str],
    group_a: str,
    group_b: str,
    seed: int = 0,
) -> pd.DataFrame:
    """Signed, CI'd acquisition-rate disparity between two subgroups.

    Direction matters and is easy to lose: reporting only |gap| hides WHO is
    being sent for fewer tests, which is the entire point of the audit.
    """
    rows = []
    for b in blocks:
        ind = np.array([b in blk for blk in acquired], dtype=float)
        gap, lo, hi = bootstrap_gap_ci(groups, ind, group_a, group_b, seed=seed)
        pv = bootstrap_gap_pvalue(groups, ind, group_a, group_b, seed=seed)
        rows.append(
            dict(
                block=b,
                group_a=group_a,
                group_b=group_b,
                rate_a=round(float(ind[groups.to_numpy() == group_a].mean()), 4),
                rate_b=round(float(ind[groups.to_numpy() == group_b].mean()), 4),
                gap=round(gap, 4),
                lo=round(lo, 4),
                hi=round(hi, 4),
                p_value=round(pv, 4),
                significant_uncorrected=bool(lo > 0 or hi < 0),
            )
        )
    n_ind = np.array([len(blk) for blk in acquired], dtype=float)
    gap, lo, hi = bootstrap_gap_ci(groups, n_ind, group_a, group_b, seed=seed)
    pv = bootstrap_gap_pvalue(groups, n_ind, group_a, group_b, seed=seed)
    rows.append(
        dict(
            block="TOTAL_TESTS", group_a=group_a, group_b=group_b,
            rate_a=round(float(n_ind[groups.to_numpy() == group_a].mean()), 4),
            rate_b=round(float(n_ind[groups.to_numpy() == group_b].mean()), 4),
            gap=round(gap, 4), lo=round(lo, 4), hi=round(hi, 4),
            p_value=round(pv, 4),
            significant_uncorrected=bool(lo > 0 or hi < 0),
        )
    )
    return pd.DataFrame(rows)


def bootstrap_gap_pvalue(
    groups: pd.Series,
    indicator: np.ndarray,
    group_a: str,
    group_b: str,
    n_boot: int = 2000,
    seed: int = 0,
) -> float:
    """Two-sided bootstrap p-value for rate(a) - rate(b) != 0."""
    rng = np.random.default_rng(seed)
    g = groups.to_numpy()
    ind = np.asarray(indicator, dtype=float)
    a = ind[g == group_a]
    b = ind[g == group_b]
    if len(a) == 0 or len(b) == 0:
        return float("nan")

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sa = a[rng.integers(0, len(a), len(a))]
        sb = b[rng.integers(0, len(b), len(b))]
        diffs[i] = sa.mean() - sb.mean()
    # Fraction of the bootstrap distribution on each side of zero.
    p_low = float((diffs <= 0).mean())
    p_high = float((diffs >= 0).mean())
    return float(min(1.0, 2.0 * min(p_low, p_high)))


def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction; returns a boolean reject array.

    The fairness audit tests every (policy, modality) combination, which is
    dozens of comparisons. At alpha=0.05 a couple of "significant" gaps are
    expected from noise alone, so an uncorrected finding here is not
    evidence of disparity. Controlling FDR is the minimum defensible bar.
    """
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    reject = np.zeros(len(p), dtype=bool)
    if ok.sum() == 0:
        return reject

    idx = np.flatnonzero(ok)
    order = idx[np.argsort(p[idx])]
    m = len(order)
    thresholds = alpha * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresholds
    if passed.any():
        kmax = np.flatnonzero(passed)[-1]
        reject[order[: kmax + 1]] = True
    return reject
