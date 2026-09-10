"""
Evaluation metrics for the acquisition-policy experiments.

Deliberately goes beyond AUC. The AD prognosis literature reports AUC almost
exclusively, which says nothing about whether predicted probabilities are
trustworthy (calibration) or whether using the model beats a default strategy
(net benefit). Both are reported here from the start.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score


def auc(y_true, y_prob) -> float:
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def brier(y_true, y_prob) -> float:
    return float(brier_score_loss(np.asarray(y_true), np.asarray(y_prob)))


def bootstrap_ci(
    y_true,
    y_prob,
    metric_fn=auc,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI, resampling SUBJECTS (rows are one per subject here)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = metric_fn(y_true[idx], y_prob[idx])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def bootstrap_delta_ci(
    y_true,
    y_prob_a,
    y_prob_b,
    metric_fn=auc,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired bootstrap CI for metric(b) - metric(a) on the same subjects.

    Paired resampling is the point: comparing two models on the same subjects
    is far more sensitive than comparing two independent CIs, and it is the
    right test for "does adding this modality help".

    Returns (delta, lo, hi).
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    a = np.asarray(y_prob_a)
    b = np.asarray(y_prob_b)
    n = len(y_true)
    point = metric_fn(y_true, b) - metric_fn(y_true, a)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        va = metric_fn(y_true[idx], a[idx])
        vb = metric_fn(y_true[idx], b[idx])
        if not (np.isnan(va) or np.isnan(vb)):
            deltas.append(vb - va)
    if not deltas:
        return (point, float("nan"), float("nan"))
    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(point), float(lo), float(hi))


def calibration_table(y_true, y_prob, n_bins: int = 5) -> pd.DataFrame:
    """Reliability table: predicted vs observed risk per bin."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append(
            dict(
                bin=f"{bins[b]:.2f}-{bins[b+1]:.2f}",
                n=int(m.sum()),
                mean_predicted=float(y_prob[m].mean()),
                observed=float(y_true[m].mean()),
            )
        )
    return pd.DataFrame(rows)


def net_benefit(y_true, y_prob, threshold: float) -> float:
    """Decision-curve net benefit at one threshold probability.

    NB = TP/n - FP/n * (pt / (1 - pt)). Answers "does acting on this model at
    this risk threshold beat treating nobody", in units of true positives.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    if n == 0:
        return float("nan")
    pred_pos = y_prob >= threshold
    tp = int(((pred_pos == 1) & (y_true == 1)).sum())
    fp = int(((pred_pos == 1) & (y_true == 0)).sum())
    w = threshold / (1 - threshold) if threshold < 1 else np.inf
    return float(tp / n - (fp / n) * w)


def decision_curve(y_true, y_prob, thresholds=None) -> pd.DataFrame:
    """Net benefit of the model vs treat-all vs treat-none across thresholds."""
    if thresholds is None:
        thresholds = np.arange(0.05, 0.55, 0.05)
    y_true = np.asarray(y_true).astype(int)
    prevalence = y_true.mean() if len(y_true) else float("nan")
    rows = []
    for t in thresholds:
        w = t / (1 - t)
        rows.append(
            dict(
                threshold=round(float(t), 3),
                model=net_benefit(y_true, y_prob, t),
                treat_all=float(prevalence - (1 - prevalence) * w),
                treat_none=0.0,
            )
        )
    return pd.DataFrame(rows)


def sensitivity_at(y_true, y_prob, threshold: float = 0.5) -> float:
    y_true = np.asarray(y_true).astype(int)
    pos = y_true == 1
    if pos.sum() == 0:
        return float("nan")
    return float((np.asarray(y_prob)[pos] >= threshold).mean())


def specificity_at(y_true, y_prob, threshold: float = 0.5) -> float:
    y_true = np.asarray(y_true).astype(int)
    neg = y_true == 0
    if neg.sum() == 0:
        return float("nan")
    return float((np.asarray(y_prob)[neg] < threshold).mean())


def subgroup_metrics(
    y_true,
    y_prob,
    groups,
    threshold: float = 0.5,
    min_n: int = 20,
) -> pd.DataFrame:
    """Per-subgroup AUC / sensitivity / specificity / calibration-in-the-large.

    Subgroups smaller than `min_n` are still reported but flagged, rather than
    dropped: silently omitting small subgroups is how disparities get missed,
    but pretending an AUC from n=6 is meaningful is equally misleading.
    """
    df = pd.DataFrame(
        {"y": np.asarray(y_true).astype(int), "p": np.asarray(y_prob), "g": np.asarray(groups)}
    )
    rows = []
    for g, sub in df.groupby("g", dropna=False):
        rows.append(
            dict(
                group=g,
                n=len(sub),
                n_pos=int(sub["y"].sum()),
                underpowered=len(sub) < min_n,
                auc=auc(sub["y"], sub["p"]),
                sensitivity=sensitivity_at(sub["y"], sub["p"], threshold),
                specificity=specificity_at(sub["y"], sub["p"], threshold),
                mean_predicted=float(sub["p"].mean()),
                observed=float(sub["y"].mean()),
            )
        )
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def equalized_odds_gap(subgroup_df: pd.DataFrame) -> dict[str, float]:
    """Max pairwise gap in sensitivity (TPR) and specificity across subgroups.

    Restricted to adequately-powered subgroups, since a gap computed against
    an n=6 subgroup's noisy TPR is not evidence of unfairness.
    """
    ok = subgroup_df[~subgroup_df["underpowered"]]
    out = {}
    for col, name in (("sensitivity", "tpr_gap"), ("specificity", "tnr_gap")):
        vals = ok[col].dropna()
        out[name] = float(vals.max() - vals.min()) if len(vals) >= 2 else float("nan")
    return out


def bootstrap_interaction_ci(
    y_a,
    base_a,
    blk_a,
    y_b,
    base_b,
    blk_b,
    metric_fn=auc,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired bootstrap CI for a DIFFERENCE OF DIFFERENCES.

    Tests whether a block's incremental value differs between two settings
    (here: two prediction horizons) on the SAME subjects:

        [metric(blk_b) - metric(base_b)] - [metric(blk_a) - metric(base_a)]

    This is the test that licenses a claim like "this modality is worth more
    at 36 months than at 12". Reporting two separate per-horizon CIs and
    eyeballing that one is bigger is not that test -- overlapping CIs can
    still hide a significant paired difference, and non-overlapping ones can
    fail to be significant.

    Subjects are resampled once per iteration and applied to both settings,
    preserving the pairing. Predictions are held fixed (fitted once, outside)
    rather than refitted per iteration, which is standard for comparing
    metric differences and keeps this tractable.

    Returns (point_estimate, lo, hi).
    """
    rng = np.random.default_rng(seed)
    y_a, base_a, blk_a = map(np.asarray, (y_a, base_a, blk_a))
    y_b, base_b, blk_b = map(np.asarray, (y_b, base_b, blk_b))
    n = len(y_a)
    if len(y_b) != n:
        raise ValueError("interaction test requires the same subjects in both settings")

    point = (metric_fn(y_b, blk_b) - metric_fn(y_b, base_b)) - (
        metric_fn(y_a, blk_a) - metric_fn(y_a, base_a)
    )

    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        da = metric_fn(y_a[idx], blk_a[idx]) - metric_fn(y_a[idx], base_a[idx])
        db = metric_fn(y_b[idx], blk_b[idx]) - metric_fn(y_b[idx], base_b[idx])
        if not (np.isnan(da) or np.isnan(db)):
            vals.append(db - da)
    if not vals:
        return (float(point), float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(point), float(lo), float(hi))
