"""
Experiment 2: does a modality's incremental value depend on the PREDICTION
HORIZON, holding the subjects fixed?

This is the project's second novelty claim, so it gets its own experiment
with a stricter design than Experiment 1. In run_modality_value.py the
matched subset differs per horizon (subjects labelled at 12 months are not
the same people as those labelled at 36), so a trend across horizons there
confounds "the horizon changed" with "the sample changed".

Here the subject set is FIXED: only subjects labelled at ALL horizons AND
holding the block in question. The horizon is then the only thing varying,
which is what makes a trend interpretable.

Reference point in the literature: the 2026 Science Advances PREVENT-AD
analysis found biomarker effects that shift with follow-up time (short-term
vs long-term predictors differ). That was a different cohort, with Cox
models and no acquisition decision. Replicating the direction of that
effect on ADNI, per modality, is what licenses a horizon-CONDITIONED
acquisition policy rather than a single fixed one.

Run:  python -m src.eval.run_horizon_effect
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.metrics import auc, bootstrap_delta_ci, bootstrap_interaction_ci
from src.eval.run_modality_value import RANDOM_STATE, _oof_predictions
from src.models.features import ACQUIRABLE_BLOCKS, subjects_with_blocks
from src.preprocessing.cohort import HORIZONS, build_cohort
from src.preprocessing.leakage_checks import assert_no_subject_leakage
from src.preprocessing.load_adni import build_subject_visits
from src.preprocessing.splits import subject_level_split

MIN_SUBJECTS = 80
MIN_POSITIVES = 25


def main() -> None:
    print("Loading real ADNI cohort...")
    cohort = build_cohort(build_subject_visits())

    label_cols = [f"LABEL_{h}mo" for h in HORIZONS]

    # Fixed subject set: labelled at EVERY horizon.
    complete = cohort[cohort[label_cols].notna().all(axis=1)].copy()
    print(f"  {len(cohort)} MCI subjects; {len(complete)} labelled at all "
          f"{len(HORIZONS)} horizons (the fixed set used below)")

    # Hold out a test set once from this fixed set; everything below is CV on
    # the remainder, so the test set stays untouched here too.
    split = subject_level_split(complete, label_col=label_cols[-1], random_state=RANDOM_STATE)
    assert_no_subject_leakage(split)
    dev = complete[complete["RID"].isin(split.train_rids | split.val_rids)].copy()
    print(f"  development set: {len(dev)} (test of {len(split.test_rids)} locked away)")
    for h in HORIZONS:
        print(f"    converters at {h}mo: {int(dev[f'LABEL_{h}mo'].sum())}")
    print()

    rows = []
    interactions: list[dict] = []
    for block in ACQUIRABLE_BLOCKS:
        matched = subjects_with_blocks(dev, [block])
        if len(matched) < MIN_SUBJECTS:
            print(f"[skip] {block}: only {len(matched)} subjects in the fixed set")
            continue

        print(f"=== {block}: fixed set of {len(matched)} subjects across all horizons ===")
        per_horizon: dict[int, tuple] = {}
        for h in HORIZONS:
            label_col = f"LABEL_{h}mo"
            y = matched[label_col].astype(int).to_numpy()
            n_pos = int(y.sum())
            if len(np.unique(y)) < 2 or n_pos < 10:
                print(f"  {h}mo: only {n_pos} converters -- skipped")
                continue

            p_base = _oof_predictions(matched, ["CHEAP"], label_col)
            p_blk = _oof_predictions(matched, ["CHEAP", block], label_col)
            d, lo, hi = bootstrap_delta_ci(y, p_base, p_blk, auc, seed=RANDOM_STATE)

            rows.append(
                dict(
                    block=block, horizon=h, n=len(matched), n_pos=n_pos,
                    underpowered=n_pos < MIN_POSITIVES,
                    auc_cheap=round(auc(y, p_base), 4),
                    auc_with_block=round(auc(y, p_blk), 4),
                    delta_auc=round(d, 4), delta_lo=round(lo, 4), delta_hi=round(hi, 4),
                    significant=bool(lo > 0 or hi < 0),
                )
            )
            per_horizon[h] = (y, p_base, p_blk, n_pos)
            flag = "  [underpowered]" if n_pos < MIN_POSITIVES else ""
            print(f"  {h}mo: n_pos={n_pos:3d}  cheap={auc(y,p_base):.3f} "
                  f"+{block}={auc(y,p_blk):.3f}  delta={d:+.4f} "
                  f"[{lo:+.4f},{hi:+.4f}]{flag}")

        # Formally test the TREND, not just the per-horizon deltas: is the
        # block worth more at the longest horizon than at the shortest, on
        # these same subjects? Eyeballing that one delta looks bigger is not
        # a test -- overlapping CIs can still hide a significant paired
        # difference, and vice versa.
        hs = sorted(per_horizon)
        if len(hs) >= 2:
            h_lo, h_hi = hs[0], hs[-1]
            y_a, ba, ka, npos_a = per_horizon[h_lo]
            y_b, bb, kb, npos_b = per_horizon[h_hi]
            pt, ilo, ihi = bootstrap_interaction_ci(
                y_a, ba, ka, y_b, bb, kb, auc, seed=RANDOM_STATE
            )
            sig = "SIGNIFICANT" if (ilo > 0 or ihi < 0) else "not significant"
            print(f"  TREND {h_lo}mo -> {h_hi}mo: delta-of-deltas = {pt:+.4f} "
                  f"[{ilo:+.4f},{ihi:+.4f}] -- {sig}")
            interactions.append(
                dict(
                    block=block, horizon_from=h_lo, horizon_to=h_hi,
                    n=len(matched), n_pos_from=npos_a, n_pos_to=npos_b,
                    underpowered=min(npos_a, npos_b) < MIN_POSITIVES,
                    delta_of_deltas=round(pt, 4),
                    lo=round(ilo, 4), hi=round(ihi, 4),
                    significant=bool(ilo > 0 or ihi < 0),
                )
            )
        print()

    results = pd.DataFrame(rows)
    if results.empty:
        print("No block had enough data in the fixed set.")
        return

    print("=== HORIZON TREND (fixed subject set per block) ===")
    piv = results.pivot_table(index="block", columns="horizon", values="delta_auc")
    print(piv.round(4).to_string())
    print("\nInterpretation guard: a trend is only meaningful where the "
          "per-horizon rows are not flagged underpowered.")

    if interactions:
        inter = pd.DataFrame(interactions)
        print()
        print("=== TREND TESTS (delta-of-deltas, paired on the same subjects) ===")
        print(inter.to_string(index=False))
        inter.to_csv("paper/tables/horizon_trend_tests.csv", index=False)

    out = "paper/tables/horizon_effect.csv"
    results.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
