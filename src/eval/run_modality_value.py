"""
Experiment 1: incremental value of each acquirable modality block.

Establishes the endpoints of the accuracy-vs-cost frontier the acquisition
policies are later judged against, and tests on our own cohort (with a
leakage-free protocol) the finding that motivates the project: that
population-average gains from expensive modalities are small.

PROTOCOL -- deliberate choices, not defaults:

  * The TEST SET IS NEVER TOUCHED HERE. It is carved out once per horizon and
    set aside for the final confirmatory policy evaluation. All estimates
    below come from repeated stratified cross-validation on the remaining
    subjects. Reporting dozens of test-set numbers during exploration is how
    selective reporting creeps in, and at n_test ~ 100 with ~12 positives the
    CIs would be too wide to learn anything from anyway.

  * Comparisons are MATCHED: cheap-only vs cheap+block on the subjects who
    actually have that block. Unmatched, a modality's apparent value is
    confounded with sample composition -- subjects who received PET are not a
    random subset of the cohort.

  * Preprocessing (imputation, scaling) is fitted inside each CV fold on that
    fold's training rows only.

  * Deltas use a paired bootstrap over pooled out-of-fold predictions for the
    same subjects, which is far more sensitive than comparing two
    independent CIs.

Run:  python -m src.eval.run_modality_value
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold

from src.eval.metrics import auc, bootstrap_ci, bootstrap_delta_ci, brier
from src.models.features import ACQUIRABLE_BLOCKS, BLOCK_AVAILABILITY, fit_design_matrix, subjects_with_blocks
from src.preprocessing.cohort import HORIZONS, build_cohort
from src.preprocessing.leakage_checks import assert_no_subject_leakage
from src.preprocessing.load_adni import build_subject_visits
from src.preprocessing.splits import subject_level_split

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 5
MIN_SUBJECTS = 80
MIN_POSITIVES = 25  # below this, results are flagged underpowered rather than trusted


def _oof_predictions(df: pd.DataFrame, blocks: list[str], label_col: str) -> np.ndarray:
    """Pooled out-of-fold predicted probabilities, averaged over CV repeats.

    Preprocessing and the model are refitted inside every fold on that fold's
    training rows only.
    """
    y = df[label_col].astype(int).to_numpy()
    preds = np.zeros(len(df))
    counts = np.zeros(len(df))

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    for tr_idx, te_idx in cv.split(np.zeros(len(df)), y):
        train_fold = df.iloc[tr_idx]
        test_fold = df.iloc[te_idx]

        dm = fit_design_matrix(train_fold, blocks)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_STATE)
            model.fit(dm.transform(train_fold), train_fold[label_col].astype(int).to_numpy())
            p = model.predict_proba(dm.transform(test_fold))[:, 1]

        preds[te_idx] += p
        counts[te_idx] += 1

    return preds / np.maximum(counts, 1)


def _compare(df: pd.DataFrame, label_col: str, block: str | None, subset_label: str) -> list[dict]:
    """Cheap-only vs cheap+block on one matched subject subset, via CV."""
    y = df[label_col].astype(int).to_numpy()
    n_pos = int(y.sum())
    if len(df) < MIN_SUBJECTS or n_pos < 10 or len(np.unique(y)) < 2:
        return []

    base_blocks = ["CHEAP"]
    p_base = _oof_predictions(df, base_blocks, label_col)

    rows = []
    lo, hi = bootstrap_ci(y, p_base, auc, seed=RANDOM_STATE)
    rows.append(
        dict(
            subset=subset_label, model="CHEAP", n=len(df), n_pos=n_pos,
            underpowered=n_pos < MIN_POSITIVES,
            auc=round(auc(y, p_base), 4), auc_lo=round(lo, 4), auc_hi=round(hi, 4),
            brier=round(brier(y, p_base), 4),
        )
    )

    if block is not None:
        blocks = ["CHEAP", block] if isinstance(block, str) else ["CHEAP", *block]
        name = "CHEAP+" + ("+".join(block) if not isinstance(block, str) else block)
        p_blk = _oof_predictions(df, blocks, label_col)
        lo, hi = bootstrap_ci(y, p_blk, auc, seed=RANDOM_STATE)
        d, dlo, dhi = bootstrap_delta_ci(y, p_base, p_blk, auc, seed=RANDOM_STATE)
        rows.append(
            dict(
                subset=subset_label, model=name, n=len(df), n_pos=n_pos,
                underpowered=n_pos < MIN_POSITIVES,
                auc=round(auc(y, p_blk), 4), auc_lo=round(lo, 4), auc_hi=round(hi, 4),
                brier=round(brier(y, p_blk), 4),
                delta_auc=round(d, 4), delta_lo=round(dlo, 4), delta_hi=round(dhi, 4),
                delta_significant=bool(dlo > 0 or dhi < 0),
            )
        )
    return rows


def main() -> None:
    print("Loading real ADNI cohort...")
    cohort = build_cohort(build_subject_visits())
    print(f"  {len(cohort)} MCI-at-baseline subjects\n")

    all_rows = []
    for h in HORIZONS:
        label_col = f"LABEL_{h}mo"
        labelled = cohort[cohort[label_col].notna()].copy()

        # Carve out and immediately set aside the test set. Nothing below sees it.
        split = subject_level_split(labelled, label_col=label_col, random_state=RANDOM_STATE)
        assert_no_subject_leakage(split)
        dev = labelled[labelled["RID"].isin(split.train_rids | split.val_rids)].copy()

        print(f"=== {h}-month horizon ===")
        print(f"  labelled={len(labelled)}  development={len(dev)} "
              f"(test set of {len(split.test_rids)} locked away, untouched)")
        print(f"  converters in development set: {int(dev[label_col].sum())}")

        avail = {b: int(dev[BLOCK_AVAILABILITY[b]].astype(bool).sum()) for b in ACQUIRABLE_BLOCKS}
        print(f"  block availability within development set: {avail}")

        all_rows += [r | {"horizon": h} for r in _compare(dev, label_col, None, "development_all")]

        for block in ACQUIRABLE_BLOCKS:
            matched = subjects_with_blocks(dev, [block])
            rows = _compare(matched, label_col, block, f"matched_{block}")
            if not rows:
                print(f"  [skip] {block}: n={len(matched)}, "
                      f"pos={int(matched[label_col].sum()) if len(matched) else 0} -- too few")
            all_rows += [r | {"horizon": h} for r in rows]

        # "Acquire everything" upper bounds. PLASMA is excluded from these:
        # only 86 MCI subjects have a baseline draw and almost none of them
        # have conversion follow-up yet (n=4 labelled at 24mo, n=0 at 36mo),
        # since plasma collection is concentrated in recent ADNI4 enrollees.
        # Requiring it would force every complete-case subset to n=0.
        for combo_name, combo in (
            ("matched_MRI+FDG+AMYLOID+CSF", ["MRI", "FDG", "AMYLOID", "CSF"]),
            ("matched_ALL5", ["MRI", "FDG", "AMYLOID", "TAU", "CSF"]),
        ):
            matched_all = subjects_with_blocks(dev, combo)
            rows = _compare(matched_all, label_col, combo, combo_name)
            if not rows:
                print(f"  [skip] {combo_name}: n={len(matched_all)}, "
                      f"pos={int(matched_all[label_col].sum()) if len(matched_all) else 0} -- too few")
            all_rows += [r | {"horizon": h} for r in rows]
        print()

    results = pd.DataFrame(all_rows)
    cols = ["horizon", "subset", "model", "n", "n_pos", "underpowered",
            "auc", "auc_lo", "auc_hi", "brier",
            "delta_auc", "delta_lo", "delta_hi", "delta_significant"]
    results = results[[c for c in cols if c in results.columns]]
    print("=== RESULTS (cross-validated on development set; test set untouched) ===")
    print(results.to_string(index=False))

    out = "paper/tables/modality_value.csv"
    results.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
