"""
Nested cross-validation: an honest performance estimate for the final models.

train_final.py picks the regularisation strength C by cross-validation and
then reports the CV score at the chosen C. That number is optimistically
biased -- C was selected to maximise exactly the quantity being reported, so
it inherits the maximum of a noisy search.

Nested CV removes that bias without spending the locked test set: the OUTER
loop measures performance, and C is re-selected from scratch inside each
outer fold using only that fold's training data. The reported score is
therefore never the score anything was tuned on.

The gap between the two numbers is itself worth reporting -- it quantifies
how much of the apparent gain is selection noise.

Run:  python -m src.models.nested_cv [--all-horizons]
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings(
    "ignore", message="invalid value encountered in reduce", category=RuntimeWarning
)

from src.eval.metrics import auc, bootstrap_ci, brier
from src.models.pooled_bank import PooledBlockModelBank
from src.models.train_final import (
    C_GRID,
    N_FOLDS,
    OPERATING_POINTS,
    RANDOM_STATE,
    prepare,
)
from src.preprocessing.cohort import HORIZONS


def _inner_select_C(train_df, pool, label_col, blocks) -> float:
    """Choose C using only this outer fold's training rows."""
    y = train_df[label_col].astype(int).to_numpy()
    if len(np.unique(y)) < 2 or len(train_df) < 3 * N_FOLDS:
        return 1.0
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    best_c, best = 1.0, -np.inf
    for c in C_GRID:
        p = np.zeros(len(train_df))
        for tr, te in inner.split(np.zeros(len(train_df)), y):
            bank = PooledBlockModelBank(extra_pool=pool, random_state=RANDOM_STATE, C=c)
            bank.fit(train_df.iloc[tr], label_col, [["CHEAP", *blocks]])
            p[te] = bank.predict_proba(train_df.iloc[te], ["CHEAP", *blocks])
        a = auc(y, p)
        if a > best:
            best_c, best = c, a
    return best_c


def nested_score(horizon: int, name: str, blocks: list[str]) -> dict:
    dev, pool, label_col, _ = prepare(horizon)
    y = dev[label_col].astype(int).to_numpy()
    outer = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    probs = np.zeros(len(dev))
    chosen = []
    for tr, te in outer.split(np.zeros(len(dev)), y):
        train_fold = dev.iloc[tr]
        c = _inner_select_C(train_fold, pool, label_col, blocks)
        chosen.append(c)
        bank = PooledBlockModelBank(extra_pool=pool, random_state=RANDOM_STATE, C=c)
        bank.fit(train_fold, label_col, [["CHEAP", *blocks]])
        probs[te] = bank.predict_proba(dev.iloc[te], ["CHEAP", *blocks])

    lo, hi = bootstrap_ci(y, probs, auc, seed=RANDOM_STATE)
    return dict(
        model=name, horizon=horizon,
        nested_auc=round(auc(y, probs), 4),
        nested_lo=round(lo, 4), nested_hi=round(hi, 4),
        nested_brier=round(brier(y, probs), 4),
        C_per_fold=",".join(str(c) for c in chosen),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--all-horizons", action="store_true")
    args = ap.parse_args()
    horizons = HORIZONS if args.all_horizons else (args.horizon,)

    rows = []
    for h in horizons:
        for name, blocks in OPERATING_POINTS.items():
            print(f"nested CV: {name} at {h}mo ...", flush=True)
            rows.append(nested_score(h, name, blocks))

    nested = pd.DataFrame(rows)
    try:
        reg = pd.read_csv("models/model_registry.csv")[["model", "horizon", "cv_auc"]]
        nested = nested.merge(reg, on=["model", "horizon"], how="left")
        nested["optimism"] = (nested["cv_auc"] - nested["nested_auc"]).round(4)
    except FileNotFoundError:
        pass

    print()
    print("=== NESTED CV (honest) vs TUNED CV (optimistic) ===")
    print(nested.to_string(index=False))
    if "optimism" in nested:
        print(f"\nmean selection optimism: {nested['optimism'].mean():+.4f} AUC")
        print("Report the nested figure. The tuned figure is what C was chosen to maximise.")

    nested.to_csv("models/nested_cv.csv", index=False)
    print("\nWrote models/nested_cv.csv")


if __name__ == "__main__":
    main()
