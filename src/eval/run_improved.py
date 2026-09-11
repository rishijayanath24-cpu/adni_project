"""
Experiment 8: can anything actually beat the published policies?

The audit found that no published acquisition policy beats a fixed or
random rule. This tests a specific, mechanistic hypothesis for why, and a
fix: the whole comparison is run on complete cases, discarding ~70% of the
converters, and every policy inherits that handicap through its bank of
subset-specific models. A missingness-native predictor can train on all of
them.

The comparison is deliberately narrow so the result is attributable: SAME
policies, SAME folds, SAME evaluation cohort, SAME cost accounting. The only
thing that changes is the predictor behind them. Any difference is therefore
caused by the predictor, not by a new policy, a new feature set, or a
different split.

Run:  python -m src.eval.run_improved [--horizon 24]
"""
from __future__ import annotations

import argparse
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings(
    "ignore", message="invalid value encountered in reduce", category=RuntimeWarning
)

import src.baselines.base as base
from src.eval.metrics import auc, bootstrap_ci, bootstrap_delta_ci, brier
from src.eval.run_policy_benchmark import BLOCKS, N_FOLDS, RANDOM_STATE, policy_factories
from src.models.features import COST_REGIMES, subjects_with_blocks
from src.models.pooled_bank import PooledBlockModelBank
from src.preprocessing.cohort import build_cohort
from src.preprocessing.leakage_checks import assert_no_subject_leakage
from src.preprocessing.load_adni import build_subject_visits
from src.preprocessing.splits import subject_level_split

POLICIES = ["cheap_only", "fixed_MRI", "fixed_MRI+FDG", "acquire_all",
            "random_rate0.5", "umos_mri_to_amyloid",
            "promuse_full_ladder_sharedpred", "dqn_lambda0.05"]


def evaluate(name, factory, dev, label_col, extra_pool=None):
    """CV a policy, optionally swapping in the missingness-native predictor."""
    y_all = dev[label_col].astype(int).to_numpy()
    probs = np.full(len(dev), np.nan)
    acquired = [[] for _ in range(len(dev))]

    # The policy modules did `from ...base import BlockModelBank` at import
    # time, so each holds its own binding. Patching only base.BlockModelBank
    # silently does nothing -- which showed up as every policy returning
    # byte-identical AUCs under both predictors. Patch every binding.
    import src.baselines.dqn as _dqn
    import src.baselines.promuse as _promuse
    import src.baselines.reference as _reference
    import src.baselines.umos as _umos

    targets = [base, _reference, _umos, _promuse, _dqn, sys.modules[__name__]]
    originals = {m: getattr(m, "BlockModelBank", None) for m in targets}

    if extra_pool is not None:
        class _Patched(PooledBlockModelBank):
            def __init__(self, *a, **kw):
                super().__init__(extra_pool=extra_pool, random_state=RANDOM_STATE)

        for m in targets:
            if originals[m] is not None:
                setattr(m, "BlockModelBank", _Patched)

    try:
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        for tr, te in cv.split(np.zeros(len(dev)), y_all):
            policy = factory().fit(dev.iloc[tr], label_col)
            res = policy.run(dev.iloc[te], label_col)
            probs[te] = res.y_prob
            for pos, blk in zip(te, res.acquired):
                acquired[pos] = blk
    finally:
        for m, orig in originals.items():
            if orig is not None:
                setattr(m, "BlockModelBank", orig)

    lo, hi = bootstrap_ci(y_all, probs, auc, seed=RANDOM_STATE)
    costs = COST_REGIMES["price_like"]
    return (
        dict(policy=name, auc=round(auc(y_all, probs), 4),
             auc_lo=round(lo, 4), auc_hi=round(hi, 4),
             brier=round(brier(y_all, probs), 4),
             cost=round(float(np.mean([sum(costs[b] for b in a) for a in acquired])), 3),
             mean_blocks=round(float(np.mean([len(a) for a in acquired])), 3)),
        probs,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=24)
    args = ap.parse_args()
    label_col = f"LABEL_{args.horizon}mo"

    cohort = build_cohort(build_subject_visits())
    labelled = cohort[cohort[label_col].notna()].copy()
    complete = subjects_with_blocks(labelled, BLOCKS)
    split = subject_level_split(complete, label_col=label_col, random_state=RANDOM_STATE)
    assert_no_subject_leakage(split)

    dev = complete[complete["RID"].isin(split.train_rids | split.val_rids)].copy()
    # Partially-observed subjects usable for training: labelled, and never in
    # the locked test set. These are exactly the subjects complete-case
    # analysis throws away.
    extra_pool = labelled[
        (~labelled["RID"].isin(split.test_rids)) & (~labelled["RID"].isin(dev["RID"]))
    ].copy()

    assert not set(extra_pool["RID"]) & set(split.test_rids), "test leakage via extra pool"
    print(f"{args.horizon}-month horizon")
    print(f"  evaluation cohort (complete cases): n={len(dev)}, "
          f"converters={int(dev[label_col].sum())}")
    print(f"  extra partially-observed training subjects: n={len(extra_pool)}, "
          f"converters={int(extra_pool[label_col].sum())}")
    print(f"  -> training pool grows to {len(dev) + len(extra_pool)} subjects, "
          f"{int(dev[label_col].sum()) + int(extra_pool[label_col].sum())} converters\n")

    factories = policy_factories()
    rows, oof = [], {}
    for name in POLICIES:
        if name not in factories:
            continue
        print(f"running {name} ...", flush=True)
        r_base, p_base = evaluate(name, factories[name], dev, label_col, extra_pool=None)
        r_base["predictor"] = "subset_models (baseline)"
        r_new, p_new = evaluate(name, factories[name], dev, label_col, extra_pool=extra_pool)
        r_new["predictor"] = "pooled (ours)"

        d, lo, hi = bootstrap_delta_ci(
            dev[label_col].astype(int).to_numpy(), p_base, p_new, auc, seed=RANDOM_STATE
        )
        r_new.update(delta_vs_baseline=round(d, 4), delta_lo=round(lo, 4),
                     delta_hi=round(hi, 4), improved=bool(lo > 0))
        rows += [r_base, r_new]
        oof[name] = (p_base, p_new)
        flag = "  <-- significant" if lo > 0 else ""
        print(f"   baseline AUC={r_base['auc']:.4f} | ours AUC={r_new['auc']:.4f} "
              f"| delta={d:+.4f} [{lo:+.4f},{hi:+.4f}]{flag}")

    results = pd.DataFrame(rows)
    print()
    print(f"=== PREDICTOR SWAP, {args.horizon}-month horizon ===")
    print(results[["policy", "predictor", "auc", "auc_lo", "auc_hi", "brier",
                   "cost", "delta_vs_baseline", "delta_lo", "delta_hi"]]
          .to_string(index=False))

    results.to_csv(f"paper/tables/improved_{args.horizon}mo.csv", index=False)
    print(f"\nWrote paper/tables/improved_{args.horizon}mo.csv")


if __name__ == "__main__":
    main()
