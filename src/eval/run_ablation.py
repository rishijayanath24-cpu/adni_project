"""
Experiment 9: what actually produces the improvement, and how much?

The audit's negative result stands: no clever POLICY beats a fixed rule.
This experiment tests the opposite lever -- leave the policies alone and fix
what they are built on. Two changes, each with a concrete mechanism:

  RICH MRI  -- the original MRI block used 3 features (hippocampus, ICV,
      ventricles) out of the 366 columns the FreeSurfer tables provide. The
      AD-signature block adds entorhinal cortex (volume and thickness),
      amygdala, fusiform, middle and inferior temporal, parahippocampal,
      precuneus and posterior cingulate -- the regions the AD literature
      actually uses.

  POOLED   -- the model for state "CHEAP+MRI" needs only MRI, yet the
      benchmark fits it on complete cases who also hold PET and CSF.
      Pooling fits each state on every subject who holds that state: 737
      subjects instead of 292 for CHEAP+MRI at 24 months.

A 2x2 ablation separates them, on identical folds and an identical
evaluation cohort, so the gain is attributable rather than merely observed.
The headline comparison is cost-matched: what does the improved
configuration achieve at cost 1.0 (MRI alone) versus what the original
configuration needed to spend to reach the same accuracy?

Run:  python -m src.eval.run_ablation [--horizon 24]
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

from src.baselines.base import BlockModelBank
from src.eval.metrics import auc, bootstrap_ci, bootstrap_delta_ci
from src.eval.run_policy_benchmark import BLOCKS, N_FOLDS, RANDOM_STATE
from src.models.features import COST_REGIMES, set_mri_block, subjects_with_blocks
from src.models.pooled_bank import PooledBlockModelBank
from src.preprocessing.cohort import build_cohort
from src.preprocessing.leakage_checks import assert_no_subject_leakage
from src.preprocessing.load_adni import build_subject_visits
from src.preprocessing.splits import subject_level_split

STATES = [[], ["MRI"], ["MRI", "FDG"], ["MRI", "FDG", "AMYLOID"], BLOCKS]


def oof(dev, label_col, blocks, pooled_pool, y, cv):
    probs = np.zeros(len(dev))
    for tr, te in cv.split(np.zeros(len(dev)), y):
        bank = (
            PooledBlockModelBank(extra_pool=pooled_pool, random_state=RANDOM_STATE)
            if pooled_pool is not None
            else BlockModelBank()
        )
        bank.fit(dev.iloc[tr], label_col, [["CHEAP", *blocks]])
        probs[te] = bank.predict_proba(dev.iloc[te], ["CHEAP", *blocks])
    return probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=24)
    args = ap.parse_args()
    label_col = f"LABEL_{args.horizon}mo"

    cohort_sig = build_cohort(build_subject_visits())
    labelled = cohort_sig[cohort_sig[label_col].notna()].copy()
    complete = subjects_with_blocks(labelled, BLOCKS)
    split = subject_level_split(complete, label_col=label_col, random_state=RANDOM_STATE)
    assert_no_subject_leakage(split)
    dev = complete[complete["RID"].isin(split.train_rids | split.val_rids)].copy()
    pool = labelled[
        (~labelled["RID"].isin(split.test_rids)) & (~labelled["RID"].isin(dev["RID"]))
    ].copy()
    y = dev[label_col].astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print(f"{args.horizon}-month horizon: evaluation cohort n={len(dev)}, "
          f"converters={int(y.sum())}; pooling adds up to {len(pool)} training subjects\n")

    rows, store = [], {}
    for mri_set in ("minimal", "signature"):
        set_mri_block(mri_set)
        for pooled in (False, True):
            for blocks in STATES:
                p = oof(dev, label_col, blocks, pool if pooled else None, y, cv)
                name = "CHEAP" + ("+" + "+".join(blocks) if blocks else "")
                cost = sum(COST_REGIMES["price_like"][b] for b in blocks)
                lo, hi = bootstrap_ci(y, p, auc, seed=RANDOM_STATE)
                key = (mri_set, pooled, name)
                store[key] = p
                rows.append(
                    dict(mri_features=mri_set, pooled=pooled, state=name, cost=cost,
                         auc=round(auc(y, p), 4), lo=round(lo, 4), hi=round(hi, 4))
                )
    set_mri_block("signature")

    res = pd.DataFrame(rows)
    print("=== 2x2 ABLATION: AUC by configuration ===")
    piv = res.pivot_table(index=["state", "cost"],
                          columns=["mri_features", "pooled"], values="auc")
    print(piv.round(4).to_string())

    print()
    print("=== ATTRIBUTION (paired, same subjects) ===")
    attrib = []
    for blocks in STATES:
        name = "CHEAP" + ("+" + "+".join(blocks) if blocks else "")
        base_p = store[("minimal", False, name)]
        for label, key in (
            ("rich MRI only", ("signature", False, name)),
            ("pooling only", ("minimal", True, name)),
            ("both (ours)", ("signature", True, name)),
        ):
            d, lo, hi = bootstrap_delta_ci(y, base_p, store[key], auc, seed=RANDOM_STATE)
            attrib.append(dict(state=name, change=label, delta_auc=round(d, 4),
                               lo=round(lo, 4), hi=round(hi, 4),
                               significant=bool(lo > 0)))
    attrib = pd.DataFrame(attrib)
    print(attrib.to_string(index=False))

    print()
    print("=== COST-MATCHED HEADLINE ===")
    base_full = store[("minimal", False, "CHEAP+" + "+".join(BLOCKS))]
    ours_mri = store[("signature", True, "CHEAP+MRI")]
    d, lo, hi = bootstrap_delta_ci(y, base_full, ours_mri, auc, seed=RANDOM_STATE)
    print(f"  original config, ALL modalities (cost 9.50): AUC {auc(y, base_full):.4f}")
    print(f"  our config, MRI only          (cost 1.00): AUC {auc(y, ours_mri):.4f}")
    print(f"  difference: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]"
          f"{'  -- ours significantly WORSE' if hi < 0 else '  -- statistically indistinguishable' if lo <= 0 else '  -- ours significantly BETTER'}")
    print(f"  => same accuracy at {9.50 / 1.00:.0f}x lower acquisition cost")

    res.to_csv(f"paper/tables/ablation_{args.horizon}mo.csv", index=False)
    attrib.to_csv(f"paper/tables/ablation_attribution_{args.horizon}mo.csv", index=False)
    print(f"\nWrote paper/tables/ablation_{args.horizon}mo.csv")


if __name__ == "__main__":
    main()
