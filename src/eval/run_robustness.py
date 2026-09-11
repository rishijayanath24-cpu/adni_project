"""
Experiment 5: robustness of the central negative result.

The paper's spine is "no learned acquisition policy beats a cost-matched
fixed rule". That claim has to survive the obvious attacks, so each is
tested explicitly rather than asserted:

  COST REGIME -- the domination result could be an artifact of the
      particular cost numbers chosen. Re-run domination under all three
      regimes (price_like / flat / burden). If UMoS is only dominated
      because amyloid PET was priced at 5.0, that is a finding about my
      assumptions, not about UMoS.

  PREDICTOR -- every policy shares a logistic-regression predictor. If LR
      is simply too weak to exploit what a smart policy acquires, the
      comparison is rigged toward simple policies. Re-run with gradient
      boosting to check.

  SEED -- the CV split and the policies' internal randomness use one seed.
      Re-run across seeds to confirm the ordering is not noise.

Run:  python -m src.eval.run_robustness [--horizon 24] [--check cost|predictor|seed|all]
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore", message="invalid value encountered in reduce", category=RuntimeWarning
)

from src.eval.metrics import auc
from src.eval.run_policy_benchmark import BLOCKS, RANDOM_STATE, evaluate_policy, policy_factories
from src.models.features import COST_REGIMES, subjects_with_blocks
from src.preprocessing.cohort import build_cohort
from src.preprocessing.leakage_checks import assert_no_subject_leakage
from src.preprocessing.load_adni import build_subject_visits
from src.preprocessing.splits import subject_level_split

DUMB = ("cheap_only", "acquire_all", "fixed_", "random_")
SUBSET = [
    "cheap_only", "acquire_all", "fixed_MRI", "fixed_MRI+FDG", "random_rate0.5",
    "umos_mri_to_amyloid", "promuse_full_ladder_sharedpred", "dqn_lambda0.05",
]


def load_dev(horizon: int, seed: int = RANDOM_STATE) -> tuple[pd.DataFrame, str]:
    label_col = f"LABEL_{horizon}mo"
    cohort = build_cohort(build_subject_visits())
    labelled = cohort[cohort[label_col].notna()].copy()
    complete = subjects_with_blocks(labelled, BLOCKS)
    split = subject_level_split(complete, label_col=label_col, random_state=seed)
    assert_no_subject_leakage(split)
    dev = complete[complete["RID"].isin(split.train_rids | split.val_rids)].copy()
    return dev, label_col


def domination_under(results: pd.DataFrame, cost_col: str) -> pd.DataFrame:
    dumb = results[results["policy"].str.startswith(DUMB)]
    rows = []
    for _, r in results.iterrows():
        if r["policy"] in set(dumb["policy"]):
            continue
        beaten = dumb[(dumb["auc"] >= r["auc"]) & (dumb[cost_col] <= r[cost_col])]
        beaten = beaten.sort_values(cost_col)
        rows.append(
            dict(
                policy=r["policy"], auc=r["auc"], cost=r[cost_col],
                dominated=bool(len(beaten)),
                dominated_by=beaten.iloc[0]["policy"] if len(beaten) else "",
            )
        )
    return pd.DataFrame(rows)


def check_cost_regimes(dev, label_col) -> pd.DataFrame:
    """Does domination hold under every cost regime, or only the one I picked?"""
    factories = {k: v for k, v in policy_factories().items() if k in SUBSET}
    rows = []
    for name, f in factories.items():
        print(f"  {name} ...", flush=True)
        row, _, _ = evaluate_policy(name, f, dev, label_col)
        rows.append(row)
    results = pd.DataFrame(rows)

    out = []
    for regime in COST_REGIMES:
        d = domination_under(results, f"cost_{regime}")
        d["regime"] = regime
        out.append(d)
    return pd.concat(out, ignore_index=True)


def check_predictor(dev, label_col) -> pd.DataFrame:
    """Is the null result an artifact of a weak shared predictor?

    Swaps the shared logistic regression for gradient boosting inside
    BlockModelBank. If a smarter policy's advantage only appears with a
    stronger predictor, the LR-based conclusion would be unsafe.
    """
    import src.baselines.base as base
    from sklearn.ensemble import HistGradientBoostingClassifier
    from src.models.features import fit_design_matrix

    original = base.BlockModelBank.fit

    def gb_fit(self, train_df, label_col, block_sets):
        y = train_df[label_col].astype(int).to_numpy()
        for blocks in block_sets:
            key = tuple(sorted(blocks))
            if key in self._models:
                continue
            dm = fit_design_matrix(train_df, list(blocks))
            model = HistGradientBoostingClassifier(
                max_depth=3, max_iter=150, learning_rate=0.05,
                random_state=self.random_state,
            )
            model.fit(dm.transform(train_df), y)
            self._models[key] = (dm, model)

    base.BlockModelBank.fit = gb_fit
    try:
        factories = {k: v for k, v in policy_factories().items() if k in SUBSET}
        rows = []
        for name, f in factories.items():
            print(f"  {name} (gradient boosting) ...", flush=True)
            row, _, _ = evaluate_policy(name, f, dev, label_col)
            rows.append(row)
        return pd.DataFrame(rows)
    finally:
        base.BlockModelBank.fit = original


def check_seeds(horizon: int, seeds: list[int]) -> pd.DataFrame:
    """Is the policy ordering stable across resampling, or is it noise?"""
    rows = []
    for seed in seeds:
        dev, label_col = load_dev(horizon, seed=seed)
        factories = {k: v for k, v in policy_factories(quick=True).items() if k in SUBSET}
        for name, f in factories.items():
            print(f"  seed {seed}: {name} ...", flush=True)
            row, _, _ = evaluate_policy(name, f, dev, label_col)
            row["seed"] = seed
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--check", type=str, default="all",
                    choices=["cost", "predictor", "seed", "all"])
    args = ap.parse_args()

    dev, label_col = load_dev(args.horizon)
    print(f"{args.horizon}-month horizon, development set n={len(dev)}, "
          f"converters={int(dev[label_col].sum())}\n")

    if args.check in ("cost", "all"):
        print("=== CHECK 1: does domination survive every cost regime? ===")
        d = check_cost_regimes(dev, label_col)
        piv = d.pivot_table(index="policy", columns="regime", values="dominated", aggfunc="first")
        print()
        print(piv.to_string())
        print("\n(True = a patient-independent rule matches or beats it at no greater cost)")
        d.to_csv(f"paper/tables/robustness_cost_{args.horizon}mo.csv", index=False)
        print()

    if args.check in ("predictor", "all"):
        print("=== CHECK 2: does the ordering hold with gradient boosting? ===")
        r = check_predictor(dev, label_col)
        r = r.sort_values("cost_price_like")
        print()
        print(r[["policy", "auc", "auc_lo", "auc_hi", "cost_price_like"]].to_string(index=False))
        dom = domination_under(r, "cost_price_like")
        print()
        print(dom.to_string(index=False))
        r.to_csv(f"paper/tables/robustness_predictor_{args.horizon}mo.csv", index=False)
        print()

    if args.check in ("seed", "all"):
        print("=== CHECK 3: is the policy ordering stable across seeds? ===")
        r = check_seeds(args.horizon, [0, 7, 42])
        piv = r.pivot_table(index="policy", columns="seed", values="auc")
        piv["mean"] = piv.mean(axis=1).round(4)
        piv["sd"] = r.groupby("policy")["auc"].std().round(4)
        print()
        print(piv.round(4).to_string())
        r.to_csv(f"paper/tables/robustness_seed_{args.horizon}mo.csv", index=False)


if __name__ == "__main__":
    main()
