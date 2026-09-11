"""
Experiment 7: does conditioning the acquisition ladder on the prediction
horizon buy anything?

This is the project's own proposed contribution, and it is tested against
the same bar everything else faced: it must beat a horizon-AGNOSTIC policy
at matched cost, and it must beat random acquisition. The comparison that
matters is the horizon-swap control -- running the 36-month ladder at the
12-month horizon and vice versa. If conditioning helps, the matched ladder
should beat the swapped one; if the two are indistinguishable, the horizon
signal is not exploitable even though the underlying effect is real.

Run:  python -m src.eval.run_horizon_policy
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore", message="invalid value encountered in reduce", category=RuntimeWarning
)

from src.baselines.horizon_conditioned import DEFAULT_LADDERS, HorizonConditionedPolicy
from src.eval.metrics import auc, bootstrap_ci, bootstrap_delta_ci
from src.eval.run_policy_benchmark import RANDOM_STATE, evaluate_policy, policy_factories
from src.eval.run_robustness import load_dev

HORIZONS = (12, 24, 36)


def main() -> None:
    rows, oof_store = [], {}

    for h in HORIZONS:
        dev, label_col = load_dev(h)
        y = dev[label_col].astype(int).to_numpy()
        print(f"=== {h}-month horizon: n={len(dev)}, converters={int(y.sum())} ===")

        variants = {
            # The matched ladder: the one our own horizon analysis motivates.
            f"matched_{h}": lambda h=h: HorizonConditionedPolicy(h),
            # Horizon-swap controls: same machinery, wrong horizon's ladder.
            **{
                f"swapped_from_{other}": (
                    lambda h=h, other=other: HorizonConditionedPolicy(
                        h, ladders={h: DEFAULT_LADDERS[other]}
                    )
                )
                for other in HORIZONS
                if other != h
            },
        }

        store = {}
        for name, factory in variants.items():
            row, p, _ = evaluate_policy(name, factory, dev, label_col)
            row["horizon"] = h
            rows.append(row)
            store[name] = p
            print(f"  {name}: AUC={row['auc']:.4f} cost={row['cost_price_like']:.2f} "
                  f"blocks={row['mean_n_blocks']:.2f}")

        # Reference points from the main benchmark.
        for ref in ["fixed_MRI+FDG", "random_rate0.5"]:
            row, p, _ = evaluate_policy(ref, policy_factories()[ref], dev, label_col)
            row["horizon"] = h
            rows.append(row)
            store[ref] = p
            print(f"  {ref}: AUC={row['auc']:.4f} cost={row['cost_price_like']:.2f}")

        oof_store[h] = (y, store)
        print()

    print("=== DOES THE MATCHED LADDER BEAT ITS CONTROLS? (paired, same subjects) ===")
    comp = []
    for h, (y, store) in oof_store.items():
        matched = store[f"matched_{h}"]
        for name, p in store.items():
            if name == f"matched_{h}":
                continue
            d, lo, hi = bootstrap_delta_ci(y, p, matched, auc, seed=RANDOM_STATE)
            comp.append(
                dict(horizon=h, matched_vs=name, delta_auc=round(d, 4),
                     lo=round(lo, 4), hi=round(hi, 4),
                     matched_better=bool(lo > 0), matched_worse=bool(hi < 0))
            )
    comp = pd.DataFrame(comp)
    print(comp.to_string(index=False))

    wins = comp[comp["matched_better"]]
    print()
    if wins.empty:
        print("VERDICT: the horizon-matched ladder does NOT significantly beat any control,")
        print("including its own horizon-swapped versions. The amyloid horizon effect is")
        print("real (Experiment 2) but is not exploitable as an acquisition rule here.")
    else:
        print("Matched ladder significantly beats:")
        print(wins.to_string(index=False))

    pd.DataFrame(rows).to_csv("paper/tables/horizon_policy.csv", index=False)
    comp.to_csv("paper/tables/horizon_policy_comparisons.csv", index=False)
    print("\nWrote paper/tables/horizon_policy*.csv")


if __name__ == "__main__":
    main()
