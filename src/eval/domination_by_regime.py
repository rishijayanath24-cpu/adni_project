"""
Cost-domination of every published configuration, under every cost regime.

Derived from the committed main-benchmark tables (paper/tables/
policy_benchmark_{h}mo.csv), which already carry each policy's mean cost
under all three regimes. Uses the same rule as run_policy_benchmark's
cost_matched_table: a published configuration is dominated if some
patient-independent rule (cheap-only, a fixed prefix, random, acquire-all)
has AUC at least as high and cost no higher, under that regime.

This replaces the earlier cost-regime check in run_robustness.py, which was
run before the MRI block was enlarged and so used different inputs from the
main benchmark.

Run:  python -m src.eval.domination_by_regime
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TABLES = Path(__file__).resolve().parents[2] / "paper" / "tables"
REFERENCE = ("fixed_", "random_", "cheap_only", "acquire_all")
PUBLISHED = ("umos", "promuse", "dqn")
REGIMES = ("price_like", "flat", "burden")


def main() -> None:
    rows = []
    for h in (12, 24, 36):
        b = pd.read_csv(TABLES / f"policy_benchmark_{h}mo.csv")
        ref = b[b.policy.str.startswith(REFERENCE)]
        pub = b[b.policy.str.startswith(PUBLISHED)]
        for _, r in pub.iterrows():
            row = dict(horizon=h, policy=r.policy, auc=r.auc)
            for reg in REGIMES:
                col = f"cost_{reg}"
                beat = ref[(ref.auc >= r.auc) & (ref[col] <= r[col])].sort_values(col)
                row[f"cost_{reg}"] = r[col]
                row[f"dominated_{reg}"] = bool(len(beat))
                row[f"dominated_by_{reg}"] = beat.iloc[0].policy if len(beat) else ""
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "domination_by_regime.csv", index=False)
    for reg in REGIMES:
        per_h = out.groupby("horizon")[f"dominated_{reg}"].sum().to_dict()
        print(f"{reg:10s} dominated {int(out[f'dominated_{reg}'].sum())}/{len(out)}  by horizon {per_h}")
    allr = out[[f"dominated_{r}" for r in REGIMES]].all(axis=1).sum()
    anyr = out[[f"dominated_{r}" for r in REGIMES]].any(axis=1).sum()
    print(f"dominated under all three regimes: {allr}/{len(out)}; under at least one: {anyr}/{len(out)}")
    print(out[out.horizon == 24].to_string(index=False))


if __name__ == "__main__":
    main()
