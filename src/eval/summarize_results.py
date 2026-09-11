"""
Consolidate the benchmark across horizons into the paper's headline table.

Answers the two questions the audit exists to answer, in one place:
  1. Does any published policy beat RANDOM acquisition? (does choosing
     smarter help?)
  2. Is any published policy dominated by a dumb fixed rule that costs no
     more? (is the personalisation worth its price?)

Run:  python -m src.eval.summarize_results
"""
from __future__ import annotations

import pandas as pd

HORIZONS = (12, 24, 36)


def main() -> None:
    bench, paired, dom = [], [], []
    for h in HORIZONS:
        try:
            b = pd.read_csv(f"paper/tables/policy_benchmark_{h}mo.csv"); b["horizon"] = h
            p = pd.read_csv(f"paper/tables/policy_paired_{h}mo.csv"); p["horizon"] = h
            d = pd.read_csv(f"paper/tables/policy_domination_{h}mo.csv"); d["horizon"] = h
        except FileNotFoundError:
            print(f"[skip] horizon {h}: tables missing")
            continue
        bench.append(b); paired.append(p); dom.append(d)

    if not bench:
        print("no benchmark tables found")
        return

    bench = pd.concat(bench, ignore_index=True)
    paired = pd.concat(paired, ignore_index=True)
    dom = pd.concat(dom, ignore_index=True)

    print("=== AUC by policy and horizon ===")
    piv = bench.pivot_table(index="policy", columns="horizon", values="auc")
    cost = bench.groupby("policy")["cost_price_like"].mean().round(2)
    piv.insert(0, "cost", cost)
    print(piv.sort_values("cost").round(4).to_string())

    print()
    print("=== Q1: does any policy BEAT random acquisition? ===")
    vs_random = paired[paired["reference"] == "random_rate0.5"]
    better = vs_random[(vs_random["significant"]) & (vs_random["delta_auc"] > 0)]
    worse = vs_random[(vs_random["significant"]) & (vs_random["delta_auc"] < 0)]
    if better.empty:
        print("  NO policy significantly beats random at any horizon.")
    else:
        print(better[["horizon", "policy", "delta_auc", "lo", "hi"]].to_string(index=False))
    print(f"  ({len(worse)} policy-horizon pairs are significantly WORSE than random.)")

    print()
    print("=== Q2: learned policies dominated by a cheaper-or-equal dumb rule ===")
    dominated = dom[dom["dominated_by"].notna() & (dom["dominated_by"] != "")]
    print(dominated[["horizon", "policy", "auc", "cost", "dominated_by",
                     "dominator_auc", "dominator_cost"]].to_string(index=False))

    out = "paper/tables/summary_across_horizons.csv"
    piv.to_csv(out)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
