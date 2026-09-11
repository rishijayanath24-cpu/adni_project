"""
Experiment 10: does our configuration Pareto-dominate the published policies?

"Higher AUC" alone is the wrong comparison when policies sit at different
acquisition costs. The right one is dominance: policy A dominates policy B
if A is at least as accurate AND costs no more. That is the claim a
clinician cares about, and it is the claim this project can actually
support -- our gains are modest in AUC but arrive at a fraction of the cost.

Run:  python -m src.eval.run_dominance
"""
from __future__ import annotations

import pandas as pd

HORIZONS = (12, 24, 36)
OURS = ("ours_MRI", "ours_MRI+FDG")
PUBLISHED_PREFIX = ("umos", "promuse", "dqn")


def main() -> None:
    rows = []
    for h in HORIZONS:
        try:
            r = pd.read_csv(f"paper/tables/policy_benchmark_{h}mo.csv")
        except FileNotFoundError:
            continue
        ours = r[r["policy"].isin(OURS)]
        published = r[r["policy"].str.startswith(PUBLISHED_PREFIX)]
        if ours.empty or published.empty:
            continue

        for _, pub in published.iterrows():
            # Cheapest configuration of ours that is at least as accurate.
            dom = ours[
                (ours["auc"] >= pub["auc"]) & (ours["cost_price_like"] <= pub["cost_price_like"])
            ].sort_values("cost_price_like")
            best = ours.sort_values("cost_price_like").iloc[0]
            rows.append(
                dict(
                    horizon=h,
                    published_policy=pub["policy"],
                    pub_auc=round(pub["auc"], 4),
                    pub_cost=round(pub["cost_price_like"], 2),
                    dominated_by_ours=dom.iloc[0]["policy"] if len(dom) else "",
                    ours_auc=round(dom.iloc[0]["auc"], 4) if len(dom) else round(best["auc"], 4),
                    ours_cost=round(dom.iloc[0]["cost_price_like"], 2) if len(dom)
                    else round(best["cost_price_like"], 2),
                    cost_ratio=round(
                        pub["cost_price_like"] / dom.iloc[0]["cost_price_like"], 2
                    ) if len(dom) and dom.iloc[0]["cost_price_like"] > 0 else None,
                )
            )

    res = pd.DataFrame(rows)
    print("=== DOES OUR CONFIGURATION DOMINATE EACH PUBLISHED POLICY? ===")
    print("(dominated = ours is at least as accurate AND costs no more)\n")
    print(res.to_string(index=False))

    print()
    n_dom = int((res["dominated_by_ours"] != "").sum())
    print(f"Dominated: {n_dom} of {len(res)} published policy-horizon pairs.")
    not_dom = res[res["dominated_by_ours"] == ""]
    if len(not_dom):
        print("\nNOT dominated (these are the honest exceptions):")
        print(not_dom[["horizon", "published_policy", "pub_auc", "pub_cost",
                       "ours_auc", "ours_cost"]].to_string(index=False))

    res.to_csv("paper/tables/dominance_vs_published.csv", index=False)
    print("\nWrote paper/tables/dominance_vs_published.csv")


if __name__ == "__main__":
    main()
