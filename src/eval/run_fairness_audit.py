"""
Experiment 4: fairness audit of the published acquisition policies.

The project's primary novelty claim. UMoS, ProMUSE and the 2025 DQN all
report cost savings; none reports whether the savings fall evenly across
demographic groups. A policy that quietly sends one group for fewer scans
is rationing diagnostic access, and that is invisible in an aggregate AUC
or an aggregate "we cut imaging by 60%" headline.

Reported per policy, per subgroup dimension (race, ethnicity, sex):
  * acquisition rate for each modality, and the max gap between subgroups
  * AUC / TPR / TNR / calibration-in-the-large, and their gaps

POWER WARNING, and it is a serious one: ADNI oversamples White,
non-Hispanic participants, so minority subgroups here are tens of subjects
with single-digit converters. Every row carries an `underpowered` flag and
gaps are reported over adequately-powered subgroups only. The honest
finding may well be "ADNI cannot support a well-powered fairness audit of
these policies", which is itself worth stating -- and is an argument for
running the same audit on a more diverse cohort.

Run:  python -m src.eval.run_fairness_audit [--horizon 24]
"""
from __future__ import annotations

import argparse
import warnings

import pandas as pd

warnings.filterwarnings(
    "ignore", message="invalid value encountered in reduce", category=RuntimeWarning
)

from src.eval.fairness import (
    acquisition_disparity,
    benjamini_hochberg,
    collapse_race,
    gaps,
    subgroup_acquisition,
    subgroup_outcomes,
)
from src.eval.run_policy_benchmark import BLOCKS, RANDOM_STATE, evaluate_policy, policy_factories
from src.models.features import subjects_with_blocks
from src.preprocessing.cohort import build_cohort
from src.preprocessing.leakage_checks import assert_no_subject_leakage
from src.preprocessing.load_adni import build_subject_visits
from src.preprocessing.splits import subject_level_split

AUDIT_POLICIES = [
    "cheap_only",
    "acquire_all",
    "fixed_MRI+FDG",
    "umos_mri_to_amyloid",
    "promuse_full_ladder_sharedpred",
    "dqn_lambda0.05",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    label_col = f"LABEL_{args.horizon}mo"

    cohort = build_cohort(build_subject_visits())
    labelled = cohort[cohort[label_col].notna()].copy()
    complete = subjects_with_blocks(labelled, BLOCKS)
    split = subject_level_split(complete, label_col=label_col, random_state=RANDOM_STATE)
    assert_no_subject_leakage(split)
    dev = complete[complete["RID"].isin(split.train_rids | split.val_rids)].copy()
    dev["RACE"] = collapse_race(dev["PTRACCAT"])
    dev["ETHNICITY"] = dev["PTETHCAT"].fillna("Unknown")
    dev["SEX"] = dev["PTGENDER"].fillna("Unknown")

    print(f"{args.horizon}-month horizon: development set n={len(dev)}, "
          f"converters={int(dev[label_col].sum())}")
    print("\nSubgroup composition of the audit cohort:")
    for dim in ["RACE", "ETHNICITY", "SEX"]:
        vc = dev[dim].value_counts()
        pos = dev.groupby(dim)[label_col].sum().astype(int)
        print(f"  {dim}:")
        for g, n in vc.items():
            print(f"    {g}: n={n}, converters={int(pos.get(g, 0))}")
    print()

    factories = policy_factories(quick=args.quick)
    acq_rows, out_rows, gap_rows, disp_rows = [], [], [], []

    for name in AUDIT_POLICIES:
        if name not in factories:
            print(f"[skip] {name} not registered")
            continue
        print(f"auditing {name} ...", flush=True)
        _, probs, acquired = evaluate_policy(name, factories[name], dev, label_col)
        y = dev[label_col].astype(int).to_numpy()

        for dim in ["RACE", "ETHNICITY", "SEX"]:
            a = subgroup_acquisition(dev[dim], acquired, BLOCKS)
            o = subgroup_outcomes(dev[dim], y, probs)
            a.insert(0, "dimension", dim); a.insert(0, "policy", name)
            o.insert(0, "dimension", dim); o.insert(0, "policy", name)
            acq_rows.append(a); out_rows.append(o)

            g = {"policy": name, "dimension": dim}
            g.update(gaps(a, [f"acq_{b}" for b in BLOCKS] + ["mean_n_blocks"]))
            g.update(gaps(o, ["auc", "tpr", "tnr"]))
            gap_rows.append(g)

            # Signed, CI'd disparity between the two largest powered groups.
            powered = a[~a["underpowered"]].sort_values("n", ascending=False)
            if len(powered) >= 2:
                ga, gb = powered.iloc[0]["group"], powered.iloc[1]["group"]
                d = acquisition_disparity(dev[dim], acquired, BLOCKS, ga, gb,
                                          seed=RANDOM_STATE)
                d.insert(0, "dimension", dim)
                d.insert(0, "policy", name)
                disp_rows.append(d)

    acq = pd.concat(acq_rows, ignore_index=True)
    out = pd.concat(out_rows, ignore_index=True)
    gap = pd.DataFrame(gap_rows)

    print()
    print("=== ACQUISITION RATES BY SUBGROUP (does the policy test groups equally?) ===")
    print(acq.to_string(index=False))
    print()
    print("=== OUTCOMES BY SUBGROUP ===")
    print(out.to_string(index=False))
    print()
    print("=== GAPS (powered subgroups only) ===")
    print(gap.to_string(index=False))

    if disp_rows:
        disp = pd.concat(disp_rows, ignore_index=True)
        # Dozens of (policy, modality) gaps are tested here, so a couple of
        # uncorrected "significant" results are expected from noise alone.
        # Only FDR-corrected findings are treated as evidence of disparity.
        disp["significant_fdr"] = benjamini_hochberg(disp["p_value"].to_numpy())
        print()
        print("=== ACQUISITION DISPARITY, signed with 95% CI (positive = group_a tested more) ===")
        print(disp.to_string(index=False))
        disp.to_csv(f"paper/tables/fairness_disparity_{args.horizon}mo.csv", index=False)

    acq.to_csv(f"paper/tables/fairness_acquisition_{args.horizon}mo.csv", index=False)
    out.to_csv(f"paper/tables/fairness_outcomes_{args.horizon}mo.csv", index=False)
    gap.to_csv(f"paper/tables/fairness_gaps_{args.horizon}mo.csv", index=False)
    print(f"\nWrote paper/tables/fairness_*_{args.horizon}mo.csv")


if __name__ == "__main__":
    main()
