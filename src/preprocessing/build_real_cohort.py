"""
Build the real MCI-at-baseline cohort from actual ADNI data and report
summary numbers -- the real-data counterpart to smoke_test.py.

Run from the project root:
    python -m src.preprocessing.build_real_cohort
"""
from __future__ import annotations

from src.preprocessing.cohort import HORIZONS, build_cohort, cohort_summary
from src.preprocessing.leakage_checks import assert_no_subject_leakage, assert_single_split_membership
from src.preprocessing.load_adni import build_subject_visits
from src.preprocessing.splits import subject_level_split


def main() -> None:
    print("Loading real ADNI tables...")
    subject_visits = build_subject_visits()
    print(f"  {len(subject_visits)} subject-visit rows, {subject_visits['RID'].nunique()} subjects")

    print("\nBuilding MCI-at-baseline cohort...")
    cohort = build_cohort(subject_visits)
    print(f"  {len(cohort)} MCI-at-baseline subjects")

    print("\nCohort summary by horizon:")
    print(cohort_summary(cohort).to_string(index=False))

    print("\nModality availability at baseline:")
    for col in ["HAS_MRI_bl", "HAS_PET_bl", "HAS_CSF_bl"]:
        pct = 100 * cohort[col].mean()
        print(f"  {col}: {pct:.1f}%  (n={int(cohort[col].sum())})")

    print("\nSubgroup composition (for the fairness audit later):")
    print("  PTRACCAT:")
    print(cohort["PTRACCAT"].value_counts(dropna=False).to_string())
    print("  PTETHCAT:")
    print(cohort["PTETHCAT"].value_counts(dropna=False).to_string())
    print("  FIELD_STRENGTH:")
    print(cohort["FIELD_STRENGTH"].value_counts(dropna=False).to_string())

    for h in HORIZONS:
        label_col = f"LABEL_{h}mo"
        usable = cohort[cohort[label_col].notna()].copy()
        print(f"\nSplitting on {h}-month horizon ({len(usable)} non-censored subjects)...")
        split = subject_level_split(usable, label_col=label_col)
        assert_no_subject_leakage(split)

        usable["split"] = "unassigned"
        usable.loc[usable["RID"].isin(split.train_rids), "split"] = "train"
        usable.loc[usable["RID"].isin(split.val_rids), "split"] = "val"
        usable.loc[usable["RID"].isin(split.test_rids), "split"] = "test"
        assert_single_split_membership(usable, "split")

        print(f"  train={len(split.train_rids)} val={len(split.val_rids)} test={len(split.test_rids)}")
        print("  leakage check passed.")

    out_path = "data/processed/mci_cohort.csv"
    cohort.to_csv(out_path, index=False)
    print(f"\nWrote cohort table to {out_path}")
    print("\nAll checks passed on real data.")


if __name__ == "__main__":
    main()
