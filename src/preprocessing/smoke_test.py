"""
End-to-end smoke test of the preprocessing pipeline using synthetic data.

Not a unit test suite -- a runnable script exercising
generate -> cohort construction -> split -> leakage check, printing a
cohort summary so the pipeline's behavior can be sanity-checked by eye
before real ADNI data arrives. Run from the project root:

    python -m src.preprocessing.smoke_test
"""
from __future__ import annotations

from src.preprocessing.cohort import HORIZONS, build_cohort, cohort_summary
from src.preprocessing.leakage_checks import assert_no_subject_leakage, assert_single_split_membership
from src.preprocessing.mock_adnimerge import generate_mock_adnimerge
from src.preprocessing.splits import subject_level_split


def main() -> None:
    print("Generating synthetic ADNIMERGE-shaped data...")
    subject_visits = generate_mock_adnimerge(n_subjects=1500, seed=0)
    print(f"  {len(subject_visits)} subject-visit rows, {subject_visits['RID'].nunique()} subjects")

    print("\nBuilding MCI-at-baseline cohort...")
    cohort = build_cohort(subject_visits)
    print(f"  {len(cohort)} MCI-at-baseline subjects")

    print("\nCohort summary by horizon:")
    print(cohort_summary(cohort).to_string(index=False))

    print("\nModality availability at baseline:")
    for col in ["HAS_MRI_bl", "HAS_PET_bl", "HAS_CSF_bl"]:
        pct = 100 * cohort[col].mean()
        print(f"  {col}: {pct:.1f}%")

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

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
