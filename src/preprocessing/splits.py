"""
Subject-level, leakage-safe train/val/test splitting.

The single largest source of inflated, non-reproducible accuracy documented
in ADNI deep-learning studies is splitting at the visit/scan level instead
of the subject level (see project README). Every split in this project MUST
go through this module, and MUST be created before any preprocessing
statistic (scaling, imputation, feature selection) is computed on the data.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class CohortSplit:
    train_rids: frozenset
    val_rids: frozenset
    test_rids: frozenset

    def __post_init__(self) -> None:
        pairs = [
            ("train", "val", self.train_rids & self.val_rids),
            ("train", "test", self.train_rids & self.test_rids),
            ("val", "test", self.val_rids & self.test_rids),
        ]
        for a, b, overlap in pairs:
            if overlap:
                raise ValueError(f"Leakage: {len(overlap)} subject(s) in both {a} and {b}: {sorted(overlap)[:5]}...")

    def apply(self, df: pd.DataFrame, rid_col: str = "RID") -> dict[str, pd.DataFrame]:
        return {
            "train": df[df[rid_col].isin(self.train_rids)].copy(),
            "val": df[df[rid_col].isin(self.val_rids)].copy(),
            "test": df[df[rid_col].isin(self.test_rids)].copy(),
        }


def _stratified_split(rids, strata, test_size, random_state):
    """Try joint (label, stratum) stratification; fall back to label-only if a
    combination is too rare to stratify on, which real ADNI subgroups will hit."""
    try:
        return train_test_split(rids, test_size=test_size, random_state=random_state, stratify=strata)
    except ValueError as exc:
        warnings.warn(
            f"Falling back to label-only stratification ({exc}). "
            "This usually means a (field-strength, label) combination is too rare -- "
            "expected with small subgroups and worth noting in the paper's limitations."
        )
        return None


def subject_level_split(
    cohort: pd.DataFrame,
    label_col: str,
    strata_col: str = "FIELD_STRENGTH",
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    rid_col: str = "RID",
) -> CohortSplit:
    """Split unique subjects (never visits) into train/val/test.

    Stratifies jointly on the outcome label and on scanner field strength, so
    scanner-effect performance can be reported per stratum instead of being
    hidden inside one aggregate number. Falls back to label-only
    stratification if a joint stratum is too small to split on.
    """
    rids = cohort[rid_col].to_numpy()
    joint_strata = cohort[strata_col].astype(str) + "_" + cohort[label_col].astype(str)
    label_only = cohort[label_col].astype(str)

    result = _stratified_split(rids, joint_strata, test_size, random_state)
    if result is None:
        result = _stratified_split(rids, label_only, test_size, random_state)
    train_val_rids, test_rids = result

    remaining = cohort[cohort[rid_col].isin(train_val_rids)]
    rel_val_size = val_size / (1 - test_size)
    remaining_joint = remaining[strata_col].astype(str) + "_" + remaining[label_col].astype(str)
    remaining_label = remaining[label_col].astype(str)

    result = _stratified_split(remaining[rid_col].to_numpy(), remaining_joint, rel_val_size, random_state)
    if result is None:
        result = _stratified_split(remaining[rid_col].to_numpy(), remaining_label, rel_val_size, random_state)
    train_rids, val_rids = result

    return CohortSplit(
        train_rids=frozenset(train_rids),
        val_rids=frozenset(val_rids),
        test_rids=frozenset(test_rids),
    )
