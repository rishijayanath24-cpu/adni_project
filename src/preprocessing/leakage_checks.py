"""
Standalone leakage assertions, meant to be run immediately after any split
is created and again immediately before any model is trained.
"""
from __future__ import annotations

import pandas as pd

from src.preprocessing.splits import CohortSplit


def assert_no_subject_leakage(split: CohortSplit) -> None:
    """Raise if any subject appears in more than one of train/val/test.

    CohortSplit already checks this at construction time; this function
    exists so it can be re-asserted right before training, after the split
    has been round-tripped through a dataframe filter/merge -- exactly the
    kind of step that has silently reintroduced leakage in published ADNI
    studies.
    """
    train_val = split.train_rids & split.val_rids
    train_test = split.train_rids & split.test_rids
    val_test = split.val_rids & split.test_rids
    if train_val or train_test or val_test:
        raise AssertionError(
            f"Subject-level leakage detected: "
            f"train∩val={len(train_val)}, train∩test={len(train_test)}, val∩test={len(val_test)}"
        )


def assert_single_split_membership(df: pd.DataFrame, split_col: str, rid_col: str = "RID") -> None:
    """Raise if any subject (RID) has rows tagged with more than one split label.

    Catches the visit-level-leak failure mode directly: a subject whose
    baseline visit ended up in "train" but whose month-24 visit ended up in
    "test".
    """
    counts = df.groupby(rid_col)[split_col].nunique()
    offenders = counts[counts > 1]
    if len(offenders) > 0:
        raise AssertionError(
            f"{len(offenders)} subject(s) have rows in more than one split: {offenders.index.tolist()[:5]}..."
        )
