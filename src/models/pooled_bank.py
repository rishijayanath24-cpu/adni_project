"""
Pooled block-model bank: fit each acquisition state on everyone who has it.

THE OBSERVATION. The benchmark's BlockModelBank fits every subset model on
the same complete-case cohort -- the subjects who hold ALL blocks in play.
But the model for state "CHEAP+MRI" does not need amyloid PET or CSF. It
only needs MRI. Restricting it to complete cases discards every subject who
has an MRI but no PET, which in ADNI is a large and non-random group.

At the 24-month horizon: the complete-case cohort is 365 subjects, but 721
labelled subjects have MRI, 569 have MRI+FDG, and so on. Each subset model
is being fitted on roughly half the data available to it.

WHY THIS IS THE RIGHT FIX RATHER THAN A BIGGER MODEL. An earlier attempt
replaced the whole bank with one NaN-native gradient-boosting model so it
could train on every subject at once. The extra data did help (+0.006), but
gradient boosting is worse than logistic regression at this sample size
(-0.020), so the net was negative. Pooling keeps the sample-efficient model
class and takes the data gain anyway: same estimator, same policies, same
evaluation cohort, more training rows per state.

LEAKAGE. The pool is filtered against the fold's evaluation subjects and
against the locked test set before fitting, so a subject is never trained on
and scored in the same fold.

HONEST CAVEAT. Different acquisition states are now fitted on different,
overlapping subject sets, so the models are no longer strictly comparable to
one another -- a state fitted on more (and differently-selected) subjects
may be better for reasons other than the modality it adds. This is a
selection effect worth reporting, and it is why the EVALUATION cohort stays
fixed at the complete cases: only the training data changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.models.features import BLOCK_AVAILABILITY, fit_design_matrix

RANDOM_STATE = 42


class PooledBlockModelBank:
    """BlockModelBank-compatible, but each state trains on all who hold it."""

    def __init__(
        self,
        extra_pool: pd.DataFrame | None = None,
        random_state: int = RANDOM_STATE,
        C: float = 1.0,
        min_train: int = 40,
    ):
        self.extra_pool = extra_pool
        self.random_state = random_state
        self.C = C
        self.min_train = min_train
        self._models: dict[tuple[str, ...], tuple] = {}
        self.train_sizes: dict[tuple[str, ...], int] = {}

    def _pool_for(self, blocks: list[str], exclude_rids: set) -> pd.DataFrame | None:
        """Subjects holding every block in this state, minus excluded ones."""
        if self.extra_pool is None or not len(self.extra_pool):
            return None
        pool = self.extra_pool[~self.extra_pool["RID"].isin(exclude_rids)]
        for b in blocks:
            if b == "CHEAP":
                continue
            flag = BLOCK_AVAILABILITY.get(b)
            if flag is not None:
                pool = pool[pool[flag].astype(bool)]
        return pool

    def fit(self, train_df: pd.DataFrame, label_col: str, block_sets: list[list[str]]) -> None:
        # Anything not in this fold's training rows is a scoring subject for
        # this fold and must never enter training, even via the pool.
        train_rids = set(train_df["RID"])
        for blocks in block_sets:
            key = tuple(sorted(blocks))
            if key in self._models:
                continue

            parts = [train_df]
            pool = self._pool_for(list(blocks), exclude_rids=set())
            if pool is not None and len(pool):
                pool = pool[~pool["RID"].isin(train_rids)]
                if len(pool):
                    parts.append(pool)
            train = pd.concat(parts, ignore_index=True) if len(parts) > 1 else train_df
            train = train[train[label_col].notna()]

            y = train[label_col].astype(int).to_numpy()
            if len(train) < self.min_train or len(np.unique(y)) < 2:
                train, y = train_df, train_df[label_col].astype(int).to_numpy()

            dm = fit_design_matrix(train, list(blocks))
            model = LogisticRegression(max_iter=2000, C=self.C, random_state=self.random_state)
            model.fit(dm.transform(train), y)
            self._models[key] = (dm, model)
            self.train_sizes[key] = len(train)

    def predict_proba(self, df: pd.DataFrame, blocks: list[str]) -> np.ndarray:
        key = tuple(sorted(blocks))
        if key not in self._models:
            raise KeyError(f"no model fitted for block set {key}")
        dm, model = self._models[key]
        return model.predict_proba(dm.transform(df))[:, 1]

    @property
    def fitted_sets(self):
        return list(self._models)
