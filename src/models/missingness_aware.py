"""
Missingness-native predictor: one model for every acquisition state.

WHY THIS SHOULD WIN, stated as a mechanism rather than a hope.

Every policy in the benchmark predicts through a bank of subset-specific
models (cheap, cheap+MRI, cheap+MRI+amyloid, ...). Each of those can only be
fitted on subjects who actually hold that subset, so the whole comparison
runs on the complete-case cohort: 365 subjects and 67 converters at 24
months, out of 904 labelled subjects and 237 converters. Roughly 70% of the
converters are discarded before any policy is even trained.

This class removes that constraint. Gradient boosting handles NaN natively,
so a single model can be fitted on EVERY labelled subject, with each
subject's absent modalities simply left missing, and can then be queried for
any acquisition state by masking the un-acquired blocks to NaN. At 24 months
that is 812 training subjects and 220 converters -- 3.3x the converters the
subset models see.

TRAIN/INFERENCE MASK MISMATCH, and why the augmentation is not optional.
ADNI's own missingness is protocol-driven: whether a subject got amyloid PET
depends on which phase enrolled them. At inference, though, the mask is
whatever the POLICY chose. A model fitted only on protocol-driven masks
would be evaluated under a different mask distribution than it was trained
on. Random modality-masking augmentation fixes this: every training subject
is presented several times with random blocks additionally masked out, so
the model learns p(y | arbitrary observed subset) rather than
p(y | the subset ADNI happened to collect).

Missingness INDICATORS are deliberately excluded as features. In ADNI,
"has amyloid PET" proxies for enrolment phase, which correlates with
outcome; a model given that flag can exploit a shortcut that will not exist
when a policy is choosing what to acquire. Leaving it out costs some
apparent accuracy and buys a result that means what it appears to mean.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.models.features import BLOCKS, CATEGORICAL_FEATURES, feature_columns


class MissingnessAwareBank:
    """Drop-in replacement for BlockModelBank backed by one NaN-native model.

    Exposes the same fit()/predict_proba() surface as BlockModelBank so the
    existing policies and benchmark harness work unchanged -- which is what
    keeps the comparison apples-to-apples: same policies, same folds, same
    evaluation cohort, only the predictor differs.
    """

    def __init__(
        self,
        extra_pool: pd.DataFrame | None = None,
        all_blocks: list[str] | None = None,
        n_mask_augment: int = 4,
        random_state: int = 42,
        max_depth: int = 3,
        max_iter: int = 200,
        learning_rate: float = 0.05,
    ):
        self.extra_pool = extra_pool
        self.all_blocks = list(all_blocks or ["MRI", "FDG", "AMYLOID", "CSF"])
        self.n_mask_augment = n_mask_augment
        self.random_state = random_state
        self.max_depth = max_depth
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self._model = None
        self._columns: list[str] = []
        self._block_cols: dict[str, list[str]] = {}

    # --- feature handling --------------------------------------------------

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Numeric frame with NaNs preserved (no imputation -- HGB uses them)."""
        x = df[self._columns].copy()
        for c in self._columns:
            if c in CATEGORICAL_FEATURES:
                x[c] = (x[c].astype(str) == "Male").astype(float)
            else:
                x[c] = pd.to_numeric(x[c], errors="coerce")
        return x

    def _mask_to(self, x: pd.DataFrame, keep_blocks: list[str]) -> pd.DataFrame:
        """NaN out every block the policy has not acquired."""
        x = x.copy()
        for b in self.all_blocks:
            if b not in keep_blocks:
                x[self._block_cols[b]] = np.nan
        return x

    # --- BlockModelBank-compatible API -------------------------------------

    def fit(self, train_df: pd.DataFrame, label_col: str, block_sets=None) -> None:
        """block_sets is accepted and ignored: one model serves every subset."""
        self._columns = feature_columns(["CHEAP", *self.all_blocks])
        self._block_cols = {b: list(BLOCKS[b]) for b in self.all_blocks}

        train = train_df
        if self.extra_pool is not None and len(self.extra_pool):
            # Never allow a subject being scored in this fold to leak into
            # training through the extra pool.
            pool = self.extra_pool[~self.extra_pool["RID"].isin(train_df["RID"])]
            train = pd.concat([train_df, pool], ignore_index=True)
        train = train[train[label_col].notna()]

        x = self._encode(train)
        y = train[label_col].astype(int).to_numpy()

        rng = np.random.default_rng(self.random_state)
        xs, ys = [x], [y]
        for _ in range(self.n_mask_augment):
            xa = x.copy()
            # Mask each block independently with p=0.5, so the model sees the
            # full lattice of acquisition states rather than only the ones
            # ADNI's protocols happened to produce.
            for b in self.all_blocks:
                drop = rng.random(len(xa)) < 0.5
                xa.loc[drop, self._block_cols[b]] = np.nan
            xs.append(xa)
            ys.append(y)

        x_aug = pd.concat(xs, ignore_index=True)
        y_aug = np.concatenate(ys)

        self._model = HistGradientBoostingClassifier(
            max_depth=self.max_depth,
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )
        self._model.fit(x_aug, y_aug)

    def predict_proba(self, df: pd.DataFrame, blocks: list[str]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("fit() must be called before predict_proba()")
        keep = [b for b in blocks if b != "CHEAP"]
        x = self._mask_to(self._encode(df), keep)
        return self._model.predict_proba(x)[:, 1]

    @property
    def fitted_sets(self):
        return [("ANY",)]
