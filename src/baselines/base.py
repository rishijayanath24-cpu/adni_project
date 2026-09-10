"""
Common interface for sequential modality-acquisition policies.

Every baseline and every proposed policy implements the same contract, so
they can be placed on one accuracy-vs-cost frontier and compared honestly:

    fit(train_df, label_col)          -- learn from training subjects only
    run(test_df, label_col)           -- decide what to acquire per subject,
                                         then predict

What `run` returns is deliberately richer than a prediction vector. It also
reports WHICH blocks each subject ended up acquiring, because that is the
object of study here: the fairness audit asks whether acquisition decisions
differ across subgroups, which is unanswerable from predictions alone.

EVALUATION CONSTRAINT: a retrospective simulation can only "acquire" a
block a subject actually has. Policies are therefore compared on a
complete-case cohort for the blocks in play, so that every policy could in
principle have acquired anything -- otherwise a policy that happens to
prefer the most-available modality wins for the wrong reason.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.models.features import COST_REGIMES, fit_design_matrix

RANDOM_STATE = 42


@dataclass
class PolicyResult:
    """Per-subject outcome of running a policy."""

    name: str
    rids: np.ndarray
    y_true: np.ndarray
    y_prob: np.ndarray
    acquired: list[list[str]] = field(default_factory=list)

    def cost(self, regime: str = "price_like") -> np.ndarray:
        costs = COST_REGIMES[regime]
        return np.array([sum(costs[b] for b in blocks) for blocks in self.acquired], dtype=float)

    def acquisition_rate(self, block: str) -> float:
        if not len(self.acquired):
            return float("nan")
        return float(np.mean([block in blocks for blocks in self.acquired]))


class BlockModelBank:
    """Fits and caches one predictor per block-subset.

    Policies need a predictor for every combination of blocks they might end
    up holding (cheap-only, cheap+MRI, cheap+MRI+amyloid, ...). Fitting them
    once here keeps the policies themselves about the DECISION rule, which is
    the contribution, rather than about model plumbing.

    Every model's preprocessing is fitted on the training rows only.
    """

    def __init__(self, random_state: int = RANDOM_STATE, C: float = 1.0):
        self.random_state = random_state
        self.C = C
        self._models: dict[tuple[str, ...], tuple] = {}

    def fit(self, train_df: pd.DataFrame, label_col: str, block_sets: list[list[str]]) -> None:
        y = train_df[label_col].astype(int).to_numpy()
        for blocks in block_sets:
            key = tuple(sorted(blocks))
            if key in self._models:
                continue
            dm = fit_design_matrix(train_df, list(blocks))
            model = LogisticRegression(max_iter=2000, C=self.C, random_state=self.random_state)
            model.fit(dm.transform(train_df), y)
            self._models[key] = (dm, model)

    def predict_proba(self, df: pd.DataFrame, blocks: list[str]) -> np.ndarray:
        key = tuple(sorted(blocks))
        if key not in self._models:
            raise KeyError(f"no model fitted for block set {key}")
        dm, model = self._models[key]
        return model.predict_proba(dm.transform(df))[:, 1]

    @property
    def fitted_sets(self) -> list[tuple[str, ...]]:
        return list(self._models)


def binary_entropy(p: np.ndarray) -> np.ndarray:
    """Predictive entropy of a Bernoulli prediction -- the usual uncertainty
    signal for threshold-gated cascades (UMoS-style)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def margin_uncertainty(p: np.ndarray) -> np.ndarray:
    """Distance from the decision boundary, rescaled so larger = less certain."""
    return 1.0 - np.abs(2.0 * np.asarray(p, dtype=float) - 1.0)


class AcquisitionPolicy(ABC):
    name: str = "policy"

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, label_col: str) -> "AcquisitionPolicy":
        ...

    @abstractmethod
    def run(self, test_df: pd.DataFrame, label_col: str) -> PolicyResult:
        ...

    def _result(self, test_df, label_col, y_prob, acquired) -> PolicyResult:
        return PolicyResult(
            name=self.name,
            rids=test_df["RID"].to_numpy(),
            y_true=test_df[label_col].astype(int).to_numpy(),
            y_prob=np.asarray(y_prob, dtype=float),
            acquired=acquired,
        )
