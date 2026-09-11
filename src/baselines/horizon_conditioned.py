"""
Horizon-conditioned acquisition policy -- this project's own contribution,
as opposed to the three re-implemented baselines.

MOTIVATION, and it is empirical rather than assumed. Experiment 2 found, on
a fixed subject set with a paired delta-of-deltas bootstrap, that amyloid
PET's incremental value RISES with the prediction horizon (+0.014 at 12
months to +0.068 at 36; interaction +0.054, 95% CI [+0.014, +0.095]) while
MRI's does not (interaction -0.019, CI [-0.057, +0.016], n.s.). Amyloid is
the only modality whose horizon dependence survives a formal test.

If that effect is real and exploitable, a policy that knows the prediction
horizon should escalate differently: reach for MRI when asked for a
12-month risk, and for amyloid PET when asked for a 36-month risk. Every
published policy audited here is horizon-agnostic -- it escalates the same
way whatever question it is being asked.

This class makes that concrete: the escalation LADDER is a function of the
horizon. It is deliberately simple, because the point is to test whether
the horizon signal is exploitable at all, not to win by having a bigger
model. If a simple horizon-conditioned ladder cannot beat a
horizon-agnostic one, a more elaborate conditioning scheme is unlikely to.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.baselines.base import (
    AcquisitionPolicy,
    BlockModelBank,
    PolicyResult,
    binary_entropy,
)

# Ladder per horizon, ordered by the measured incremental value at that
# horizon (Findings section 1): MRI leads at 12 months, the molecular
# markers lead at 36.
DEFAULT_LADDERS = {
    12: ["MRI", "FDG"],
    24: ["MRI", "FDG", "AMYLOID"],
    36: ["AMYLOID", "FDG", "MRI"],
}


class HorizonConditionedPolicy(AcquisitionPolicy):
    def __init__(
        self,
        horizon: int,
        ladders: dict[int, list[str]] | None = None,
        target_escalation_rate: float = 0.5,
        random_state: int = 42,
    ):
        self.horizon = int(horizon)
        self.ladders = ladders or DEFAULT_LADDERS
        self.ladder = list(self.ladders[self.horizon])
        self.target_escalation_rate = float(target_escalation_rate)
        self.random_state = random_state
        self.bank = BlockModelBank()
        self._thresholds: list[float] = []
        self.name = f"horizon_cond[{self.horizon}mo:{'->'.join(self.ladder)}]"

    def _prefixes(self) -> list[list[str]]:
        sets = [["CHEAP"]]
        for i in range(1, len(self.ladder) + 1):
            sets.append(["CHEAP", *self.ladder[:i]])
        return sets

    def fit(self, train_df: pd.DataFrame, label_col: str) -> "HorizonConditionedPolicy":
        self.bank.fit(train_df, label_col, self._prefixes())

        # One uncertainty threshold per rung, each calibrated on training
        # predictions only so no threshold is chosen on scored data.
        self._thresholds = []
        acquired_so_far: list[str] = []
        for _ in self.ladder:
            p = self.bank.predict_proba(train_df, ["CHEAP", *acquired_so_far])
            u = binary_entropy(p)
            self._thresholds.append(float(np.quantile(u, 1.0 - self.target_escalation_rate)))
            acquired_so_far.append(self.ladder[len(acquired_so_far)])
        return self

    def run(self, test_df: pd.DataFrame, label_col: str) -> PolicyResult:
        if not self._thresholds:
            raise RuntimeError("fit() must be called before run()")

        n = len(test_df)
        acquired: list[list[str]] = [[] for _ in range(n)]
        probs = self.bank.predict_proba(test_df, ["CHEAP"])

        for rung, (block, thr) in enumerate(zip(self.ladder, self._thresholds)):
            u = binary_entropy(probs)
            # Only subjects who escalated through every previous rung are
            # eligible for this one -- escalation is a cascade, not a menu.
            eligible = np.array([len(a) == rung for a in acquired])
            esc = (u > thr) & eligible
            if not esc.any():
                continue
            idx = np.flatnonzero(esc)
            sub = test_df.iloc[idx]
            probs[idx] = self.bank.predict_proba(sub, ["CHEAP", *self.ladder[: rung + 1]])
            for i in idx:
                acquired[i] = list(self.ladder[: rung + 1])

        return self._result(test_df, label_col, probs, acquired)
