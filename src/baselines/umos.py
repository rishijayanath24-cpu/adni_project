"""
UMoS-style uncertainty-driven modality selection.

Re-implementation of the mechanism described in Zheng et al.,
"Uncertainty-driven modality selection for data-efficient prediction of
Alzheimer's disease" (Alzheimer's & Dementia 2022 abstract; IISE
Transactions on Healthcare Systems Engineering 14(1), 2023).

Mechanism: fit a first-stage model on the cheaper modality, take its
PREDICTIVE UNCERTAINTY per patient, and escalate only the uncertain
patients to a second-stage model that also uses the expensive modality.
Certain patients are answered by stage one and never scanned.

Published reference numbers (ADNI, MCI conversion): n = 1,319 with T1-MRI
of whom 612 also had amyloid-PET; AUC 0.851; 46.2% of patients answered by
the MRI-only model and 53.8% escalated to MRI+PET.

DELIBERATE ADAPTATIONS, stated rather than buried:
  * Stage one here is CHEAP+MRI, not MRI alone. Demographics, cognition and
    APOE are already in hand at no acquisition cost in any real clinic, so
    an MRI-only stage would be an artificially weak first stage and would
    flatter the escalation step.
  * This is NOT the original authors' code, which is not public. It is a
    re-implementation from the published description, so its numbers should
    be read as "our implementation of the UMoS mechanism", not as a
    reproduction of their result.

The escalation threshold is set on TRAINING data (either to hit a target
escalation rate or by sweeping for the best validation objective) and then
applied unchanged to test. Choosing it on test would let the baseline peek.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.baselines.base import (
    AcquisitionPolicy,
    BlockModelBank,
    PolicyResult,
    binary_entropy,
    margin_uncertainty,
)

UNCERTAINTY_FNS = {"entropy": binary_entropy, "margin": margin_uncertainty}


class UMoS(AcquisitionPolicy):
    def __init__(
        self,
        stage1_blocks: list[str] | None = None,
        stage2_blocks: list[str] | None = None,
        target_escalation_rate: float | None = 0.538,
        uncertainty: str = "entropy",
    ):
        # Defaults mirror the published setup: MRI first, amyloid-PET second.
        self.stage1_blocks = list(stage1_blocks or ["MRI"])
        self.stage2_blocks = list(stage2_blocks or ["AMYLOID"])
        self.target_escalation_rate = target_escalation_rate
        self.uncertainty_name = uncertainty
        self.uncertainty_fn = UNCERTAINTY_FNS[uncertainty]
        self.threshold_: float | None = None
        self.bank = BlockModelBank()
        rate = "tuned" if target_escalation_rate is None else f"{target_escalation_rate:g}"
        self.name = (
            f"umos[{'+'.join(self.stage1_blocks)}->{'+'.join(self.stage2_blocks)},rate={rate}]"
        )

    # --- internals ---------------------------------------------------------

    def _s1(self) -> list[str]:
        return ["CHEAP", *self.stage1_blocks]

    def _s2(self) -> list[str]:
        return ["CHEAP", *self.stage1_blocks, *self.stage2_blocks]

    def fit(self, train_df: pd.DataFrame, label_col: str) -> "UMoS":
        self.bank.fit(train_df, label_col, [self._s1(), self._s2()])

        # Threshold calibrated on training predictions only.
        p1_train = self.bank.predict_proba(train_df, self._s1())
        u_train = self.uncertainty_fn(p1_train)

        if self.target_escalation_rate is None:
            # Sweep candidate thresholds, keep the one maximising training AUC
            # of the resulting cascade. Still training-only, no test peeking.
            from src.eval.metrics import auc

            y = train_df[label_col].astype(int).to_numpy()
            best, best_auc = None, -np.inf
            for q in np.linspace(0.05, 0.95, 19):
                thr = float(np.quantile(u_train, 1 - q))
                probs = self._apply(train_df, thr)[0]
                a = auc(y, probs)
                if a > best_auc:
                    best, best_auc = thr, a
            self.threshold_ = best
        else:
            # Escalate the most-uncertain `rate` fraction.
            self.threshold_ = float(np.quantile(u_train, 1 - self.target_escalation_rate))
        return self

    def _apply(self, df: pd.DataFrame, threshold: float):
        p1 = self.bank.predict_proba(df, self._s1())
        u = self.uncertainty_fn(p1)
        escalate = u > threshold

        probs = p1.copy()
        if escalate.any():
            probs[escalate] = self.bank.predict_proba(df[escalate], self._s2())

        acquired = [
            list(self.stage1_blocks) + (list(self.stage2_blocks) if e else [])
            for e in escalate
        ]
        return probs, acquired, escalate

    # --- interface ---------------------------------------------------------

    def run(self, test_df: pd.DataFrame, label_col: str) -> PolicyResult:
        if self.threshold_ is None:
            raise RuntimeError("fit() must be called before run()")
        probs, acquired, _ = self._apply(test_df, self.threshold_)
        return self._result(test_df, label_col, probs, acquired)
