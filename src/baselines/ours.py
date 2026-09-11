"""
Our configuration, as a first-class policy on the benchmark frontier.

Not a new decision rule. The audit found that clever decision rules do not
beat fixed ones, so this deliberately keeps the DUMBEST possible policy --
acquire a fixed set for everyone -- and changes only what that policy is
built on:

  * the AD-signature MRI feature block rather than three volumetrics, and
  * pooled training, fitting each acquisition state on every subject who
    holds it rather than on complete cases only.

Keeping the policy trivial is the point. It means any gain is attributable
to the inputs, and it sidesteps the disparity risk that patient-dependent
policies carry (a fixed rule cannot allocate tests unequally by subgroup,
because it does not look at the patient).
"""
from __future__ import annotations

import pandas as pd

from src.baselines.base import AcquisitionPolicy, PolicyResult
from src.models.pooled_bank import PooledBlockModelBank


class OursFixed(AcquisitionPolicy):
    def __init__(self, blocks: list[str], extra_pool: pd.DataFrame | None = None,
                 random_state: int = 42):
        self.blocks = list(blocks)
        self.extra_pool = extra_pool
        self.random_state = random_state
        self.bank = PooledBlockModelBank(extra_pool=extra_pool, random_state=random_state)
        self.name = f"ours[{'+'.join(self.blocks) if self.blocks else 'CHEAP'}]"

    def fit(self, train_df, label_col):
        self.bank.fit(train_df, label_col, [["CHEAP", *self.blocks]])
        return self

    def run(self, test_df, label_col) -> PolicyResult:
        p = self.bank.predict_proba(test_df, ["CHEAP", *self.blocks])
        return self._result(test_df, label_col, p,
                            [list(self.blocks) for _ in range(len(test_df))])
