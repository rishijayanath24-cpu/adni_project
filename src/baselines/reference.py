"""
Reference policies that bound the frontier.

These are not from the literature -- they are the endpoints and null
comparisons any learned policy must beat to be worth anything:

  CheapOnly    -- never acquires. The "scan nobody" floor.
  AcquireAll   -- always acquires everything. The "scan everyone" ceiling,
                  and the implicit assumption behind most multimodal fusion
                  papers.
  FixedOrder   -- acquires a fixed prefix of blocks for every patient,
                  ignoring the patient. Separates "this many tests help"
                  from "choosing per patient helps"; a learned policy that
                  cannot beat FixedOrder at matched cost has not
                  demonstrated personalisation, only that more data helps.
  RandomSubset -- acquires a random subset at a matched rate. Controls for
                  the possibility that any escalation at all looks good.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.baselines.base import AcquisitionPolicy, BlockModelBank, PolicyResult, RANDOM_STATE


class CheapOnly(AcquisitionPolicy):
    name = "cheap_only"

    def __init__(self):
        self.bank = BlockModelBank()

    def fit(self, train_df, label_col):
        self.bank.fit(train_df, label_col, [["CHEAP"]])
        return self

    def run(self, test_df, label_col) -> PolicyResult:
        p = self.bank.predict_proba(test_df, ["CHEAP"])
        return self._result(test_df, label_col, p, [[] for _ in range(len(test_df))])


class AcquireAll(AcquisitionPolicy):
    name = "acquire_all"

    def __init__(self, blocks: list[str]):
        self.blocks = list(blocks)
        self.bank = BlockModelBank()

    def fit(self, train_df, label_col):
        self.bank.fit(train_df, label_col, [["CHEAP", *self.blocks]])
        return self

    def run(self, test_df, label_col) -> PolicyResult:
        p = self.bank.predict_proba(test_df, ["CHEAP", *self.blocks])
        return self._result(test_df, label_col, p, [list(self.blocks) for _ in range(len(test_df))])


class FixedOrder(AcquisitionPolicy):
    """Acquire the first `k` blocks of a fixed order, for every patient."""

    def __init__(self, order: list[str], k: int):
        self.order = list(order)
        self.k = int(k)
        self.blocks = self.order[: self.k]
        self.name = f"fixed_{'+'.join(self.blocks) if self.blocks else 'none'}"
        self.bank = BlockModelBank()

    def fit(self, train_df, label_col):
        self.bank.fit(train_df, label_col, [["CHEAP", *self.blocks]])
        return self

    def run(self, test_df, label_col) -> PolicyResult:
        p = self.bank.predict_proba(test_df, ["CHEAP", *self.blocks])
        return self._result(test_df, label_col, p, [list(self.blocks) for _ in range(len(test_df))])


class RandomSubset(AcquisitionPolicy):
    """Acquire a random subset of blocks, sized to match a target rate."""

    def __init__(self, blocks: list[str], rate: float = 0.5, random_state: int = RANDOM_STATE):
        self.blocks = list(blocks)
        self.rate = float(rate)
        self.random_state = random_state
        self.name = f"random_rate{self.rate:g}"
        self.bank = BlockModelBank()

    def fit(self, train_df, label_col):
        sets = [["CHEAP"]]
        # Every subset the policy could sample must have a fitted model.
        for r in range(1, len(self.blocks) + 1):
            from itertools import combinations

            for combo in combinations(self.blocks, r):
                sets.append(["CHEAP", *combo])
        self.bank.fit(train_df, label_col, sets)
        return self

    def run(self, test_df, label_col) -> PolicyResult:
        rng = np.random.default_rng(self.random_state)
        probs = np.zeros(len(test_df))
        acquired: list[list[str]] = []
        for i in range(len(test_df)):
            chosen = [b for b in self.blocks if rng.random() < self.rate]
            row = test_df.iloc[[i]]
            probs[i] = self.bank.predict_proba(row, ["CHEAP", *chosen])[0]
            acquired.append(chosen)
        return self._result(test_df, label_col, probs, acquired)
