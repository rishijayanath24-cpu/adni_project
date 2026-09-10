"""
DQN-style sequential test-ordering policy.

Re-implementation of the mechanism in "Efficient Alzheimer's Diagnosis
Through Sequential Decision-Making with Reinforcement Learning"
(Proc. 2025 7th Int. Conf. on Big-Data Service and Intelligent
Computation, DOI 10.1145/3778265.3778280), which frames diagnosis as a
sequential decision problem on ADNI and reports a DQN reaching 89.2%
accuracy while favouring high-yield cognitive assessments over costly
modalities, and a PPO-XGB variant at 86.93% with 5.06 tests on average.

Formulation used here:
  state   -- cheap features, plus each acquired block's (standardised)
             values, plus a binary mask of what has been acquired
  actions -- acquire one not-yet-acquired block, or STOP and predict
  reward  -- minus the cost of each acquisition; on STOP, a bounded
             accuracy term 1 - 2*|y - p| in [-1, 1]

The cost weight lambda is the knob that traces the frontier: sweeping it
moves the policy from "scan nobody" to "scan everybody", which is how a
single learned policy yields a whole accuracy-vs-cost curve rather than
one point.

DELIBERATE ADAPTATIONS, stated rather than buried:
  * Re-implementation from the published description; the authors' code is
    not public. Read the numbers as "our implementation of the mechanism".
  * PREDICTION is delegated to the shared BlockModelBank rather than learned
    inside the Q-network. This is on purpose: every policy in the benchmark
    then uses the identical predictor for a given acquired set, so
    differences between policies are differences in the ACQUISITION
    DECISION, which is what is being compared. A Q-network that also learned
    to predict would confound the two.
  * The published accuracy figures are not directly comparable to ours: they
    report accuracy on their own cohort and task framing, we report AUC on
    an MCI-conversion cohort at fixed horizons.
"""
from __future__ import annotations

import random
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.baselines.base import AcquisitionPolicy, BlockModelBank, PolicyResult, RANDOM_STATE
from src.models.features import COST_REGIMES, fit_design_matrix

STOP = "STOP"


class QNet(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class StateEncoder:
    """Builds [cheap | per-block values (zeroed when unacquired) | mask].

    Each block gets its own fitted transformer, so a block's contribution can
    be masked out cleanly. Zeroing an unacquired block plus an explicit mask
    bit is what lets one network handle every acquisition state, instead of
    needing a separate network per subset.
    """

    def __init__(self, blocks: list[str]):
        self.blocks = list(blocks)
        self.dm_cheap = None
        self.dm_blocks: dict[str, object] = {}
        self.block_dims: dict[str, int] = {}

    def fit(self, train_df: pd.DataFrame) -> "StateEncoder":
        self.dm_cheap = fit_design_matrix(train_df, ["CHEAP"])
        self.cheap_dim = np.asarray(self.dm_cheap.transform(train_df.head(1))).shape[1]
        for b in self.blocks:
            dm = fit_design_matrix(train_df, [b])
            self.dm_blocks[b] = dm
            self.block_dims[b] = np.asarray(dm.transform(train_df.head(1))).shape[1]
        return self

    @property
    def state_dim(self) -> int:
        return self.cheap_dim + sum(self.block_dims.values()) + len(self.blocks)

    def encode(self, df: pd.DataFrame, masks: np.ndarray) -> np.ndarray:
        """masks: (n_subjects, n_blocks) of 0/1 for acquired."""
        parts = [np.asarray(self.dm_cheap.transform(df), dtype=np.float32)]
        for j, b in enumerate(self.blocks):
            vals = np.asarray(self.dm_blocks[b].transform(df), dtype=np.float32)
            parts.append(vals * masks[:, [j]].astype(np.float32))
        parts.append(masks.astype(np.float32))
        return np.concatenate(parts, axis=1)

    def precompute(self, df: pd.DataFrame) -> dict:
        """Transform a whole dataframe ONCE.

        Calling sklearn transform per row inside the RL loop is what made
        training intractable (thousands of single-row transforms per
        episode). Everything downstream indexes these arrays instead.
        """
        return {
            "cheap": np.asarray(self.dm_cheap.transform(df), dtype=np.float32),
            "blocks": {
                b: np.asarray(self.dm_blocks[b].transform(df), dtype=np.float32)
                for b in self.blocks
            },
        }

    def state_from_pre(self, pre: dict, i: int, mask: np.ndarray) -> np.ndarray:
        """Assemble one state vector with pure numpy (no sklearn call)."""
        parts = [pre["cheap"][i]]
        for j, b in enumerate(self.blocks):
            parts.append(pre["blocks"][b][i] * np.float32(mask[j]))
        parts.append(mask.astype(np.float32))
        return np.concatenate(parts)


class DQNPolicy(AcquisitionPolicy):
    def __init__(
        self,
        blocks: list[str],
        cost_weight: float = 0.05,
        cost_regime: str = "price_like",
        episodes: int = 4000,
        hidden: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 20000,
        batch_size: int = 64,
        target_sync: int = 200,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        random_state: int = RANDOM_STATE,
    ):
        self.blocks = list(blocks)
        self.cost_weight = float(cost_weight)
        self.cost_regime = cost_regime
        self.episodes = episodes
        self.hidden = hidden
        self.lr = lr
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_sync = target_sync
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.random_state = random_state
        self.name = f"dqn[lambda={self.cost_weight:g}]"

        self.encoder = StateEncoder(self.blocks)
        self.bank = BlockModelBank()
        self.qnet: QNet | None = None

    # actions: 0..n_blocks-1 acquire that block, n_blocks = STOP
    @property
    def n_actions(self) -> int:
        return len(self.blocks) + 1

    def _all_block_sets(self) -> list[list[str]]:
        from itertools import combinations

        sets = [["CHEAP"]]
        for r in range(1, len(self.blocks) + 1):
            for combo in combinations(self.blocks, r):
                sets.append(["CHEAP", *combo])
        return sets

    def _mask_key(self, mask: np.ndarray) -> tuple[str, ...]:
        acquired = [b for j, b in enumerate(self.blocks) if mask[j]]
        return tuple(sorted(["CHEAP", *acquired]))

    def _precompute_predictions(self, df: pd.DataFrame) -> dict:
        """Predicted probability for every subject under every acquisition subset.

        Computed once per dataframe. Calling predict_proba per row inside the
        RL loop was half of what made training intractable.
        """
        table = {}
        for blocks in self._all_block_sets():
            table[tuple(sorted(blocks))] = self.bank.predict_proba(df, list(blocks))
        return table

    def _acquire_cost(self, block: str) -> float:
        return float(COST_REGIMES[self.cost_regime][block])

    def fit(self, train_df: pd.DataFrame, label_col: str) -> "DQNPolicy":
        rng = random.Random(self.random_state)
        np_rng = np.random.default_rng(self.random_state)
        torch.manual_seed(self.random_state)

        self.encoder.fit(train_df)
        self.bank.fit(train_df, label_col, self._all_block_sets())

        pre = self.encoder.precompute(train_df)
        pred = self._precompute_predictions(train_df)
        y_all = train_df[label_col].astype(int).to_numpy()
        n_train = len(train_df)
        n_b = len(self.blocks)

        self.qnet = QNet(self.encoder.state_dim, self.n_actions, self.hidden)
        target = QNet(self.encoder.state_dim, self.n_actions, self.hidden)
        target.load_state_dict(self.qnet.state_dict())
        opt = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)

        buf_s, buf_a, buf_r, buf_ns, buf_done, buf_legal = [], [], [], [], [], []

        def terminal_reward(i: int, mask: np.ndarray) -> float:
            p = float(pred[self._mask_key(mask)][i])
            # Bounded accuracy-style reward: +1 confident-correct,
            # -1 confident-wrong, 0 for a coin flip.
            return 1.0 - 2.0 * abs(float(y_all[i]) - p)

        def legal_vector(mask: np.ndarray) -> np.ndarray:
            v = np.zeros(self.n_actions, dtype=np.float32)
            v[:n_b] = 1.0 - mask  # can only acquire what is not yet acquired
            v[n_b] = 1.0          # STOP always legal
            return v

        steps = 0
        for ep in range(self.episodes):
            i = int(np_rng.integers(0, n_train))
            mask = np.zeros(n_b, dtype=np.float32)
            eps = self.eps_end + (self.eps_start - self.eps_end) * max(
                0.0, 1.0 - ep / (0.7 * self.episodes)
            )

            while True:
                state = self.encoder.state_from_pre(pre, i, mask)
                legal = legal_vector(mask)

                if rng.random() < eps:
                    action = int(rng.choice(list(np.flatnonzero(legal))))
                else:
                    with torch.no_grad():
                        q = self.qnet(torch.from_numpy(state)).numpy()
                    action = int(np.argmax(np.where(legal > 0, q, -np.inf)))

                if action == n_b:  # STOP
                    buf_s.append(state)
                    buf_a.append(action)
                    buf_r.append(terminal_reward(i, mask))
                    buf_ns.append(np.zeros_like(state))
                    buf_done.append(1.0)
                    buf_legal.append(np.ones(self.n_actions, dtype=np.float32))
                    break

                block = self.blocks[action]
                new_mask = mask.copy()
                new_mask[action] = 1.0
                next_state = self.encoder.state_from_pre(pre, i, new_mask)

                buf_s.append(state)
                buf_a.append(action)
                buf_r.append(-self.cost_weight * self._acquire_cost(block))
                buf_ns.append(next_state)
                buf_done.append(0.0)
                buf_legal.append(legal_vector(new_mask))
                mask = new_mask

                if mask.sum() == n_b:
                    # Everything acquired: forced STOP so the episode ends.
                    buf_s.append(next_state)
                    buf_a.append(n_b)
                    buf_r.append(terminal_reward(i, mask))
                    buf_ns.append(np.zeros_like(state))
                    buf_done.append(1.0)
                    buf_legal.append(np.ones(self.n_actions, dtype=np.float32))
                    break

            if len(buf_s) > self.buffer_size:
                cut = len(buf_s) - self.buffer_size
                del buf_s[:cut], buf_a[:cut], buf_r[:cut]
                del buf_ns[:cut], buf_done[:cut], buf_legal[:cut]

            if len(buf_s) >= self.batch_size:
                idx = np_rng.integers(0, len(buf_s), self.batch_size)
                s_b = torch.from_numpy(np.stack([buf_s[k] for k in idx]))
                a_b = torch.tensor([buf_a[k] for k in idx], dtype=torch.long)
                r_b = torch.tensor([buf_r[k] for k in idx], dtype=torch.float32)
                ns_b = torch.from_numpy(np.stack([buf_ns[k] for k in idx]))
                done_b = torch.tensor([buf_done[k] for k in idx], dtype=torch.float32)
                legal_b = torch.from_numpy(np.stack([buf_legal[k] for k in idx]))

                with torch.no_grad():
                    qn = target(ns_b).masked_fill(legal_b <= 0, float("-inf"))
                    best = torch.nan_to_num(qn.max(dim=1).values, neginf=0.0)
                    targets = r_b + (1.0 - done_b) * self.gamma * best

                q_pred = self.qnet(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(q_pred, targets)
                opt.zero_grad()
                loss.backward()
                opt.step()

                steps += 1
                if steps % self.target_sync == 0:
                    target.load_state_dict(self.qnet.state_dict())

        self.qnet.eval()
        return self

    def run(self, test_df: pd.DataFrame, label_col: str) -> PolicyResult:
        if self.qnet is None:
            raise RuntimeError("fit() must be called before run()")

        pre = self.encoder.precompute(test_df)
        pred = self._precompute_predictions(test_df)
        n_b = len(self.blocks)

        probs = np.zeros(len(test_df))
        acquired: list[list[str]] = []

        for i in range(len(test_df)):
            mask = np.zeros(n_b, dtype=np.float32)
            while True:
                state = self.encoder.state_from_pre(pre, i, mask)
                legal = np.zeros(self.n_actions, dtype=np.float32)
                legal[:n_b] = 1.0 - mask
                legal[n_b] = 1.0
                with torch.no_grad():
                    q = self.qnet(torch.from_numpy(state)).numpy()
                action = int(np.argmax(np.where(legal > 0, q, -np.inf)))
                if action == n_b or mask.sum() == n_b:
                    break
                mask[action] = 1.0

            got = [b for j, b in enumerate(self.blocks) if mask[j]]
            probs[i] = float(pred[self._mask_key(mask)][i])
            acquired.append(got)

        return self._result(test_df, label_col, probs, acquired)
