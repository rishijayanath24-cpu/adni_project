"""
ProMUSE-style progressive, uncertainty-guided staged evidential
classification.

Re-implementation of the mechanism described in "ProMUSE: Progressive
Multi-modal Uncertainty-guided Staged Evidential Alzheimer Disease
Classification" (arXiv:2606.19371, 2026), which reports 50-90% reductions
in MRI/PET usage on ADNI/AIBL/OASIS at competitive accuracy.

Mechanism, in three parts:

 1. EVIDENTIAL CLASSIFICATION per modality. Instead of a softmax
    probability, each modality's network emits non-negative evidence e_k,
    giving a Dirichlet posterior alpha_k = e_k + 1. With S = sum(alpha),
    belief b_k = e_k / S and uncertainty mass u = K / S. Unlike a softmax,
    this can say "little evidence either way" rather than being forced to a
    confident-looking 0.5.
    (Sensoy et al., Evidential Deep Learning, NeurIPS 2018.)

 2. STAGED ESCALATION. Start from the cheap block. If u exceeds the
    threshold, acquire the next modality and fuse; repeat down the ladder.
    Patients whose evidence is already decisive are never scanned.

 3. DEMPSTER-SHAFER FUSION of modality opinions rather than feature
    concatenation, so a confident modality dominates an uncertain one
    instead of both being averaged into mush.

DELIBERATE ADAPTATIONS, stated rather than buried:
  * Re-implementation from the published description; the authors' code is
    not public. Numbers here are "our implementation of the ProMUSE
    mechanism", not a reproduction of their reported result.
  * Small MLPs with weight decay and a fixed epoch budget, because the
    cohort is a few hundred subjects per horizon -- a larger network would
    overfit and would not test the mechanism.
  * Escalation thresholds are fitted on TRAINING data only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.baselines.base import AcquisitionPolicy, BlockModelBank, PolicyResult, RANDOM_STATE
from src.models.features import fit_design_matrix

K_CLASSES = 2


class EvidentialMLP(nn.Module):
    """Small MLP emitting non-negative Dirichlet evidence."""

    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, K_CLASSES),
        )

    def forward(self, x):
        # softplus keeps evidence non-negative; alpha = evidence + 1
        return torch.nn.functional.softplus(self.net(x))


def evidential_loss(evidence: torch.Tensor, y_onehot: torch.Tensor, kl_weight: float) -> torch.Tensor:
    """Expected-MSE evidential loss with an annealed KL regulariser.

    The KL term pulls evidence for WRONG classes toward zero, which is what
    makes the uncertainty mass meaningful rather than a rescaled softmax
    confidence.
    """
    alpha = evidence + 1.0
    S = alpha.sum(dim=1, keepdim=True)
    p_hat = alpha / S

    err = (y_onehot - p_hat).pow(2).sum(dim=1)
    var = (alpha * (S - alpha) / (S * S * (S + 1.0))).sum(dim=1)

    # KL(Dir(alpha_tilde) || Dir(1)), with evidence removed from the true class
    alpha_tilde = y_onehot + (1.0 - y_onehot) * alpha
    S_t = alpha_tilde.sum(dim=1, keepdim=True)
    kl = (
        torch.lgamma(S_t.squeeze(1))
        - torch.lgamma(torch.tensor(float(K_CLASSES)))
        - torch.lgamma(alpha_tilde).sum(dim=1)
        + ((alpha_tilde - 1.0) * (torch.digamma(alpha_tilde) - torch.digamma(S_t))).sum(dim=1)
    )
    return (err + var + kl_weight * kl).mean()


def opinion_from_evidence(evidence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert evidence to (belief per class, uncertainty mass)."""
    alpha = evidence + 1.0
    S = alpha.sum(axis=1, keepdims=True)
    belief = evidence / S
    uncertainty = float(K_CLASSES) / S.squeeze(1)
    return belief, uncertainty


def dempster_shafer_fuse(b1, u1, b2, u2):
    """Dempster's rule of combination for two binary opinions.

    C is the conflict mass between the two opinions; dividing by (1 - C)
    renormalises after discarding it. A modality with low uncertainty
    dominates one with high uncertainty, which is the point of DS fusion
    rather than averaging.
    """
    C = (b1[:, 0] * b2[:, 1]) + (b1[:, 1] * b2[:, 0])
    denom = np.clip(1.0 - C, 1e-9, None)

    b = np.empty_like(b1)
    for k in range(K_CLASSES):
        b[:, k] = (b1[:, k] * b2[:, k] + b1[:, k] * u2 + b2[:, k] * u1) / denom
    u = (u1 * u2) / denom
    return b, u


def _expected_prob(belief: np.ndarray, uncertainty: np.ndarray) -> np.ndarray:
    """Expected probability of the positive class under the Dirichlet."""
    return belief[:, 1] + uncertainty / float(K_CLASSES)


class ProMUSE(AcquisitionPolicy):
    def __init__(
        self,
        ladder: list[str] | None = None,
        target_escalation_rate: float = 0.5,
        hidden: int = 32,
        epochs: int = 300,
        lr: float = 1e-2,
        weight_decay: float = 1e-3,
        predictor: str = "evidential",
        random_state: int = RANDOM_STATE,
    ):
        """predictor:
        "evidential" -- faithful to the paper: the evidential networks supply
            BOTH the escalation uncertainty and the final prediction.
        "shared"     -- ablation: evidential uncertainty still decides what to
            acquire, but the final prediction comes from the same shared
            logistic-regression bank every other policy in the benchmark
            uses. This isolates the ACQUISITION RULE from the predictor.
            Without it, ProMUSE is penalised for its neural predictor being
            weaker than logistic regression at this sample size (~290
            training subjects), which would make it a strawman rather than a
            fair baseline.
        """
        if predictor not in ("evidential", "shared"):
            raise ValueError("predictor must be 'evidential' or 'shared'")
        self.ladder = list(ladder or ["MRI", "AMYLOID"])
        self.target_escalation_rate = float(target_escalation_rate)
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.predictor = predictor
        self.random_state = random_state
        suffix = "" if predictor == "evidential" else ",pred=shared"
        self.name = (
            f"promuse[CHEAP->{'->'.join(self.ladder)},"
            f"rate={self.target_escalation_rate:g}{suffix}]"
        )

        self._models: dict[str, tuple] = {}
        self._thresholds: list[float] = []
        self.bank = BlockModelBank()

    def _fit_one(self, train_df: pd.DataFrame, blocks: list[str], label_col: str):
        torch.manual_seed(self.random_state)
        dm = fit_design_matrix(train_df, blocks)
        x = torch.tensor(np.asarray(dm.transform(train_df), dtype=np.float32))
        y = train_df[label_col].astype(int).to_numpy()
        y_onehot = torch.tensor(np.eye(K_CLASSES, dtype=np.float32)[y])

        model = EvidentialMLP(x.shape[1], self.hidden)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        model.train()
        for epoch in range(self.epochs):
            # KL weight annealed in, as in Sensoy et al.: ramping from 0
            # avoids crushing all evidence to zero early in training.
            kl_w = min(1.0, epoch / max(1, self.epochs // 2)) * 0.1
            opt.zero_grad()
            loss = evidential_loss(model(x), y_onehot, kl_w)
            loss.backward()
            opt.step()

        model.eval()
        return dm, model

    def _evidence(self, df: pd.DataFrame, key: str) -> np.ndarray:
        dm, model = self._models[key]
        x = torch.tensor(np.asarray(dm.transform(df), dtype=np.float32))
        with torch.no_grad():
            return model(x).numpy()

    def fit(self, train_df: pd.DataFrame, label_col: str) -> "ProMUSE":
        # One evidential model per modality, trained independently. Fusion
        # happens at the opinion level, not by joint training, which is what
        # lets a modality be skipped at inference time.
        self._models["CHEAP"] = self._fit_one(train_df, ["CHEAP"], label_col)
        for block in self.ladder:
            self._models[block] = self._fit_one(train_df, [block], label_col)

        # One threshold per rung, each set on training uncertainties so the
        # target fraction escalates.
        self._thresholds = []
        b, u = opinion_from_evidence(self._evidence(train_df, "CHEAP"))
        for block in self.ladder:
            thr = float(np.quantile(u, 1.0 - self.target_escalation_rate))
            self._thresholds.append(thr)
            esc = u > thr
            if esc.any():
                b2, u2 = opinion_from_evidence(self._evidence(train_df, block))
                b_new, u_new = dempster_shafer_fuse(b, u, b2, u2)
                b = np.where(esc[:, None], b_new, b)
                u = np.where(esc, u_new, u)

        if self.predictor == "shared":
            # Every acquisition state the ladder can reach -- NOT just
            # prefixes: a subject can escalate at one rung, fall below the
            # threshold and skip the next, then escalate again later, so
            # arbitrary subsets occur.
            from itertools import combinations

            sets = [["CHEAP"]]
            for r in range(1, len(self.ladder) + 1):
                for combo in combinations(self.ladder, r):
                    sets.append(["CHEAP", *combo])
            self.bank.fit(train_df, label_col, sets)
        return self

    def run(self, test_df: pd.DataFrame, label_col: str) -> PolicyResult:
        if not self._thresholds:
            raise RuntimeError("fit() must be called before run()")

        b, u = opinion_from_evidence(self._evidence(test_df, "CHEAP"))
        acquired: list[list[str]] = [[] for _ in range(len(test_df))]

        for block, thr in zip(self.ladder, self._thresholds):
            esc = u > thr
            if not esc.any():
                continue
            b2, u2 = opinion_from_evidence(self._evidence(test_df, block))
            b_new, u_new = dempster_shafer_fuse(b, u, b2, u2)
            b = np.where(esc[:, None], b_new, b)
            u = np.where(esc, u_new, u)
            for i, e in enumerate(esc):
                if e:
                    acquired[i].append(block)

        if self.predictor == "evidential":
            probs = _expected_prob(b, u)
        else:
            probs = np.zeros(len(test_df))
            for i in range(len(test_df)):
                got = acquired[i]
                probs[i] = self.bank.predict_proba(test_df.iloc[[i]], ["CHEAP", *got])[0]

        return self._result(test_df, label_col, probs, acquired)
