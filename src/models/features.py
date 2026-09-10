"""
Modality-block feature assembly for the acquisition-policy experiments.

The project's whole question is "which modality block is worth acquiring for
this patient", so features are organised into acquirable BLOCKS with real
acquisition costs, not a flat feature vector:

    CHEAP  -- demographics + cognitive tests + APOE genotype. Treated as
              always available: no scan, no lumbar puncture.
    MRI    -- FreeSurfer volumetrics (hippocampus, ICV, ventricles).
    PET    -- FDG / amyloid / tau SUVR summaries.
    CSF    -- lumbar-puncture biomarkers (Abeta, p-tau).

Costs are deliberately parameterised rather than hardcoded to one number:
they are contested, and the paper reports sensitivity across regimes rather
than pretending one dollar figure is authoritative.

Preprocessing (imputation + scaling) is fitted on TRAIN ONLY and applied to
val/test. Fitting an imputer or scaler on the full cohort before splitting
is a documented leakage mode in this literature -- see project README.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- Blocks -----------------------------------------------------------------

BLOCKS: dict[str, list[str]] = {
    "CHEAP": ["AGE", "PTEDUCAT", "APOE4", "MMSE_bl", "CDRSB_bl", "PTGENDER"],
    "MRI": ["Hippocampus_bl", "ICV_bl", "Ventricles_bl"],
    # FDG, amyloid and tau are separate scans with separate tracers and
    # separate costs -- ordering an amyloid scan does not hand you tau, so
    # they are separate acquirable blocks rather than one "PET" block.
    "FDG": ["FDG_bl"],
    "AMYLOID": ["AMYLOID_SUVR_bl"],
    "TAU": ["TAU_SUVR_bl"],
    "CSF": ["CSF_ABETA_bl", "CSF_TAU_bl", "CSF_PTAU_bl"],
    # Blood draw: the cheap, low-burden rung of the modern diagnostic
    # pathway. Plasma p-tau217 is reported competitive with amyloid PET for
    # amyloid status in the 2024-25 literature, so a policy that can reach
    # for it first is the clinically realistic comparison.
    "PLASMA": ["PLASMA_PTAU217_bl", "PLASMA_AB42_AB40_bl"],
}

CATEGORICAL_FEATURES = {"PTGENDER"}

# Availability flag per acquirable block, as produced by cohort.py.
BLOCK_AVAILABILITY = {
    "MRI": "HAS_MRI_bl",
    "FDG": "HAS_FDG_bl",
    "AMYLOID": "HAS_AMYLOID_bl",
    "TAU": "HAS_TAU_bl",
    "CSF": "HAS_CSF_bl",
    "PLASMA": "HAS_PLASMA_bl",
}

ACQUIRABLE_BLOCKS = ["PLASMA", "MRI", "FDG", "AMYLOID", "TAU", "CSF"]

# Relative acquisition cost regimes. Units are deliberately abstract
# ("cost units"), anchored to CHEAP = 0 since cognition/demographics/APOE are
# already in hand. Three regimes are carried so every headline result can be
# reported as a sensitivity sweep instead of one contestable point estimate.
COST_REGIMES: dict[str, dict[str, float]] = {
    # Roughly proportional to US list prices: MRI ~ $1k, FDG-PET ~ $2k,
    # amyloid/tau PET ~ $5k, LP + CSF assay ~ $1.5k.
    "price_like": {"CHEAP": 0.0, "PLASMA": 0.3, "MRI": 1.0, "FDG": 2.0, "AMYLOID": 5.0, "TAU": 5.0, "CSF": 1.5},
    # Flat: every added modality costs the same. Isolates "how many tests"
    # from "which tests".
    "flat": {"CHEAP": 0.0, "PLASMA": 1.0, "MRI": 1.0, "FDG": 1.0, "AMYLOID": 1.0, "TAU": 1.0, "CSF": 1.0},
    # Burden-weighted: penalises the invasive lumbar puncture and the
    # radiation-exposing PET tracers over a non-invasive MRI.
    "burden": {"CHEAP": 0.0, "PLASMA": 0.5, "MRI": 1.0, "FDG": 3.0, "AMYLOID": 4.0, "TAU": 4.0, "CSF": 6.0},
}


def block_cost(blocks: list[str], regime: str = "price_like") -> float:
    costs = COST_REGIMES[regime]
    return float(sum(costs[b] for b in blocks))


# --- Design matrix ----------------------------------------------------------


@dataclass
class DesignMatrix:
    """Fitted feature pipeline plus the column list it consumes."""

    pipeline: Pipeline
    columns: list[str]

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline.transform(df[self.columns])


def feature_columns(blocks: list[str]) -> list[str]:
    cols: list[str] = []
    for b in blocks:
        cols.extend(BLOCKS[b])
    return cols


def fit_design_matrix(train_df: pd.DataFrame, blocks: list[str]) -> DesignMatrix:
    """Fit imputation + scaling on TRAIN ONLY for the given blocks."""
    cols = feature_columns(blocks)
    numeric = [c for c in cols if c not in CATEGORICAL_FEATURES]
    categorical = [c for c in cols if c in CATEGORICAL_FEATURES]

    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
                    ]
                ),
                categorical,
            )
        )

    pipeline = Pipeline([("prep", ColumnTransformer(transformers))])
    pipeline.fit(train_df[cols])
    return DesignMatrix(pipeline=pipeline, columns=cols)


def available_blocks(row: pd.Series) -> list[str]:
    """Which acquirable blocks does this subject actually have data for?"""
    return [b for b in ACQUIRABLE_BLOCKS if bool(row.get(BLOCK_AVAILABILITY[b], False))]


def subjects_with_blocks(cohort: pd.DataFrame, blocks: list[str]) -> pd.DataFrame:
    """Restrict to subjects who actually have every requested acquirable block.

    Used for matched-subset comparisons: the incremental value of a modality
    must be measured on subjects who have it, otherwise the comparison
    confounds modality value with sample composition -- the exact trap that
    makes naive "we fused four modalities" results hard to interpret.
    """
    mask = pd.Series(True, index=cohort.index)
    for b in blocks:
        if b in BLOCK_AVAILABILITY:
            mask &= cohort[BLOCK_AVAILABILITY[b]].astype(bool)
    return cohort[mask]
