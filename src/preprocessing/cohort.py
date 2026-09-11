"""
Cohort construction and horizon-conversion labeling for the ADNI MCI cohort.

Operates on an ADNIMERGE-shaped subject-visit table (real or the synthetic
stand-in in mock_adnimerge.py). Produces one row per subject with baseline
features and horizon-specific conversion labels.

Label semantics per horizon h in {12, 24, 36} months:
    1   -> observed to convert to Dementia at or before month h
    0   -> observed as MCI (or reverted to CN) at a visit >= month h,
           without having converted by then
    NaN -> insufficient follow-up to know the outcome at month h (censored) --
           these subjects MUST be excluded from horizon h's classification
           metrics, never silently coded as 0

Reverters (MCI -> CN without conversion) are flagged separately so they can
be reported as their own group rather than merged into "stable".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (12, 24, 36)


def _subject_label(visits: pd.DataFrame, horizon: int) -> tuple[float, bool]:
    """Return (label, is_reverter) for one subject at one horizon, from their visit history."""
    visits = visits.sort_values("Month")
    ever_reverted = False
    last_seen_month = visits["Month"].max()

    for _, row in visits.iterrows():
        if row["DX"] == "Dementia" and row["Month"] <= horizon:
            return 1.0, ever_reverted
        if row["DX"] == "CN":
            ever_reverted = True

    if last_seen_month >= horizon:
        return 0.0, ever_reverted
    return np.nan, ever_reverted


def build_cohort(subject_visits: pd.DataFrame) -> pd.DataFrame:
    """Build the baseline-feature + horizon-label cohort table from subject-visit rows.

    Inclusion: DX == "MCI" at a recorded baseline ('bl') visit. ADNI's own
    DIAGNOSIS field carries only {CN, MCI, Dementia} -- the older EMCI/LMCI
    subtype split lives in DXSUM's free-text DXMDES field, not DIAGNOSIS,
    and is not required for this project's inclusion criterion.
    """
    baseline = subject_visits[subject_visits["VISCODE"] == "bl"].set_index("RID")
    baseline = baseline[baseline["DX"] == "MCI"]

    records = []
    for rid, base_row in baseline.iterrows():
        subj_visits = subject_visits[subject_visits["RID"] == rid]
        record = {
            "RID": rid,
            "COLPROT": base_row["COLPROT"],
            "FIELD_STRENGTH": base_row["FIELD_STRENGTH"],
            # FreeSurfer pipeline version behind the MRI volumes -- pooled across
            # UCSFFSX* tables, so this belongs in the model as a covariate / in
            # the paper as a reported confound, not silently mixed.
            "FS_PIPELINE": base_row.get("FS_PIPELINE"),
            "AGE": base_row["AGE"],
            "PTGENDER": base_row["PTGENDER"],
            "PTEDUCAT": base_row["PTEDUCAT"],
            "PTRACCAT": base_row["PTRACCAT"],
            "PTETHCAT": base_row["PTETHCAT"],
            "APOE4": base_row["APOE4"],
            "MMSE_bl": base_row["MMSE"],
            "CDRSB_bl": base_row["CDRSB"],
            # AD-signature FreeSurfer ROIs (see load_adni._AD_SIGNATURE_ROIS).
            **{
                f"{roi}_bl": base_row.get(roi)
                for roi in (
                    "Hippocampus", "Amygdala", "Entorhinal_CV", "Entorhinal_TA",
                    "Fusiform_CV", "MidTemporal_CV", "InfTemporal_CV",
                    "ParaHippocampal_CV", "Precuneus_CV", "PosteriorCingulate_CV",
                    "Ventricles", "ICV",
                )
            },
            "FDG_bl": base_row["FDG"],
            "AMYLOID_SUVR_bl": base_row["AMYLOID_SUVR"],
            "AMYLOID_STATUS_bl": base_row.get("AMYLOID_STATUS"),
            "TAU_SUVR_bl": base_row.get("TAU_SUVR"),
            "CSF_ABETA_bl": base_row["CSF_ABETA"],
            "CSF_TAU_bl": base_row.get("CSF_TAU"),
            "CSF_PTAU_bl": base_row["CSF_PTAU"],
            "PLASMA_PTAU217_bl": base_row.get("PLASMA_PTAU217"),
            "PLASMA_AB42_AB40_bl": base_row.get("PLASMA_AB42_AB40"),
            # Per-modality availability. FDG, amyloid and tau PET are SEPARATE
            # scans with separate tracers and separate costs -- ordering one
            # does not yield the others, so they are tracked (and later
            # acquired) independently rather than as one "PET" block.
            "HAS_MRI_bl": pd.notna(base_row["Hippocampus"]),
            "HAS_FDG_bl": pd.notna(base_row["FDG"]),
            "HAS_AMYLOID_bl": pd.notna(base_row["AMYLOID_SUVR"]),
            "HAS_TAU_bl": pd.notna(base_row.get("TAU_SUVR")),
            "HAS_CSF_bl": pd.notna(base_row["CSF_ABETA"]) or pd.notna(base_row["CSF_PTAU"]),
            "HAS_PLASMA_bl": pd.notna(base_row.get("PLASMA_PTAU217")),
            # Retained for cohort-table reporting only ("had any PET at all").
            "HAS_PET_bl": pd.notna(base_row["FDG"]) or pd.notna(base_row["AMYLOID_SUVR"]),
            "N_VISITS": len(subj_visits),
            "LAST_MONTH_SEEN": subj_visits["Month"].max(),
        }
        for h in HORIZONS:
            label, is_reverter = _subject_label(subj_visits, h)
            record[f"LABEL_{h}mo"] = label
            record[f"REVERTER_{h}mo"] = is_reverter
        records.append(record)

    return pd.DataFrame(records)


def cohort_summary(cohort: pd.DataFrame) -> pd.DataFrame:
    """Per-horizon N / converters / stable / reverters / censored.

    Mirrors the cohort-reporting format standard in this literature (e.g.
    the medRxiv 2026 5-biomarker-class comparison), which is what a reviewer
    will expect to see reproduced in Table 1.
    """
    rows = []
    for h in HORIZONS:
        lab = cohort[f"LABEL_{h}mo"]
        rev = cohort[f"REVERTER_{h}mo"]
        rows.append(
            dict(
                horizon_months=h,
                n_total=len(cohort),
                n_converters=int((lab == 1).sum()),
                n_stable=int(((lab == 0) & (~rev)).sum()),
                n_reverters=int(((lab == 0) & rev).sum()),
                n_censored=int(lab.isna().sum()),
            )
        )
    return pd.DataFrame(rows)
