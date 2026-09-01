"""
Synthetic ADNIMERGE-schema generator for pipeline development and testing ONLY.

This is NOT real ADNI data and must never be used to produce or report any
result. It exists so the cohort-construction, splitting, and leakage-check
code can be built and exercised before real ADNI access is granted. Column
names and value domains mirror ADNIMERGE; the values themselves are randomly
generated and carry no epidemiological meaning.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VISITS = ["bl", "m06", "m12", "m18", "m24", "m36"]
VISIT_MONTHS = {"bl": 0, "m06": 6, "m12": 12, "m18": 18, "m24": 24, "m36": 36}

DX_BL_MCI_CODES = ("EMCI", "LMCI")

# ADNI is documented to oversample White, non-Hispanic participants relative
# to the US population -- mirrored here so downstream fairness-audit code has
# a realistically imbalanced subgroup distribution to contend with.
RACE_CATEGORIES = ["White", "Black", "Asian", "More than one", "Am Indian/Alaskan", "Hawaiian/Other PI", "Unknown"]
RACE_PROBS = [0.85, 0.05, 0.04, 0.02, 0.01, 0.01, 0.02]
ETH_CATEGORIES = ["Not Hisp/Latino", "Hisp/Latino", "Unknown"]
ETH_PROBS = [0.92, 0.06, 0.02]


def generate_mock_adnimerge(n_subjects: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Generate a synthetic, ADNIMERGE-shaped subject-visit table of MCI-at-baseline subjects.

    One row per (subject, visit) actually "observed" -- dropout is simulated
    by truncating a subject's visit history, not by NaN-filling rows that
    were never collected.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for i in range(n_subjects):
        rid = 1000 + i
        age_bl = rng.normal(74, 7)
        sex = rng.choice(["Male", "Female"], p=[0.53, 0.47])
        educ = int(np.clip(rng.normal(15.5, 3), 6, 20))
        race = rng.choice(RACE_CATEGORIES, p=RACE_PROBS)
        ethnicity = rng.choice(ETH_CATEGORIES, p=ETH_PROBS)
        apoe4 = int(rng.choice([0, 1, 2], p=[0.55, 0.35, 0.10]))
        dx_bl = rng.choice(DX_BL_MCI_CODES, p=[0.55, 0.45])
        colprot = rng.choice(["ADNI1", "ADNIGO", "ADNI2", "ADNI3"], p=[0.30, 0.10, 0.35, 0.25])
        field_strength = "1.5T" if (colprot == "ADNI1" and rng.random() < 0.8) else "3T"

        # Latent conversion-risk score -- synthetic, not epidemiologically calibrated.
        risk = (
            0.03 * (age_bl - 70)
            + 0.5 * apoe4
            + (0.4 if dx_bl == "LMCI" else 0.0)
            + rng.normal(0, 0.6)
        )

        # Modality availability is fixed per subject across their visits, mirroring
        # how ADNI sub-study enrollment behaves in practice (mostly all-or-nothing).
        has_pet = rng.random() < 0.42
        has_csf = rng.random() < 0.32
        has_mri = rng.random() < 0.90

        max_visit_idx = rng.choice(len(VISITS), p=[0.05, 0.10, 0.15, 0.15, 0.25, 0.30])

        state = "MCI"
        for v_idx, visit in enumerate(VISITS):
            if v_idx > max_visit_idx:
                break
            month = VISIT_MONTHS[visit]

            if v_idx > 0 and state == "MCI":
                p_convert = 1 / (1 + np.exp(-(risk - 2.2 + 0.03 * month)))
                p_revert = 0.02 if risk < -0.5 else 0.005
                draw = rng.random()
                if draw < p_convert * 0.15:
                    state = "Dementia"
                elif draw < p_convert * 0.15 + p_revert:
                    state = "CN"

            dx = state
            mmse = float(np.clip(rng.normal(27 - 0.05 * month - (3 if state == "Dementia" else 0), 2), 0, 30))
            cdrsb = float(np.clip(rng.normal(1.5 + 0.03 * month + (2.5 if state == "Dementia" else 0), 1), 0, 18))

            hippocampus = rng.normal(6800 - 15 * month - 200 * apoe4, 700) if has_mri else np.nan
            fdg = rng.normal(1.25 - 0.002 * month - 0.05 * apoe4, 0.12) if (has_pet and rng.random() < 0.95) else np.nan
            av45 = rng.normal(1.1 + 0.15 * apoe4 + 0.002 * month, 0.2) if (has_pet and rng.random() < 0.85) else np.nan
            abeta = rng.normal(900 - 120 * apoe4, 200) if (has_csf and rng.random() < 0.9) else np.nan
            ptau = rng.normal(25 + 8 * apoe4, 10) if (has_csf and rng.random() < 0.9) else np.nan

            rows.append(
                dict(
                    RID=rid,
                    VISCODE=visit,
                    Month=month,
                    COLPROT=colprot,
                    FIELD_STRENGTH=field_strength,
                    AGE=round(age_bl + month / 12, 1),
                    PTGENDER=sex,
                    PTEDUCAT=educ,
                    PTRACCAT=race,
                    PTETHCAT=ethnicity,
                    APOE4=apoe4,
                    DX_bl=dx_bl,
                    DX=dx,
                    MMSE=round(mmse, 1),
                    CDRSB=round(cdrsb, 1),
                    Hippocampus=None if np.isnan(hippocampus) else round(float(hippocampus), 1),
                    FDG=None if np.isnan(fdg) else round(float(fdg), 3),
                    AV45=None if np.isnan(av45) else round(float(av45), 3),
                    ABETA=None if np.isnan(abeta) else round(float(abeta), 1),
                    PTAU=None if np.isnan(ptau) else round(float(ptau), 2),
                )
            )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_mock_adnimerge()
    out_path = "data/processed/mock_adnimerge.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} subject-visit rows for {df['RID'].nunique()} subjects to {out_path}")
