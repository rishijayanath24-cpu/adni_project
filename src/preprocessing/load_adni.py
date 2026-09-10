"""
Loader for the real ADNI ADNIMERGE2 raw tables (.rda), producing a
subject-visit dataframe schema-compatible with
mock_adnimerge.generate_mock_adnimerge()'s output, so the existing
cohort.py / splits.py / leakage_checks.py pipeline can run against it
largely unchanged.

FreeSurfer ROI codes (ST29SV, ST88SV, ST10CV, ST37SV, ST96SV) were looked
up directly in data/raw/DATADIC_01Sep2026.csv (TBLNAME == "UCSFFSX51") and
then verified present in every other UCSFFSX* table, not guessed.

Visit-code handling (this was the first real bug found and fixed here):
ADNI's plain VISCODE is protocol-specific and NOT comparable across phases
(ADNI1 uses "v01"/"y1"-style codes, ADNI4 prefixes "4_bl"/"4_m12", etc.).
The harmonized cross-phase code is VISCODE2 ("bl", "m06", "m12", ...),
present on every table used here except UPENNBIOMK_MASTER, which already
stores harmonized-style codes directly in its own VISCODE. All merges join
on this single harmonized "VISIT" column, never on raw VISCODE.

Known simplifications, flagged rather than hidden:
- Exact VISIT-code joins, not nearest-date matching -- a small number of
  scans/assessments taken near but not exactly at a diagnosis visit won't
  join and will show up as missing rather than approximately matched.
- Demographics/APOE are collapsed to one row per subject (first record by
  date) since they are collected once, not per visit.
- MRI volumes are pooled across FreeSurfer pipeline-version tables
  (UCSFFSX7 first, then UCSFFSX51 / UCSFFSX6 / UCSFFSX /
  UCSFFSX51_ADNI1_3T for residual gaps), because each ADNI phase was
  processed under a different version with its own table. The
  FS_PIPELINE column records the source per subject-visit and SHOULD be
  reported/adjusted for -- volumes from different FreeSurfer versions
  are not perfectly interchangeable.
- FreeSurfer rows flagged OVERALLQC == "Fail" are dropped; "Pass" is
  deliberately NOT required, since OVERALLQC is unpopulated for most
  UCSFFSX7 rows and requiring it would discard unlabeled-but-usable
  scans. Sensitivity to a stricter QC rule is worth checking later.

This module reads from data/raw/ only (gitignored) and must never write
subject-level fields anywhere outside data/ -- see project README.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr

DATA_DIR = Path("data/raw/ADNIMERGE2/ADNIMERGE2/data")


def _load(name: str) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = pyreadr.read_r(str(DATA_DIR / f"{name}.rda"))
    df = result[name].copy()
    if "RID" in df.columns:
        df["RID"] = pd.to_numeric(df["RID"], errors="coerce").astype("Int64")
    return df


def _harmonized_visit(df: pd.DataFrame) -> pd.Series:
    """Return the harmonized cross-phase visit code as a plain Series named 'VISIT'.

    Uses VISCODE2 where present (the norm); falls back to VISCODE for the
    one table that lacks VISCODE2 (UPENNBIOMK_MASTER), which already stores
    harmonized-style codes there.
    """
    return df["VISCODE2"] if "VISCODE2" in df.columns else df["VISCODE"]


def _coalesce_screening_into_baseline(df: pd.DataFrame, extra_baseline_codes: tuple[str, ...] = ()) -> pd.DataFrame:
    """Treat ADNI's separate screening visit ('sc'/'scmri', +table-specific
    codes) as baseline when a true 'bl' row is absent for that subject.

    ADNI1 sometimes recorded eligibility measures (e.g. MMSE) at screening
    rather than repeating them at the baseline visit itself -- confirmed
    directly (RID 2: MMSE only at 'sc', DIAGNOSIS only at 'bl'). A strict
    'bl'-only join silently drops real baseline data for these subjects. A
    genuine 'bl' row always wins when both exist.

    `extra_baseline_codes` handles table-specific quirks -- e.g. UCSFFSX51's
    baseline-equivalent MRI is coded 'm03' for a large fraction of subjects
    (920 of 4896 rows, confirmed directly), not just 'scmri'.
    """
    df = df.copy()
    screening_codes = ("sc", "scmri", *extra_baseline_codes)
    is_screening = df["VISIT"].isin(screening_codes)
    df["_priority"] = is_screening.astype(int)  # 0 = true baseline, 1 = screening fallback
    df.loc[is_screening, "VISIT"] = "bl"
    return df.sort_values("_priority").drop(columns="_priority")


def _viscode_to_month(viscode) -> float:
    if not isinstance(viscode, str):
        return np.nan
    v = viscode.lower()
    if v in ("bl", "sc", "scmri"):
        return 0.0
    if v in ("nv", "uns1", "init"):
        return np.nan
    m = re.match(r"m(\d+)$", v)
    if m:
        return float(m.group(1))
    return np.nan


def _apoe4_count(genotype) -> float:
    if not isinstance(genotype, str) or "/" not in genotype:
        return np.nan
    alleles = genotype.split("/")
    return float(sum(1 for a in alleles if a.strip() == "4"))


def _first_per_subject(df: pd.DataFrame, date_col: str | None = None) -> pd.DataFrame:
    """Collapse a subject-level table (demographics, genotype) to one row per RID."""
    df = df.dropna(subset=["RID"]).copy()
    if date_col and date_col in df.columns:
        df = df.sort_values(date_col)
    return df.groupby("RID", as_index=False).first()


def load_diagnosis() -> pd.DataFrame:
    """Per-visit diagnosis (CN / MCI / Dementia) from DXSUM.

    Screening is coalesced into baseline here too: 771 subjects (294 of them
    MCI) carry a diagnosis only at the screening visit and no 'bl' row at
    all, so a strict 'bl'-only rule silently drops them from the cohort --
    and drops them unevenly by ADNI phase, which would bias the sample. A
    genuine 'bl' diagnosis always takes priority where both exist, so a
    subject who was MCI at screening but already Dementia at baseline is
    still correctly read as Dementia (and thus excluded from an
    MCI-at-baseline cohort).
    """
    dx = _load("DXSUM")
    dx["VISIT"] = _harmonized_visit(dx)
    dx = dx[["RID", "VISIT", "EXAMDATE", "DIAGNOSIS", "COLPROT"]].dropna(subset=["RID", "DIAGNOSIS"])
    dx = _coalesce_screening_into_baseline(dx)
    dx = dx.rename(columns={"DIAGNOSIS": "DX"})
    dx["Month"] = dx["VISIT"].apply(_viscode_to_month)
    dx = dx.dropna(subset=["Month"])
    return dx.drop_duplicates(subset=["RID", "VISIT"], keep="first")


def load_demographics() -> pd.DataFrame:
    demog = _load("PTDEMOG")
    keep = ["RID", "VISDATE", "PTGENDER", "PTDOB", "PTEDUCAT", "PTRACCAT", "PTETHCAT"]
    demog = _first_per_subject(demog[keep], "VISDATE")
    demog["PTDOB_YEAR"] = pd.to_datetime(demog["PTDOB"], format="%m/%Y", errors="coerce").dt.year
    return demog[["RID", "PTGENDER", "PTDOB_YEAR", "PTEDUCAT", "PTRACCAT", "PTETHCAT"]]


def load_apoe() -> pd.DataFrame:
    apoe = _load("APOERES")
    apoe = _first_per_subject(apoe[["RID", "APTESTDT", "GENOTYPE"]], "APTESTDT")
    apoe["APOE4"] = apoe["GENOTYPE"].apply(_apoe4_count)
    return apoe[["RID", "APOE4"]]


def load_mmse() -> pd.DataFrame:
    mmse = _load("MMSE")
    mmse["VISIT"] = _harmonized_visit(mmse)
    mmse = _coalesce_screening_into_baseline(mmse)
    mmse = mmse.rename(columns={"MMSCORE": "MMSE"})
    return mmse[["RID", "VISIT", "MMSE"]].drop_duplicates(subset=["RID", "VISIT"], keep="first")


def load_cdr() -> pd.DataFrame:
    cdr = _load("CDR")
    cdr["VISIT"] = _harmonized_visit(cdr)
    cdr = _coalesce_screening_into_baseline(cdr)
    return cdr[["RID", "VISIT", "CDRSB"]].drop_duplicates(subset=["RID", "VISIT"], keep="first")


def load_fdg() -> pd.DataFrame:
    fdg = _load("UCBERKELEYFDG_8mm")
    fdg["VISIT"] = _harmonized_visit(fdg)
    fdg = _coalesce_screening_into_baseline(fdg)
    fdg = fdg[fdg["ROINAME"] == "MetaROI"]
    fdg = fdg.rename(columns={"MEAN": "FDG"})
    return fdg[["RID", "VISIT", "FDG"]].drop_duplicates(subset=["RID", "VISIT"], keep="first")


def load_amyloid() -> pd.DataFrame:
    amy = _load("UCBERKELEY_AMY_6MM")
    amy["VISIT"] = _harmonized_visit(amy)
    amy = _coalesce_screening_into_baseline(amy)
    amy = amy.rename(columns={"SUMMARY_SUVR": "AMYLOID_SUVR"})
    return amy[["RID", "VISIT", "AMYLOID_SUVR", "AMYLOID_STATUS"]].drop_duplicates(subset=["RID", "VISIT"], keep="first")


def load_tau() -> pd.DataFrame:
    tau = _load("UCBERKELEY_TAU_6MM")
    tau["VISIT"] = _harmonized_visit(tau)
    tau = _coalesce_screening_into_baseline(tau)
    tau = tau.rename(columns={"META_TEMPORAL_SUVR": "TAU_SUVR"})
    return tau[["RID", "VISIT", "TAU_SUVR"]].drop_duplicates(subset=["RID", "VISIT"], keep="first")


def load_csf() -> pd.DataFrame:
    csf = _load("UPENNBIOMK_MASTER")
    csf["VISIT"] = _harmonized_visit(csf)  # falls back to VISCODE -- already harmonized-style here
    csf = _coalesce_screening_into_baseline(csf)
    csf = csf.rename(columns={"TAU": "CSF_TAU", "PTAU": "CSF_PTAU", "ABETA": "CSF_ABETA"})
    return csf[["RID", "VISIT", "CSF_ABETA", "CSF_TAU", "CSF_PTAU"]].drop_duplicates(subset=["RID", "VISIT"], keep="first")


# FreeSurfer ROI codes, verified present in every UCSFFSX* table (checked
# directly per table rather than assumed to carry over across versions).
_FS_ROI_CODES = ["ST29SV", "ST88SV", "ST10CV", "ST37SV", "ST96SV"]

# Priority order for FreeSurfer source tables. UCSFFSX7 is the FreeSurfer 7.x
# reprocessing spanning ADNI1/GO/2/3/4 and alone covers 3,195 of the 3,230
# subjects that have any baseline-equivalent FreeSurfer row, so it goes first:
# one consistent pipeline version for nearly everyone avoids pipeline-version
# confounding. The older per-phase tables only fill residual gaps.
_FS_TABLE_PRIORITY = ["UCSFFSX7", "UCSFFSX51", "UCSFFSX6", "UCSFFSX", "UCSFFSX51_ADNI1_3T"]


def load_mri_volumes() -> pd.DataFrame:
    """Hippocampus (ST29SV+ST88SV), ICV (ST10CV), lateral ventricles (ST37SV+ST96SV).

    Combines every UCSFFSX* pipeline-version table, because each ADNI phase's
    scans were processed under a different FreeSurfer version with its own
    table -- querying only one silently zeroes out whole phases (UCSFFSX51
    alone gave 100% MRI coverage for ADNIGO/ADNI2 and exactly 0% for
    ADNI1/3/4). Rows are taken in _FS_TABLE_PRIORITY order and deduplicated
    per (RID, VISIT), so the newest pipeline wins where a subject-visit was
    processed more than once.

    FS_PIPELINE records which table each row came from, so pipeline version
    can be reported and adjusted for as a covariate rather than silently
    mixed into one column.

    QC handling: rows explicitly flagged OVERALLQC == "Fail" are dropped, but
    "Pass" is NOT required -- OVERALLQC is unpopulated (NaN) for 11,086 of
    12,151 UCSFFSX7 rows, so requiring "Pass" would discard ~91% of the
    primary table for being unlabeled rather than for failing QC.
    """
    frames = []
    for table in _FS_TABLE_PRIORITY:
        fsx = _load(table)
        fsx["VISIT"] = _harmonized_visit(fsx)
        fsx = _coalesce_screening_into_baseline(fsx, extra_baseline_codes=("m03",))

        if "OVERALLQC" in fsx.columns:
            fsx = fsx[fsx["OVERALLQC"] != "Fail"]

        for col in _FS_ROI_CODES:
            fsx[col] = pd.to_numeric(fsx[col], errors="coerce")

        fsx["Hippocampus"] = fsx["ST29SV"] + fsx["ST88SV"]
        fsx["ICV"] = fsx["ST10CV"]
        fsx["Ventricles"] = fsx["ST37SV"] + fsx["ST96SV"]
        fsx["FS_PIPELINE"] = table

        # Drop rows with no usable volume before they can shadow a later
        # table's populated row for the same subject-visit.
        fsx = fsx.dropna(subset=["Hippocampus", "ICV", "Ventricles"], how="all")
        fsx = fsx.drop_duplicates(subset=["RID", "VISIT"], keep="first")
        frames.append(fsx[["RID", "VISIT", "Hippocampus", "ICV", "Ventricles", "FS_PIPELINE"]])

    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(subset=["RID", "VISIT"], keep="first")


def load_field_strength() -> pd.DataFrame:
    """Scanner field strength.

    ADNI splits this across two tables by era: MRIMETA covers the 1.5T
    protocol (its FIELD_STRENGTH is unconditionally "1.5 Tesla" -- confirmed
    directly, it carries no 3T rows at all), while MRI3META covers the 3T
    protocol used from ADNI-GO/2 onward. Both must be loaded and combined,
    or every later-phase subject looks like missing field-strength data.
    """
    frames = []
    for table in ("MRIMETA", "MRI3META"):
        meta = _load(table)
        meta["VISIT"] = _harmonized_visit(meta)
        meta = _coalesce_screening_into_baseline(meta, extra_baseline_codes=("m03",))
        meta = meta.dropna(subset=["FIELD_STRENGTH"])
        frames.append(meta[["RID", "VISIT", "FIELD_STRENGTH"]])

    combined = pd.concat(frames, ignore_index=True)
    combined["FIELD_STRENGTH"] = combined["FIELD_STRENGTH"].replace(
        {"1.5 Tesla": "1.5T", "3 Tesla": "3T"}
    )
    return combined.drop_duplicates(subset=["RID", "VISIT"], keep="first")


def build_subject_visits() -> pd.DataFrame:
    """Merge all tables into one subject-visit dataframe."""
    dx = load_diagnosis()
    demog = load_demographics()
    apoe = load_apoe()

    df = dx.merge(demog, on="RID", how="left")
    df = df.merge(apoe, on="RID", how="left")

    for loader in (
        load_mmse,
        load_cdr,
        load_fdg,
        load_amyloid,
        load_tau,
        load_csf,
        load_mri_volumes,
        load_field_strength,
    ):
        piece = loader()
        df = df.merge(piece, on=["RID", "VISIT"], how="left")

    exam_year = pd.to_datetime(df["EXAMDATE"], errors="coerce").dt.year
    df["AGE"] = exam_year - df["PTDOB_YEAR"]

    return df.rename(columns={"VISIT": "VISCODE"})


if __name__ == "__main__":
    subject_visits = build_subject_visits()
    print(f"{len(subject_visits)} subject-visit rows, {subject_visits['RID'].nunique()} subjects")
    preview_cols = ["RID", "VISCODE", "Month", "DX", "AGE", "MMSE", "CDRSB", "Hippocampus", "FDG", "AMYLOID_SUVR", "TAU_SUVR"]
    print(subject_visits[preview_cols].head(10).to_string(index=False))
    print("\nNon-null counts:")
    print(subject_visits[preview_cols].notna().sum())
