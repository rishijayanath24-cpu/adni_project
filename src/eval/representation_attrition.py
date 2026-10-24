"""
Representation attrition through cohort construction (24-month horizon).

Traces how many non-White and Hispanic participants survive each stage of
complete-case multimodal cohort construction, adds the phase-level context
(how diverse the newest ADNI phase is versus how much follow-up it has), and
counts the participants available to a race- or ethnicity-stratified audit in
the 24-month development cohort. Writes paper/tables/representation_attrition_24mo.csv
and paper/tables/representation_audit_cohort_24mo.csv.

Run:  python -m src.eval.representation_attrition
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.features import subjects_with_blocks
from src.preprocessing.cohort import build_cohort
from src.preprocessing.load_adni import build_subject_visits
from src.preprocessing.splits import subject_level_split

BLOCKS = ["MRI", "FDG", "AMYLOID", "CSF"]
LABEL = "LABEL_24mo"


def stage_row(name: str, df: pd.DataFrame) -> dict:
    white = df["PTRACCAT"].eq("White")
    return dict(
        stage=name, n=len(df), pct_white=round(100 * white.mean(), 1),
        n_non_white=int((~white).sum()), pct_non_white=round(100 * (~white).mean(), 1),
        n_hispanic=int(df["PTETHCAT"].eq("Hispanic or Latino").sum()),
    )


def main() -> None:
    visits = build_subject_visits()
    cohort = build_cohort(visits)
    labelled = cohort[cohort[LABEL].notna()]
    complete = subjects_with_blocks(labelled, BLOCKS)
    rows = [
        stage_row("1. MCI at baseline (all)", cohort),
        stage_row("2. + has 24mo outcome label", labelled),
        stage_row("3. + complete 4 modalities", complete),
    ]
    out = pd.DataFrame(rows)
    out.to_csv("paper/tables/representation_attrition_24mo.csv", index=False)
    print(out.to_string(index=False))

    # Newest phase: diversity versus available follow-up.
    phase = cohort.groupby("COLPROT").apply(
        lambda g: pd.Series(dict(
            n=len(g),
            pct_non_white=round(100 * (~g["PTRACCAT"].eq("White")).mean(), 1),
            pct_with_24mo_label=round(100 * g[LABEL].notna().mean(), 1),
        )), include_groups=False)
    print("\nMCI-at-baseline cohort by ADNI phase:\n", phase.to_string())
    phase.to_csv("paper/tables/representation_by_phase.csv")

    split = subject_level_split(complete, label_col=LABEL, random_state=42)
    dev = complete[~complete["RID"].isin(split.test_rids)]
    groups = {
        "Black or African American": dev["PTRACCAT"].eq("Black or African American"),
        "Asian": dev["PTRACCAT"].eq("Asian"),
        "Hispanic or Latino": dev["PTETHCAT"].eq("Hispanic or Latino"),
        "Female": dev["PTGENDER"].astype(str).str.startswith("F"),
        "Male": dev["PTGENDER"].astype(str).str.startswith("M"),
    }
    aud = pd.DataFrame([
        dict(group=g, n=int(m.sum()), converters=int(dev.loc[m, LABEL].sum()))
        for g, m in groups.items()
    ])
    aud.to_csv("paper/tables/representation_audit_cohort_24mo.csv", index=False)
    print(f"\n24-month development cohort (n={len(dev)}):\n", aud.to_string(index=False))


if __name__ == "__main__":
    main()
