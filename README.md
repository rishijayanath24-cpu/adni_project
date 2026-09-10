# Fairness- and Horizon-Audited Sequential Modality Acquisition for Alzheimer's Risk Stratification on ADNI

## Status
Data pipeline complete and running on real ADNI data. First two experiments
done (modality value, horizon effect). Baseline re-implementations (UMoS /
ProMUSE / DQN-2025) not started. Literature novelty check still pending a
formal Scopus/Web of Science search.

**Cohort:** 1,622 MCI-at-baseline subjects from 3,777 total. Converters:
121 / 237 / 297 at 12 / 24 / 36 months, with reverters (110 / 107 / 104) and
censored subjects tracked separately and never folded into "stable".

## What this is
Not a new fusion architecture. A **fairness and prediction-horizon audit** of three existing, published, cost-aware sequential modality-acquisition policies for Alzheimer's diagnosis on ADNI:

- **UMoS** (Zheng et al., *Alzheimer's & Dementia* 2022 / *IISE Trans. Healthcare Sys. Eng.* 2023) — MRI-model uncertainty gates amyloid-PET acquisition.
- **ProMUSE** (arXiv:2606.19371, 2026) — staged evidential (Dirichlet/Dempster-Shafer) uncertainty gating, clinical → MRI → PET.
- **Sequential-decision RL policy** (ACM ICBDSIC 2025) — DQN/PPO test-ordering policy.

None of these three report (a) disparate acquisition/error rates by race, ethnicity, or sex, or (b) whether the optimal policy changes with the prediction horizon (12 vs. 24 vs. 36 months). This project re-implements all three as baselines, then adds both audits, plus a fairness-constrained and horizon-conditioned policy variant.

**Do not claim "first to do sequential/cost-aware acquisition on ADNI."** That claim is false — the three papers above already do it. The claim is: first fairness + horizon audit of this specific line of work.

## Repo layout
```
data/
  raw/            # ADNIMERGE + supplementary tables, once ADNI access is granted (not raw images)
  processed/      # cohort-constructed, split, feature-engineered tables
src/
  preprocessing/  # cohort construction, subject-level splitting, leakage checks
  baselines/      # re-implementations of UMoS, ProMUSE, DQN-2025
  models/         # fairness-constrained / horizon-conditioned acquisition policy
  eval/           # Pareto frontiers, calibration, fairness metrics, statistical tests
notebooks/        # exploratory analysis only — nothing load-bearing lives here
paper/
  figures/
  tables/
```

## Data
Uses the **ADNIMERGE2 package's raw component tables** (217 `.rda` files),
not a single pre-merged table — that package ships the source tables rather
than one merged frame, which is preferable here since we get full raw fields.
Read with `pyreadr` (no R install required). Tables actually used:

| purpose | table |
|---|---|
| diagnosis over time | `DXSUM` |
| demographics | `PTDEMOG` |
| genotype | `APOERES` |
| cognition | `MMSE`, `CDR` |
| MRI volumes | `UCSFFSX7` + `UCSFFSX51`/`UCSFFSX6`/`UCSFFSX`/`UCSFFSX51_ADNI1_3T` |
| FDG-PET | `UCBERKELEYFDG_8mm` (MetaROI) |
| amyloid-PET | `UCBERKELEY_AMY_6MM` (`SUMMARY_SUVR`) |
| tau-PET | `UCBERKELEY_TAU_6MM` (`META_TEMPORAL_SUVR`) |
| CSF | `UPENNBIOMK_ROCHE_ELECSYS` |
| plasma | `C2N_PRECIVITYAD2_PLASMA` |
| scanner field strength | `MRIMETA` (1.5T) + `MRI3META` (3T) |

No raw MRI/PET image preprocessing is required. FreeSurfer ROI codes were
looked up in `DATADIC_01Sep2026.csv` and verified per table.

**Visit codes:** all merges join on the harmonized `VISCODE2` (`bl`, `m06`,
…), never raw `VISCODE`, which is protocol-specific and not comparable
across phases. Screening visits coalesce into baseline where no true `bl`
row exists, with a genuine `bl` row always winning.

## Non-negotiable rigor rules
- **Subject-level train/val/test splits, enforced before any preprocessing statistic is computed.** This is the single largest source of inflated accuracy documented in this field (a 28-point swing in one directly-audited case).
- Stratify and report separately by scanner/field-strength (ADNI-1 1.5T vs. ADNI-2/3 3T).
- MCI reverters are a distinct label category, never silently merged into "stable."
- Report calibration (Brier score, reliability) and decision-curve analysis, not AUC alone.
- Report fairness metrics (equalized odds, equal opportunity, demographic parity) for every model, including the three re-implemented baselines.

## Milestones
1. ✅ ADNI data obtained (ADNIMERGE2 package: 217 raw `.rda` tables + data
   dictionary). Read via `pyreadr`; no R install needed.
2. ✅ Cohort construction + leakage-safe subject-level splits, verified on
   real data. Scanner-stratified splits confirmed balanced.
3. ✅ Modality-value experiment (`src/eval/run_modality_value.py`).
4. ✅ Horizon-effect experiment with formal trend tests
   (`src/eval/run_horizon_effect.py`).
5. ⬜ Re-implement and validate UMoS, ProMUSE, DQN-2025 against published
   numbers. **This is the next task.**
6. ⬜ Fairness audit of all baselines' acquisition behaviour.
7. ⬜ Horizon-conditioned acquisition policy.
8. ⬜ Fairness-constrained policy variant.
9. ⬜ Full evaluation: cost-accuracy Pareto frontiers, calibration,
   decision-curve analysis, fairness tables.
10. ⬜ Re-run the systematic literature search before finalising the
    novelty claim.
11. ⬜ Write-up.

## Findings so far

Incremental AUC over a cheap-only model (demographics + cognition + APOE),
cross-validated on the development set, matched to subjects holding each
block. Test set untouched.

| block | 12mo | 24mo | 36mo |
|---|---|---|---|
| MRI | +0.054* | +0.037* | +0.039* |
| FDG | +0.022 | +0.038* | +0.047* |
| AMYLOID | +0.015 | +0.036* | +0.068* |
| CSF | +0.014 | +0.030* | +0.047* |
| TAU | – | +0.149† | +0.167† |
| MRI+FDG+AMYLOID+CSF | +0.032 | +0.063* | +0.111* |

`*` significant (paired bootstrap CI excludes 0); `†` underpowered.

**Horizon effect** — tested on a fixed subject set labelled at all horizons,
using a paired delta-of-deltas bootstrap (12mo → 36mo):

| block | Δ-of-Δ | 95% CI | significant |
|---|---|---|---|
| AMYLOID | +0.054 | [+0.014, +0.095] | **yes** |
| CSF | +0.024 | [−0.009, +0.055] | no |
| FDG | +0.020 | [−0.014, +0.054] | no |
| MRI | −0.019 | [−0.057, +0.016] | no |
| TAU | +0.022 | [−0.094, +0.134] | underpowered |

The per-horizon table above *looks* like a clean monotone story across four
modalities, and MRI looks like it declines while molecular markers rise. Only
amyloid's horizon dependence survives a formal interaction test. The
horizon-conditioning premise is supported specifically for amyloid PET — the
most cost-consequential modality — and must be stated that narrowly.

## Modality availability (MCI cohort, baseline)
MRI 84.7% (1374) · FDG 53.5% (867) · AMYLOID 49.7% (806) · CSF 46.8% (759) ·
TAU 20.3% (329) · PLASMA 5.3% (86).

No subject holds all six blocks. Largest complete cases: MRI+FDG 862,
+AMYLOID 643, +CSF 423, all five imaging/CSF blocks 113. Plasma is
effectively unusable for outcome prediction here — collection concentrates
in recent ADNI4 enrollees who have no conversion follow-up yet (4 labelled
at 24 months, 0 at 36).

## Known open risks and limitations
- **Novelty claim unverified.** The literature check so far is web-search
  based, not a formal Scopus/Web of Science systematic search. Treat as
  "likely open, not confirmed".
- **Cohort size vs literature.** Our 1,622 MCI-at-baseline (1,798 MCI at any
  visit) is below the ~2,430 reported in the medRxiv 2026 biomarker
  comparison. That preprint's full text is inaccessible (HTTP 403), so exact
  inclusion criteria can't be compared. Likely a cohort-definition
  difference, but our criteria must be stated precisely in the paper.
- **FreeSurfer pipeline mixing.** MRI volumes are pooled across UCSFFSX*
  version tables (99% from UCSFFSX7). `FS_PIPELINE` records the source per
  subject and should be reported/adjusted for.
- **Exact visit-code joins**, not nearest-date matching, so measurements
  taken near but not at a diagnosis visit read as missing. Understates
  availability somewhat.
- **CSF QC / assay.** Single assay (Roche Elecsys) by choice; the legacy
  table is not pooled in. FreeSurfer rows flagged `OVERALLQC == "Fail"` are
  dropped but "Pass" is not required, since QC is unlabeled for most
  UCSFFSX7 rows. Sensitivity to stricter QC untested.
- **Retrospective acquisition.** Any cost saving a policy shows is
  simulated; ADNI missingness is not missing-at-random (protocol- and
  phase-driven). Frame as decision support, not a deployed policy.
