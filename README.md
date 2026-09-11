# Fairness- and Horizon-Audited Sequential Modality Acquisition for Alzheimer's Risk Stratification on ADNI

## Status
Data pipeline, all three baseline re-implementations, the policy benchmark
(12/24/36 months) and the fairness audit are done and running on real ADNI
data. Literature novelty check still pending a formal Scopus/Web of Science
search.

**Both original novelty claims have landed as null results** (Findings §2-4).
The horizon effect holds only for amyloid; no learned acquisition policy
beats a random or fixed rule; and the race-fairness audit is not feasible on
ADNI conversion cohorts. The defensible contribution is now the audit itself
— see "Implication for the paper" below before planning further modelling.

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
5. ✅ Re-implemented UMoS, ProMUSE and DQN-2025 (`src/baselines/`) and
   benchmarked them at 12/24/36 months.
6. ✅ Fairness audit of acquisition behaviour (`src/eval/fairness.py`).
7. ⬜ Horizon-conditioned acquisition policy. **Next task** — though see
   Findings §3: the bar is now beating a fixed rule, which nothing has yet.
8. ⬜ Fairness-constrained policy variant.
9. ⬜ Full evaluation: cost-accuracy Pareto frontiers, calibration,
   decision-curve analysis, fairness tables.
10. ⬜ Re-run the systematic literature search before finalising the
    novelty claim.
11. ⬜ Write-up.

## Findings so far

### 1. Incremental value of each modality
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

`*` paired bootstrap CI excludes 0; `†` underpowered.

### 2. Horizon effect — only amyloid survives a formal test
On a fixed subject set labelled at all horizons, paired delta-of-deltas
bootstrap (12mo → 36mo):

| block | Δ-of-Δ | 95% CI | significant |
|---|---|---|---|
| AMYLOID | +0.054 | [+0.014, +0.095] | **yes** |
| CSF | +0.024 | [−0.009, +0.055] | no |
| FDG | +0.020 | [−0.014, +0.054] | no |
| MRI | −0.019 | [−0.057, +0.016] | no |

The per-horizon table looks like a clean monotone story across four
modalities; only amyloid's horizon dependence survives an interaction test.
State the claim that narrowly.

### 3. Policy benchmark — the central negative result
AUC by policy and horizon (mean cost in "price-like" units):

| policy | cost | 12mo | 24mo | 36mo |
|---|---|---|---|---|
| cheap_only | 0.00 | 0.817 | 0.824 | 0.796 |
| fixed_MRI | 1.00 | 0.827 | 0.855 | 0.839 |
| **fixed_MRI+FDG** | 3.00 | 0.843 | 0.882 | 0.874 |
| umos_mri_to_amyloid | 3.64 | 0.838 | 0.879 | 0.850 |
| promuse_full_ladder_sharedpred | 4.71 | 0.836 | 0.890 | 0.870 |
| dqn_lambda0.01 | 4.86 | 0.842 | 0.881 | 0.874 |
| random_rate0.5 | 5.07 | 0.859 | 0.895 | 0.867 |
| acquire_all | 9.50 | 0.863 | 0.894 | 0.895 |

**Q1 — does choosing smarter help? No.** No learned policy significantly
beats *random* acquisition at any horizon (only `acquire_all` does, and
only at 36 months). Twelve policy-horizon pairs are significantly *worse*
than random.

**Q2 — is the personalisation worth its price? Usually not, but this one
is cost-assumption dependent.** Under the price-like cost regime most
learned policies are dominated by a patient-independent rule costing no
more — usually `fixed_MRI+FDG` (cost 3.00). A robustness sweep (Findings
§5) shows this is *not* regime-independent: the DQN is dominated under all
three cost regimes, UMoS only under price-like, and ProMUSE's full ladder
under none. Q1 is the regime-independent claim, since it is a paired
statistical test that makes no cost assumption at all; Q2 should be
reported with its cost caveat attached.

The large cost savings these papers report do reproduce in the weak sense:
you *can* cut acquisition cost substantially without losing accuracy. But
that saving does not come from personalisation — a fixed rule captures it
just as well.

### 4. Fairness audit
**Race/ethnicity: infeasible on ADNI conversion cohorts today.**

| stage | n | % non-White | n non-White |
|---|---|---|---|
| MCI at baseline | 1622 | 17.1% | 277 |
| + has 24-month outcome | 904 | 7.6% | 69 |
| + complete 4 modalities | 457 | 5.9% | 27 |

ADNI4 is **49.8% non-White** but only **1.7%** of its subjects have a
24-month label yet — the diversity gains are too recent to have follow-up.
The audit cohort holds n=10 Black and n=8 Hispanic participants with zero
converters between them.

**Sex: powered, and no disparity survives correction.** Uncorrected, the
DQN sends males for MRI 12.2pp more often (p=0.011) and UMoS sends males
for amyloid PET 10.6pp more often (p=0.040); neither survives
Benjamini-Hochberg across ~30 (policy, modality) tests.

**Fixed policies have zero acquisition disparity by construction** — they
ignore the patient — while matching or beating the learned policies on AUC.

### 5. Robustness of the negative result
The central claim was stress-tested rather than asserted.

| check | result |
|---|---|
| Cost regime (price_like / flat / burden) | Domination is **regime-dependent**: DQN dominated under all three, UMoS under one, ProMUSE's full ladder under none. Corrected in Q2 above. |
| Predictor (logistic regression → gradient boosting) | Negative result **strengthens**. All three learned policies dominated; `fixed_MRI+FDG` reaches 0.878 at cost 3.0, matching `acquire_all` (0.878) at a third of the cost. Not an artifact of a weak predictor. |
| CV / policy seed (0 / 7 / 42) | **Stable.** Per-policy SD 0.003-0.011. Mean ordering unchanged: random 0.886 still above every learned policy; `fixed_MRI+FDG` 0.884 ties ProMUSE (0.884) and beats UMoS (0.881) and DQN (0.873) at lower cost. |

### Implication for the paper
Both original novelty claims landed as null results. The defensible
contribution is now the audit itself: a leakage-free, cost-matched
benchmark showing that three published acquisition policies are dominated
by patient-independent rules, plus a quantified demonstration that ADNI
cannot currently support a race-fairness audit of conversion models.

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
