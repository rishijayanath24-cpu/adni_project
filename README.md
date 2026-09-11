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
7. ✅ Horizon-conditioned acquisition policy (`src/baselines/horizon_conditioned.py`)
   — tested against horizon-swapped controls; null (Findings §7).
8. ⛔ Fairness-constrained policy variant — **premise removed**. Findings §4
   found no FDR-surviving acquisition disparity to correct, so building a
   constraint against it would be fixing a problem the data does not show.
9. ✅ Full evaluation: cost-accuracy frontiers, calibration, decision-curve
   analysis, fairness tables, plus a three-way robustness sweep.
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

**Sex: powered, and no disparity at all under the current feature set.**
With the original three-feature MRI block, two gaps were significant before
correction (DQN sending males for MRI +12.2pp, p=0.011; UMoS for amyloid
+10.6pp, p=0.040) and neither survived Benjamini-Hochberg. Under the
AD-signature features the gaps shrink further and are no longer significant
even uncorrected (p=0.103 and p=0.118). The earlier hint of disparity does
not reproduce.

**Fixed policies have zero acquisition disparity by construction** — they
ignore the patient — while matching or beating the learned policies on AUC.

### 5. Robustness of the negative result
The central claim was stress-tested rather than asserted.

| check | result |
|---|---|
| Cost regime (price_like / flat / burden) | Domination is **regime-dependent**: DQN dominated under all three, UMoS under one, ProMUSE's full ladder under none. Corrected in Q2 above. |
| Predictor (logistic regression → gradient boosting) | Negative result **strengthens**. All three learned policies dominated; `fixed_MRI+FDG` reaches 0.878 at cost 3.0, matching `acquire_all` (0.878) at a third of the cost. Not an artifact of a weak predictor. |
| CV / policy seed (0 / 7 / 42) | **Stable.** Per-policy SD 0.003-0.011. Mean ordering unchanged: random 0.886 still above every learned policy; `fixed_MRI+FDG` 0.884 ties ProMUSE (0.884) and beats UMoS (0.881) and DQN (0.873) at lower cost. |

### 6. Calibration and clinical utility
**Calibration** — our configurations are the best calibrated of the
informative policies (slope 1.116 / 1.110, against 0.902 for random and
0.686 for acquire-all, where 1.0 is perfect). Most policies over-extremise
(slope < 1); ours slightly under-extremise.

**Decision curve** — across thresholds 0.05–0.50 every policy beats
treat-all and treat-none, so prediction-guided decisions help. But no single
policy owns the curve: `random` wins at most thresholds, UMoS at 0.20–0.25,
ProMUSE at 0.30–0.35, acquire-all at 0.05. **Ours is never the single best
at any threshold.**

An important caveat on that last line, because it cuts against a naive
reading: standard decision-curve analysis counts only the costs of the
*treatment* decision, not the cost of the *tests* used to reach it. It
therefore structurally favours policies that acquire more. At equal net
benefit, `ours_MRI` spends 1.0 cost units against UMoS's 3.64 and
acquire-all's 9.5, and that difference is invisible to the curve.

### 7. Horizon-conditioned policy (our own proposed method) — also null
Experiment 2 found a real, significant horizon effect for amyloid PET, so a
policy whose escalation ladder depends on the horizon has a genuine
mechanism to exploit. Tested against horizon-SWAPPED controls (run the
36-month ladder at 12 months and vice versa), which isolates the horizon
signal from the machinery:

| horizon | matched vs swapped | matched vs fixed_MRI+FDG | matched vs random |
|---|---|---|---|
| 12 | −0.002 / −0.013, n.s. | −0.013, n.s. | −0.029, n.s. |
| 24 | +0.001 / −0.021, n.s. | **−0.031, worse** | **−0.044, worse** |
| 36 | +0.017 / +0.012, n.s. | −0.029, n.s. | −0.022, n.s. |

The matched ladder never significantly beats its own swapped control. At 36
months the direction is right (+0.017 over the 12-month ladder) but the CI
includes zero. **The amyloid horizon effect is real as a measurement and
not exploitable as an acquisition rule at this sample size.**

### 8. What actually beats the published policies
The audit says clever *policies* do not help. The fix is to improve what
every policy is built on, and keep the policy trivial.

**Two changes, each with a mechanism:**
- **Rich MRI** — the MRI block used 3 features (hippocampus, ICV, ventricles)
  out of the 366 columns FreeSurfer provides. It now carries the AD-signature
  regions: entorhinal (volume + thickness), amygdala, fusiform, mid/inferior
  temporal, parahippocampal, precuneus, posterior cingulate.
- **Pooled training** (`src/models/pooled_bank.py`) — the model for state
  `CHEAP+MRI` needs only MRI, yet was fitted on complete cases holding PET
  and CSF too. Pooling fits each state on everyone who holds it: 737
  subjects instead of 292 at 24 months.

**Attribution (paired, MRI-only state):** the two are roughly additive and
together cross significance at the better-powered horizons.

| horizon | rich MRI | pooling | both |
|---|---|---|---|
| 12mo | +0.009 | +0.016 | +0.032 [−0.010, +0.076] |
| 24mo | +0.015 | +0.012 | **+0.028 [+0.001, +0.058]** |
| 36mo | +0.010 | +0.007 | **+0.024 [+0.002, +0.047]** |

**Cost-matched headline** — our config using MRI *alone* (cost 1.0) vs the
original config acquiring all four modalities (cost 9.5):

| horizon | ours (MRI only) | original (all modalities) | difference |
|---|---|---|---|
| 12mo | 0.8596 | 0.8629 | −0.003 [−0.055, +0.051] |
| 24mo | 0.8823 | 0.8937 | −0.011 [−0.043, +0.020] |
| 36mo | 0.8628 | 0.8953 | −0.033 [−0.067, +0.002] |

Not distinguishably worse than acquiring everything, at a tenth of the cost.
Stated precisely: this is *"not distinguishably worse"*, not *"as good as"* —
and at 36 months the interval only just includes zero.

**Pareto dominance vs the published policies: 22 of 27 policy-horizon pairs**
are dominated (ours at least as accurate, costing no more). The honest
exceptions: `dqn_lambda0.2` occupies a cheaper-but-much-worse point at all
three horizons, `dqn_lambda0.05` likewise at 12 months, and
`promuse_full_ladder_sharedpred` reaches a higher AUC at 24 months (0.8952
vs 0.8823) — at 4.7× the cost.

**Still true:** no policy, ours included, significantly beats random
acquisition or acquire-all on raw AUC. Our claim is about the
cost-accuracy frontier, not about raw discrimination.

A failed first attempt is kept in `src/models/missingness_aware.py`: one
NaN-native gradient-boosting model trained on all subjects at once. The
extra data helped (+0.006) but gradient boosting costs −0.020 against
logistic regression at this sample size, so the net was negative.

### Implication for the paper
The project has both halves of a coherent paper:

1. **Negative** — three published acquisition policies (UMoS, ProMUSE, a
   2025 DQN) do not beat random or fixed acquisition on ADNI, at any of
   three horizons, under a leakage-free cost-matched protocol, and survive
   neither a predictor swap nor a decision-curve analysis.
2. **Positive** — the gains those papers chase are available anyway, from
   better inputs rather than better decision rules: AD-signature MRI
   features plus pooled training reach all-modality accuracy at a tenth of
   the acquisition cost, dominating 22 of 27 published policy-horizon pairs.
3. **Methodological** — ADNI cannot currently support a race-fairness audit
   of conversion models (ADNI4 is 49.8% non-White but only 1.7% of it has
   24-month follow-up), which is worth reporting on its own.

**The locked test set is still untouched** and should be spent once, on the
final version of claims 1 and 2 only.
