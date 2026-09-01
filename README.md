# Fairness- and Horizon-Audited Sequential Modality Acquisition for Alzheimer's Risk Stratification on ADNI

## Status
Design phase complete, literature novelty check in progress (moderate-to-good confidence, pending a formal Scopus/Web of Science search — see `docs/search_protocol.md` once the search is actually run). Implementation not yet started.

## What this is
Not a new fusion architecture. A **fairness and prediction-horizon audit** of three existing, published, cost-aware sequential modality-acquisition policies for Alzheimer's diagnosis on ADNI:

- **UMoS** (Zheng et al., *Alzheimer's & Dementia* 2022 / *IISE Trans. Healthcare Sys. Eng.* 2023) — MRI-model uncertainty gates amyloid-PET acquisition.
- **ProMUSE** (arXiv:2606.19371, 2026) — staged evidential (Dirichlet/Dempster-Shafer) uncertainty gating, clinical → MRI → PET.
- **Sequential-decision RL policy** (ACM ICBDSIC 2025) — DQN/PPO test-ordering policy.

None of these three report (a) disparate acquisition/error rates by race, ethnicity, or sex, or (b) whether the optimal policy changes with the prediction horizon (12 vs. 24 vs. 36 months). This project re-implements all three as baselines, then adds both audits, plus a fairness-constrained and horizon-conditioned policy variant.

**Do not claim "first to do sequential/cost-aware acquisition on ADNI."** That claim is false — see `docs/`. The claim is: first fairness + horizon audit of this specific line of work.

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
Uses **ADNIMERGE only** (16,378 subject-visit rows, 115 variables — cognition, MRI-derived volumes, PET SUVR, CSF biomarkers, APOE, diagnosis). No raw MRI/PET image preprocessing is required for the core result. Cross-reference `DXSUM`, `UCBERKELEY PET SUVR`, `UPENN CSF`, `APOERES` only if ADNIMERGE fields are ambiguous for a given subject-visit.

## Non-negotiable rigor rules
- **Subject-level train/val/test splits, enforced before any preprocessing statistic is computed.** This is the single largest source of inflated accuracy documented in this field (a 28-point swing in one directly-audited case).
- Stratify and report separately by scanner/field-strength (ADNI-1 1.5T vs. ADNI-2/3 3T).
- MCI reverters are a distinct label category, never silently merged into "stable."
- Report calibration (Brier score, reliability) and decision-curve analysis, not AUC alone.
- Report fairness metrics (equalized odds, equal opportunity, demographic parity) for every model, including the three re-implemented baselines.

## Milestones
1. ADNI data-use access request submitted (scope: ADNIMERGE + supplementary tables, not raw imaging archive).
2. Cohort construction + leakage-safe subject-level split.
3. Re-implement and validate UMoS, ProMUSE, DQN-2025 against their published numbers.
4. Fairness audit of all three baselines' acquisition behavior.
5. Horizon-conditioned acquisition policy + audit.
6. Fairness-constrained policy variant.
7. Full evaluation: cost-accuracy Pareto frontiers, calibration, fairness tables, horizon-behavior analysis.
8. Re-run the systematic literature search (see full protocol in the project review artifact) before finalizing the novelty claim in the manuscript.
9. Write-up.

## Known open risk
The novelty check so far is web-search-based (WebSearch/WebFetch across arXiv, PubMed, Europe PMC), not a formal Scopus/Web of Science systematic search. Treat the novelty claim as "likely, not confirmed" until that search is actually run — protocol is ready, execution is not done yet.
