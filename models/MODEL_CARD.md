# Model card — MCI-to-AD conversion risk, cost-aware configuration

## What this is
Logistic-regression models predicting conversion from MCI to AD dementia at
12 / 24 / 36 months, trained on ADNI. Two operating points per horizon:

| model | modalities acquired | relative cost |
|---|---|---|
| `ours_MRI` | structural MRI only | 1.0 |
| `ours_MRI+FDG` | MRI + FDG-PET | 3.0 |

Demographics, cognitive tests (MMSE, CDR-SB) and APOE genotype are treated
as always available and cost nothing — they are in hand before any scan is
ordered.

## Why it exists
Three published policies (UMoS, ProMUSE, a 2025 DQN) try to cut acquisition
cost by deciding per patient which scan to order. An audit in this repo
found none of them beats a fixed rule or random acquisition on ADNI. The
saving they chase is real but does not require a clever policy — it comes
from using the cheap modality better. These models do that and keep the
policy trivial (acquire a fixed set for everyone), which also means they
cannot allocate tests unequally across subgroups, because they never look
at the patient when deciding.

## Performance (nested cross-validation, honest)
Nested CV re-selects the regularisation strength inside every outer fold, so
these figures are not the ones anything was tuned on. Selection optimism was
measured at +0.0016 AUC on average.

| horizon | ours_MRI (cost 1.0) | ours_MRI+FDG (cost 3.0) | acquire-all (cost 9.5) |
|---|---|---|---|
| 12 mo | 0.872 | **0.885** | 0.862 |
| 24 mo | 0.883 | **0.897** | 0.893 |
| 36 mo | 0.860 | 0.885 | 0.898 |

At cost 3.0, accuracy matches or exceeds acquiring all four modalities at
cost 9.5 at two of three horizons (36 mo: 0.013 below, not significant).
Compared with random acquisition at the same cost, ours_MRI is better by
+0.056 (24 mo) and +0.046 (36 mo) AUC, Holm-adjusted p = 0.010 and 0.016;
ours_MRI+FDG is better nominally but not after adjustment.

**Held-out test (scored on two ADNI releases, so not untouched; Oct 2026
results):** ours_MRI 0.944 / 0.908 / 0.855, ours_MRI+FDG 0.951 / 0.901 /
0.903, acquire-all 0.928 / 0.922 / 0.944 at 12 / 24 / 36 months. The test
sets hold only 9 / 17 / 22 converters, so this does not establish
equivalence with acquire-all (see `paper/` and `models/TEST_SET_SPENT.json`).
Quote the nested figures above as expected performance.

## Training data
ADNI, MCI at baseline. Every labelled subject outside the locked test set,
including subjects missing some modalities — each acquisition state is
fitted on everyone holding that state rather than on complete cases only
(563–988 subjects per model, versus 334–402 under complete-case fitting).

## Features
Cheap block: age, education, sex, APOE4 allele count, MMSE, CDR-SB.
MRI block: AD-signature FreeSurfer regions — hippocampus, amygdala,
entorhinal (volume and thickness), fusiform, middle and inferior temporal,
parahippocampal, precuneus, posterior cingulate, ventricles, ICV.
FDG block: MetaROI SUVR.

## Limitations — read before using
- **Not clinically validated.** Retrospective, single cohort, no prospective
  or external validation. Not a medical device.
- **The cohort is 94% White.** ADNI's recent diversity gains (among MCI
  participants ADNI4 is 38.8% non-White) are invisible here because only
  16.3% of ADNI4 has a 24-month label. Performance in under-represented groups is unmeasured, not
  demonstrated to be equal.
- **Cost units are relative and contestable**, not dollars. Conclusions were
  checked under three cost regimes; the dominance result is regime-dependent
  for some comparisons.
- **Acquisition savings are simulated.** ADNI's missingness is
  protocol-driven, not random, so a retrospective estimate of "what we could
  have skipped" is not a prospective guarantee.
- **MRI volumes pool several FreeSurfer pipeline versions** (99% from
  UCSFFSX7). `FS_PIPELINE` records the source and should be adjusted for.
- **Small samples.** 36–86 converters per horizon in the development set;
  confidence intervals are wide and most between-model differences are not
  statistically distinguishable.

## Reproducing
```
python -m src.preprocessing.build_real_cohort   # cohort from raw ADNI tables
python -m src.models.train_final --all-horizons # fit and persist
python -m src.models.nested_cv --all-horizons   # honest performance estimate
```
Artifacts land in `models/` and are gitignored: they are derived from
DUA-governed data and are cheap to regenerate.
