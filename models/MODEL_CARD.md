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
measured at +0.0008 AUC on average.

| horizon | ours_MRI (cost 1.0) | ours_MRI+FDG (cost 3.0) | acquire-all (cost 9.5) |
|---|---|---|---|
| 12 mo | 0.873 | **0.885** | 0.863 |
| 24 mo | 0.884 | **0.896** | 0.894 |
| 36 mo | 0.864 | 0.888 | 0.900 |

At cost 3.0, accuracy matches or exceeds acquiring all four modalities at
cost 9.5 at two of three horizons, and is 0.012 below at 36 months.

**The held-out test set has NOT been used.** These are development-set
figures. Run `src/eval/run_final_test.py --i-am-sure` once, when the
analysis is final, for the confirmatory number.

## Training data
ADNI, MCI at baseline. Every labelled subject outside the locked test set,
including subjects missing some modalities — each acquisition state is
fitted on everyone holding that state rather than on complete cases only
(559–910 subjects per model, versus 333–402 under complete-case fitting).

## Features
Cheap block: age, education, sex, APOE4 allele count, MMSE, CDR-SB.
MRI block: AD-signature FreeSurfer regions — hippocampus, amygdala,
entorhinal (volume and thickness), fusiform, middle and inferior temporal,
parahippocampal, precuneus, posterior cingulate, ventricles, ICV.
FDG block: MetaROI SUVR.

## Limitations — read before using
- **Not clinically validated.** Retrospective, single cohort, no prospective
  or external validation. Not a medical device.
- **The cohort is 94% White.** ADNI's recent diversity gains (ADNI4 is 49.8%
  non-White) are invisible here because only 1.7% of ADNI4 has 24-month
  follow-up. Performance in under-represented groups is unmeasured, not
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
