"""
Why do held-out test AUCs sit above development AUCs, especially at 12 months?

DESCRIPTIVE ONLY. Nothing here feeds back into modelling: no feature, model,
operating point or hyperparameter is chosen from these numbers, and the test
set is used only for the AUCs already reported in FINAL_TEST_*.csv plus
baseline covariates (no outcome-conditioned model fitting on test rows).

Two checks per horizon:
  1. SAMPLING NULL. Draw many random subsets of the development out-of-fold
     predictions with the same size and number of converters as the test set,
     and place the observed test AUC in that distribution. With 9-22
     converters a subset AUC is very noisy, so a gap of +0.05-0.10 can be
     chance; the percentile says how surprising it is. Subsets are drawn
     without replacement from the (n_dev) pooled OOF predictions, so this
     assumes test and development are exchangeable -- which check 2 probes.
  2. COVARIATE SHIFT. Baseline characteristics of test vs development
     subjects (standardised mean differences), overall and among converters.

Run:  python -m src.eval.investigate_test_gap
Writes paper/tables/test_gap_null.csv and paper/tables/test_gap_covariates.csv
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.eval.metrics import auc
from src.eval.run_policy_benchmark import evaluate_policy, policy_factories
from src.eval.run_random_sweep import BLOCKS, RANDOM_STATE, fast_auc
from src.models.features import subjects_with_blocks
from src.preprocessing.cohort import build_cohort
from src.preprocessing.leakage_checks import assert_no_subject_leakage
from src.preprocessing.load_adni import build_subject_visits
from src.preprocessing.splits import subject_level_split

MODELS = ["cheap_only", "fixed_MRI+FDG", "ours_MRI", "ours_MRI+FDG", "acquire_all"]
COVARIATES = ["AGE", "PTEDUCAT", "MMSE_bl", "CDRSB_bl", "APOE4", "Hippocampus_bl", "Ventricles_bl"]
N_DRAWS = 20000


def subset_null(y, p, n_test, n_pos_test, rng):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    out = np.empty(N_DRAWS)
    for k in range(N_DRAWS):
        idx = np.concatenate([rng.choice(pos, n_pos_test, replace=False),
                              rng.choice(neg, n_test - n_pos_test, replace=False)])
        out[k] = fast_auc(y[idx], p[idx])
    return out


def smd(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna(), b.dropna()
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((b.mean() - a.mean()) / sd) if sd > 0 else float("nan")


def main() -> None:
    cohort = build_cohort(build_subject_visits())
    rng = np.random.default_rng(RANDOM_STATE)
    null_rows, cov_rows = [], []
    for h in (12, 24, 36):
        L = f"LABEL_{h}mo"
        labelled = cohort[cohort[L].notna()].copy()
        complete = subjects_with_blocks(labelled, BLOCKS)
        split = subject_level_split(complete, label_col=L, random_state=RANDOM_STATE)
        assert_no_subject_leakage(split)
        test = complete[complete["RID"].isin(split.test_rids)]
        dev = complete[complete["RID"].isin(split.train_rids | split.val_rids)]
        pool = labelled[(~labelled["RID"].isin(split.test_rids)) & (~labelled["RID"].isin(dev["RID"]))]
        y = dev[L].astype(int).to_numpy()
        n_test, n_pos_test = len(test), int(test[L].sum())
        facs = {k: v for k, v in policy_factories(quick=True, extra_pool=pool).items() if k in MODELS}
        final = pd.read_csv(f"paper/tables/FINAL_TEST_{h}mo.csv").set_index("model")
        for name in MODELS:
            _, oof, _ = evaluate_policy(name, facs[name], dev, L)
            dev_auc = auc(y, oof)
            null = subset_null(y, oof, n_test, n_pos_test, rng)
            test_auc = float(final.loc[name, "auc"])
            null_rows.append(dict(
                horizon=h, model=name, dev_auc=round(dev_auc, 4), test_auc=round(test_auc, 4),
                gap=round(test_auc - dev_auc, 4), null_sd=round(null.std(ddof=1), 4),
                null_lo=round(np.percentile(null, 2.5), 4), null_hi=round(np.percentile(null, 97.5), 4),
                percentile=round(100 * float((null <= test_auc).mean()), 1),
                n_test=n_test, n_pos_test=n_pos_test,
            ))
            print(null_rows[-1], flush=True)
        for c in COVARIATES:
            cov_rows.append(dict(
                horizon=h, covariate=c,
                dev_mean=round(dev[c].mean(), 3), test_mean=round(test[c].mean(), 3),
                smd_all=round(smd(dev[c], test[c]), 3),
                smd_converters=round(smd(dev.loc[dev[L] == 1, c], test.loc[test[L] == 1, c]), 3),
                smd_nonconverters=round(smd(dev.loc[dev[L] == 0, c], test.loc[test[L] == 0, c]), 3),
            ))
        cov_rows.append(dict(horizon=h, covariate="prevalence",
                             dev_mean=round(float(y.mean()), 3), test_mean=round(n_pos_test / n_test, 3)))
    pd.DataFrame(null_rows).to_csv("paper/tables/test_gap_null.csv", index=False)
    pd.DataFrame(cov_rows).to_csv("paper/tables/test_gap_covariates.csv", index=False)
    print(pd.DataFrame(null_rows).to_string(index=False))
    print(pd.DataFrame(cov_rows).to_string(index=False))


if __name__ == "__main__":
    main()
