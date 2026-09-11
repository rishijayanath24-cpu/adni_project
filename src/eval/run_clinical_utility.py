"""
Experiment 6: calibration and clinical utility of the acquisition policies.

AUC is a ranking metric. It says nothing about whether a policy's predicted
risks are trustworthy numbers (calibration) or whether acting on them beats
the default strategies of treating everyone or no one (net benefit). The AD
prognosis literature reports AUC almost exclusively, which is precisely why
this project committed to reporting more.

It also matters for the central negative result. If no policy separates on
AUC, the interesting question becomes whether they differ on the axes that
actually determine clinical usefulness. A policy could be equally
discriminating but badly calibrated -- which would make its risk estimates
unusable for a real "should we scan this patient" decision even though its
AUC looked fine.

Reported per policy:
  * reliability table (predicted vs observed risk by bin) and Brier score
  * decision curve: net benefit across threshold probabilities, against
    treat-all and treat-none

Run:  python -m src.eval.run_clinical_utility [--horizon 24]
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore", message="invalid value encountered in reduce", category=RuntimeWarning
)

from src.eval.metrics import brier, calibration_table, decision_curve
from src.eval.run_policy_benchmark import RANDOM_STATE, evaluate_policy, policy_factories
from src.eval.run_robustness import load_dev
from src.models.features import subjects_with_blocks
from src.preprocessing.cohort import build_cohort
from src.preprocessing.load_adni import build_subject_visits
from src.preprocessing.splits import subject_level_split
from src.eval.run_policy_benchmark import BLOCKS

POLICIES = [
    "cheap_only",
    "ours_MRI",
    "ours_MRI+FDG",
    "fixed_MRI+FDG",
    "umos_mri_to_amyloid",
    "promuse_full_ladder_sharedpred",
    "dqn_lambda0.05",
    "random_rate0.5",
    "acquire_all",
]


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Calibration slope and intercept from a logistic recalibration fit.

    Slope < 1 means predictions are too extreme (over-fit risk estimates);
    intercept away from 0 means systematic over- or under-prediction. These
    are the standard summaries a clinical-prediction reviewer expects, and
    they carry information a Brier score alone hides.
    """
    from sklearn.linear_model import LogisticRegression

    eps = 1e-6
    logit = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    # C=inf is the unpenalised fit; penalty=None is deprecated from sklearn 1.8.
    lr = LogisticRegression(C=np.inf, max_iter=1000)
    lr.fit(logit.reshape(-1, 1), y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def load_dev_with_pool(horizon: int):
    """Development cohort plus the partially-observed pool our config trains on."""
    label_col = f"LABEL_{horizon}mo"
    cohort = build_cohort(build_subject_visits())
    labelled = cohort[cohort[label_col].notna()].copy()
    complete = subjects_with_blocks(labelled, BLOCKS)
    split = subject_level_split(complete, label_col=label_col, random_state=RANDOM_STATE)
    dev = complete[complete["RID"].isin(split.train_rids | split.val_rids)].copy()
    pool = labelled[
        (~labelled["RID"].isin(split.test_rids)) & (~labelled["RID"].isin(dev["RID"]))
    ].copy()
    return dev, label_col, pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=24)
    args = ap.parse_args()

    dev, label_col, extra_pool = load_dev_with_pool(args.horizon)
    y = dev[label_col].astype(int).to_numpy()
    prevalence = float(y.mean())
    print(f"{args.horizon}-month horizon: n={len(dev)}, converters={int(y.sum())} "
          f"(prevalence {prevalence:.3f})\n")

    factories = policy_factories(extra_pool=extra_pool)
    cal_rows, dca_rows, summary = [], [], []

    for name in POLICIES:
        if name not in factories:
            continue
        print(f"running {name} ...", flush=True)
        _, p, _ = evaluate_policy(name, factories[name], dev, label_col)

        slope, intercept = calibration_slope_intercept(y, p)
        summary.append(
            dict(policy=name, brier=round(brier(y, p), 4),
                 cal_slope=round(slope, 3), cal_intercept=round(intercept, 3),
                 mean_predicted=round(float(p.mean()), 4), observed=round(prevalence, 4))
        )

        c = calibration_table(y, p, n_bins=5); c.insert(0, "policy", name)
        cal_rows.append(c)
        d = decision_curve(y, p); d.insert(0, "policy", name)
        dca_rows.append(d)

    s = pd.DataFrame(summary)
    print()
    print("=== CALIBRATION SUMMARY (slope 1.0 and intercept 0.0 are perfect) ===")
    print(s.to_string(index=False))

    cal = pd.concat(cal_rows, ignore_index=True)
    dca = pd.concat(dca_rows, ignore_index=True)

    print()
    print("=== NET BENEFIT by threshold (higher is better; compare to treat_all/none) ===")
    piv = dca.pivot_table(index="threshold", columns="policy", values="model")
    ta = dca.groupby("threshold")["treat_all"].first()
    piv.insert(0, "treat_all", ta)
    piv["treat_none"] = 0.0
    print(piv.round(4).to_string())

    print()
    best = piv.drop(columns=["treat_none"]).idxmax(axis=1)
    print("Best strategy at each threshold:")
    print(best.to_string())

    s.to_csv(f"paper/tables/calibration_summary_{args.horizon}mo.csv", index=False)
    cal.to_csv(f"paper/tables/calibration_bins_{args.horizon}mo.csv", index=False)
    dca.to_csv(f"paper/tables/decision_curve_{args.horizon}mo.csv", index=False)
    print(f"\nWrote paper/tables/calibration_*_{args.horizon}mo.csv and decision_curve_{args.horizon}mo.csv")


if __name__ == "__main__":
    main()
