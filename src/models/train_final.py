"""
Train and persist the final production models.

This fits the configuration the ablation justified -- AD-signature MRI
features plus pooled training -- at each prediction horizon, on ALL labelled
subjects outside the locked test set.

SCOPE, deliberately bounded: this script trains and saves. It does NOT touch
the held-out test set. That evaluation is one-shot and is left to
src/eval/run_final_test.py, to be run once when the analysis is final.

Two operating points are trained, because the project's claim is about the
cost-accuracy frontier rather than a single model:
  ours_MRI      -- cost 1.0, the headline "all-modality accuracy for an MRI"
  ours_MRI+FDG  -- cost 3.0, the best accuracy we reach before returns flatten

Hyperparameters are selected by cross-validation on the development set
only. Nothing is chosen on test.

Run:  python -m src.models.train_final [--horizon 24] [--all-horizons]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings(
    "ignore", message="invalid value encountered in reduce", category=RuntimeWarning
)

from src.eval.metrics import auc, bootstrap_ci, brier
from src.models.features import COST_REGIMES, feature_columns, subjects_with_blocks
from src.models.pooled_bank import PooledBlockModelBank
from src.preprocessing.cohort import HORIZONS, build_cohort
from src.preprocessing.leakage_checks import assert_no_subject_leakage
from src.preprocessing.load_adni import build_subject_visits
from src.preprocessing.splits import subject_level_split

RANDOM_STATE = 42
N_FOLDS = 5
C_GRID = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
OPERATING_POINTS = {"ours_MRI": ["MRI"], "ours_MRI+FDG": ["MRI", "FDG"]}
MODEL_DIR = Path("models")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def prepare(horizon: int):
    """Development cohort, training pool, and the locked test RIDs (not used)."""
    label_col = f"LABEL_{horizon}mo"
    cohort = build_cohort(build_subject_visits())
    labelled = cohort[cohort[label_col].notna()].copy()
    complete = subjects_with_blocks(labelled, ["MRI", "FDG", "AMYLOID", "CSF"])
    split = subject_level_split(complete, label_col=label_col, random_state=RANDOM_STATE)
    assert_no_subject_leakage(split)

    dev = complete[complete["RID"].isin(split.train_rids | split.val_rids)].copy()
    pool = labelled[
        (~labelled["RID"].isin(split.test_rids)) & (~labelled["RID"].isin(dev["RID"]))
    ].copy()
    return dev, pool, label_col, set(split.test_rids)


def select_C(dev, pool, label_col, blocks) -> tuple[float, float]:
    """Pick the regularisation strength by CV on development data only."""
    y = dev[label_col].astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    best_c, best_auc = None, -np.inf
    for c in C_GRID:
        probs = np.zeros(len(dev))
        for tr, te in cv.split(np.zeros(len(dev)), y):
            bank = PooledBlockModelBank(extra_pool=pool, random_state=RANDOM_STATE, C=c)
            bank.fit(dev.iloc[tr], label_col, [["CHEAP", *blocks]])
            probs[te] = bank.predict_proba(dev.iloc[te], ["CHEAP", *blocks])
        a = auc(y, probs)
        print(f"      C={c:<6} CV AUC={a:.4f}")
        if a > best_auc:
            best_c, best_auc = c, a
    return best_c, best_auc


def train_one(horizon: int) -> list[dict]:
    dev, pool, label_col, test_rids = prepare(horizon)
    y = dev[label_col].astype(int).to_numpy()
    print(f"\n=== {horizon}-month horizon ===")
    print(f"  development: n={len(dev)}, converters={int(y.sum())}")
    print(f"  pooled training adds up to {len(pool)} partially-observed subjects")
    print(f"  locked test set: {len(test_rids)} subjects (NOT touched here)")

    MODEL_DIR.mkdir(exist_ok=True)
    records = []

    for name, blocks in OPERATING_POINTS.items():
        print(f"\n  -- {name} (blocks: {'+'.join(blocks)}) --")
        print("    selecting regularisation by CV on development data:")
        best_c, cv_auc = select_C(dev, pool, label_col, blocks)
        print(f"    chosen C={best_c} (CV AUC {cv_auc:.4f})")

        # Honest CV estimate at the chosen C, with an interval, so the model
        # card carries an expected-performance figure that is not the
        # selection-optimised maximum.
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        probs = np.zeros(len(dev))
        for tr, te in cv.split(np.zeros(len(dev)), y):
            b = PooledBlockModelBank(extra_pool=pool, random_state=RANDOM_STATE, C=best_c)
            b.fit(dev.iloc[tr], label_col, [["CHEAP", *blocks]])
            probs[te] = b.predict_proba(dev.iloc[te], ["CHEAP", *blocks])
        lo, hi = bootstrap_ci(y, probs, auc, seed=RANDOM_STATE)

        # Final fit: every labelled subject outside the locked test set.
        final = PooledBlockModelBank(extra_pool=pool, random_state=RANDOM_STATE, C=best_c)
        final.fit(dev, label_col, [["CHEAP", *blocks]])
        key = tuple(sorted(["CHEAP", *blocks]))
        n_train = final.train_sizes.get(key, len(dev))
        print(f"    final fit on {n_train} subjects")

        import pickle

        path = MODEL_DIR / f"{name}_{horizon}mo.pkl"
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "bank": final,
                    "blocks": ["CHEAP", *blocks],
                    "horizon": horizon,
                    "label_col": label_col,
                    "C": best_c,
                    "features": feature_columns(["CHEAP", *blocks]),
                },
                f,
            )

        rec = dict(
            model=name, horizon=horizon, blocks="+".join(blocks),
            acquisition_cost=sum(COST_REGIMES["price_like"][b] for b in blocks),
            C=best_c, n_train_final=int(n_train),
            n_dev=len(dev), n_dev_converters=int(y.sum()),
            cv_auc=round(auc(y, probs), 4), cv_auc_lo=round(lo, 4), cv_auc_hi=round(hi, 4),
            cv_brier=round(brier(y, probs), 4),
            artifact=str(path).replace("\\", "/"),
        )
        records.append(rec)
        print(f"    CV AUC {rec['cv_auc']:.4f} [{rec['cv_auc_lo']:.4f}, {rec['cv_auc_hi']:.4f}]"
              f"  Brier {rec['cv_brier']:.4f}  -> saved {path}")

    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--all-horizons", action="store_true")
    args = ap.parse_args()

    horizons = HORIZONS if args.all_horizons else (args.horizon,)
    all_records = []
    for h in horizons:
        all_records += train_one(h)

    df = pd.DataFrame(all_records)
    print("\n=== TRAINED MODELS ===")
    print(df.to_string(index=False))

    MODEL_DIR.mkdir(exist_ok=True)
    df.to_csv(MODEL_DIR / "model_registry.csv", index=False)
    with open(MODEL_DIR / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "trained_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": _git_commit(),
                "random_state": RANDOM_STATE,
                "C_grid": C_GRID,
                "test_set_touched": Path("models/TEST_SET_SPENT.json").exists(),  # training never reads the test set; this records whether it has since been scored
                "models": all_records,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {MODEL_DIR}/model_registry.csv and training_metadata.json")
    print("Locked test set was NOT used. Run src/eval/run_final_test.py to spend it.")


if __name__ == "__main__":
    main()
