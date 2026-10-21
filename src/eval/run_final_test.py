"""
FINAL CONFIRMATORY EVALUATION ON THE LOCKED TEST SET.

*** RUN THIS ONCE, DELIBERATELY, WHEN THE ANALYSIS IS FINAL. ***

The test set has been held out untouched through every experiment in this
project. Its value comes entirely from having been looked at exactly once:
if it is run, inspected, and then used to guide another modelling decision,
it stops being a held-out set and becomes a second validation set, and the
headline numbers stop meaning what they claim to mean.

So this script is deliberately separate, requires an explicit --i-am-sure
flag, and records the fact that it ran. If you find yourself wanting to run
it twice, the honest move is to report the first result.

It evaluates the persisted models from src/models/train_final.py against the
comparison points the paper claims to beat.

Run:  python -m src.eval.run_final_test --i-am-sure [--horizon 24]
"""
from __future__ import annotations

import argparse
import json
import pickle
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore", message="invalid value encountered in reduce", category=RuntimeWarning
)

from src.baselines.base import BlockModelBank
from src.eval.metrics import auc, bootstrap_ci, bootstrap_delta_ci, brier
from src.models.features import COST_REGIMES, subjects_with_blocks
from src.models.train_final import OPERATING_POINTS, RANDOM_STATE
from src.preprocessing.cohort import build_cohort
from src.preprocessing.leakage_checks import assert_no_subject_leakage
from src.preprocessing.load_adni import build_subject_visits
from src.preprocessing.splits import subject_level_split

BLOCKS = ["MRI", "FDG", "AMYLOID", "CSF"]
RECEIPT = Path("models/TEST_SET_SPENT.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--i-am-sure", action="store_true",
                    help="required: confirms you intend to spend the held-out test set")
    args = ap.parse_args()

    if not args.i_am_sure:
        print(__doc__)
        print("Refusing to run without --i-am-sure.")
        if RECEIPT.exists():
            print(f"\nNOTE: {RECEIPT} exists -- the test set has already been spent:")
            print(RECEIPT.read_text(encoding="utf-8"))
        return

    if RECEIPT.exists():
        print(f"WARNING: {RECEIPT} already exists. The test set has been spent before:")
        print(RECEIPT.read_text(encoding="utf-8"))
        print("\nRunning again does not restore its validity. Report every scoring and state it.")

    label_col = f"LABEL_{args.horizon}mo"
    cohort = build_cohort(build_subject_visits())
    labelled = cohort[cohort[label_col].notna()].copy()
    complete = subjects_with_blocks(labelled, BLOCKS)
    split = subject_level_split(complete, label_col=label_col, random_state=RANDOM_STATE)
    assert_no_subject_leakage(split)

    test = complete[complete["RID"].isin(split.test_rids)].copy()
    dev = complete[complete["RID"].isin(split.train_rids | split.val_rids)].copy()
    pool = labelled[
        (~labelled["RID"].isin(split.test_rids)) & (~labelled["RID"].isin(dev["RID"]))
    ].copy()
    y = test[label_col].astype(int).to_numpy()

    print(f"{args.horizon}-month horizon -- LOCKED TEST SET")
    print(f"  n={len(test)}, converters={int(y.sum())}\n")
    if len(np.unique(y)) < 2:
        print("Test set has a single class; cannot evaluate.")
        return

    rows, preds = [], {}

    # Our persisted models.
    for name in OPERATING_POINTS:
        path = Path(f"models/{name}_{args.horizon}mo.pkl")
        if not path.exists():
            print(f"[skip] {path} missing -- run src.models.train_final first")
            continue
        with open(path, "rb") as f:
            art = pickle.load(f)
        p = art["bank"].predict_proba(test, art["blocks"])
        preds[name] = p
        lo, hi = bootstrap_ci(y, p, auc, seed=RANDOM_STATE)
        rows.append(dict(model=name, cost=sum(COST_REGIMES["price_like"][b]
                                              for b in art["blocks"] if b != "CHEAP"),
                         auc=round(auc(y, p), 4), lo=round(lo, 4), hi=round(hi, 4),
                         brier=round(brier(y, p), 4)))

    # Comparison points, refitted on development data only.
    for label, blocks in (("cheap_only", []), ("fixed_MRI+FDG", ["MRI", "FDG"]),
                          ("acquire_all", BLOCKS)):
        bank = BlockModelBank()
        bank.fit(dev, label_col, [["CHEAP", *blocks]])
        p = bank.predict_proba(test, ["CHEAP", *blocks])
        preds[label] = p
        lo, hi = bootstrap_ci(y, p, auc, seed=RANDOM_STATE)
        rows.append(dict(model=label,
                         cost=sum(COST_REGIMES["price_like"][b] for b in blocks),
                         auc=round(auc(y, p), 4), lo=round(lo, 4), hi=round(hi, 4),
                         brier=round(brier(y, p), 4)))

    res = pd.DataFrame(rows).sort_values("cost")
    print("=== TEST-SET RESULTS ===")
    print(res.to_string(index=False))

    print("\n=== PAIRED vs acquire_all (the claim: same accuracy, lower cost) ===")
    comp = []
    if "acquire_all" in preds:
        for name, p in preds.items():
            if name == "acquire_all":
                continue
            d, lo, hi = bootstrap_delta_ci(y, preds["acquire_all"], p, auc, seed=RANDOM_STATE)
            comp.append(dict(model=name, delta_vs_acquire_all=round(d, 4),
                             lo=round(lo, 4), hi=round(hi, 4),
                             worse=bool(hi < 0), better=bool(lo > 0)))
        print(pd.DataFrame(comp).to_string(index=False))

    res.to_csv(f"paper/tables/FINAL_TEST_{args.horizon}mo.csv", index=False)
    # Append, never overwrite: every scoring of the test set stays on the record.
    runs = []
    if RECEIPT.exists():
        prev = json.loads(RECEIPT.read_text(encoding="utf-8"))
        runs = prev["runs"] if isinstance(prev, dict) and "runs" in prev else [prev]
    runs.append({
        "scored_utc": datetime.now(timezone.utc).isoformat(),
        "horizon": args.horizon,
        "n_test": len(test),
        "n_converters": int(y.sum()),
        "data_release": ", ".join(sorted(f.name for f in Path("data/raw").glob("DATADIC_*.csv"))) or "unknown",
    })
    RECEIPT.write_text(json.dumps({"runs": runs}, indent=2), encoding="utf-8")
    print(f"\nWrote paper/tables/FINAL_TEST_{args.horizon}mo.csv and {RECEIPT}")


if __name__ == "__main__":
    main()
