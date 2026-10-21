"""
Experiment: random-acquisition baselines matched to each policy's budget.

WHY THIS EXISTS
  The original benchmark used one random baseline (each modality acquired
  independently with probability 0.5). That is not cost-matched: it costs
  more than almost every published configuration. This script sweeps the
  acquisition rate, repeats every rate over many seeds, and compares each
  policy with random acquisition at the SAME expected cost under each of the
  three cost regimes.

  It also fixes a quirk of the original RandomSubset: it re-seeded its
  generator identically in every CV fold, so the ~n/5 draw patterns of one
  fold were replayed in the other four. The realised cost of "random 0.5" was
  therefore averaged over ~n/5 independent draws, not n, which is why the
  observed price-like cost (5.01-5.10) sat above the theoretical 4.75 =
  0.5 x 9.5. Here the generator is seeded per (seed, fold, rate), so every
  patient receives an independent draw.

DESIGN
  * Same cohort, locked test set, development set and CV folds as
    run_policy_benchmark (StratifiedKFold, seed 42). The test set is not used.
  * One BlockModelBank per fold is fitted once on all 16 modality subsets;
    every random draw then only selects which subset's prediction a patient
    gets, so thousands of draws are cheap.
  * Per rate: AUC mean/SD/2.5-97.5 percentile ACROSS SEEDS (randomness of the
    policy), a subject-bootstrap CI of the seed-averaged AUC (sampling
    uncertainty of the cohort), and realised mean cost under every regime.
  * Matched comparison: for each policy and regime, random acquisition at
    rate = policy_cost / cost_of_all_four (expected-cost match). The policy
    minus seed-averaged random AUC difference gets a paired subject-bootstrap
    CI and a bootstrap p-value (10,000 resamples, +1 corrected so it never
    reads as 0), Holm-adjusted within (horizon, regime) over the policies that
    are not identically random (cheap-only and acquire-all are excluded).

Run:  python -m src.eval.run_random_sweep [--horizon 24] [--seeds 50]
      python -m src.eval.run_random_sweep --no-policies      # sweep only
      python -m src.eval.run_random_sweep --mock --no-policies --seeds 5  # smoke test
"""
from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings(
    "ignore", message="invalid value encountered in reduce", category=RuntimeWarning
)

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold

from src.baselines.base import BlockModelBank
from src.models.features import COST_REGIMES, subjects_with_blocks
from src.preprocessing.cohort import build_cohort
from src.preprocessing.leakage_checks import assert_no_subject_leakage
from src.preprocessing.load_adni import build_subject_visits
from src.preprocessing.splits import subject_level_split

RANDOM_STATE = 42
BLOCKS = ["MRI", "FDG", "AMYLOID", "CSF"]
N_FOLDS = 5
GRID = [round(r, 2) for r in np.arange(0.0, 1.0001, 0.1)]


def fast_auc(y: np.ndarray, p: np.ndarray) -> float:
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_auc(y: np.ndarray, p: np.ndarray, boot_idx: np.ndarray) -> np.ndarray:
    """AUC of (y, p) on every bootstrap resample at once (rows of boot_idx)."""
    yb = y[boot_idx]
    ranks = rankdata(p[boot_idx], axis=1)
    n1 = yb.sum(axis=1)
    n0 = yb.shape[1] - n1
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.sum(ranks * yb, axis=1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def subset_cost_vector(regime: str) -> np.ndarray:
    """Cost of each of the 16 modality subsets; bit i of the code = BLOCKS[i]."""
    c = COST_REGIMES[regime]
    return np.array(
        [sum(c[b] for i, b in enumerate(BLOCKS) if code >> i & 1) for code in range(16)]
    )


def fold_subset_probs(dev: pd.DataFrame, label_col: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per CV fold: (test indices, 16 x n_test matrix of predictions per subset)."""
    y = dev[label_col].astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    out = []
    for tr_idx, te_idx in cv.split(np.zeros(len(dev)), y):
        train, test = dev.iloc[tr_idx], dev.iloc[te_idx]
        sets = [["CHEAP", *[b for i, b in enumerate(BLOCKS) if code >> i & 1]] for code in range(16)]
        bank = BlockModelBank()
        bank.fit(train, label_col, sets)
        P = np.vstack([bank.predict_proba(test, s) for s in sets])
        out.append((te_idx, P))
    return out


def random_oof(folds, n: int, rate: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """One random policy: pooled OOF predictions and acquired-subset codes."""
    probs = np.zeros(n)
    codes = np.zeros(n, dtype=int)
    weights = 1 << np.arange(len(BLOCKS))
    for k, (te_idx, P) in enumerate(folds):
        rng = np.random.default_rng([seed, k, int(round(rate * 1000))])
        mask = rng.random((len(te_idx), len(BLOCKS))) < rate
        code = mask @ weights
        probs[te_idx] = P[code, np.arange(len(te_idx))]
        codes[te_idx] = code
    return probs, codes


def sweep(folds, y: np.ndarray, rates: list[float], n_seeds: int, n_boot: int, rng_seed: int):
    n = len(y)
    boot_idx = np.random.default_rng(rng_seed).integers(0, n, (n_boot, n))
    cost_vec = {r: subset_cost_vector(r) for r in COST_REGIMES}
    rows, store = [], {}
    for rate in rates:
        aucs, cost_by_regime, oofs = [], {r: [] for r in COST_REGIMES}, []
        for s in range(n_seeds):
            p, codes = random_oof(folds, n, rate, s)
            aucs.append(fast_auc(y, p))
            oofs.append(p)
            for reg, cv_ in cost_vec.items():
                cost_by_regime[reg].append(cv_[codes].mean())
        aucs = np.array(aucs)
        oofs = np.vstack(oofs)
        boot = np.nanmean(np.vstack([boot_auc(y, o, boot_idx) for o in oofs]), axis=0)
        row = dict(
            rate=rate, n_seeds=n_seeds, n=n, n_pos=int(y.sum()),
            auc_mean=aucs.mean(), auc_sd=aucs.std(ddof=1) if n_seeds > 1 else np.nan,
            auc_seed_lo=np.percentile(aucs, 2.5), auc_seed_hi=np.percentile(aucs, 97.5),
            auc_ci_lo=np.nanpercentile(boot, 2.5), auc_ci_hi=np.nanpercentile(boot, 97.5),
        )
        for reg in COST_REGIMES:
            row[f"cost_{reg}"] = float(np.mean(cost_by_regime[reg]))
            row[f"cost_{reg}_sd"] = float(np.std(cost_by_regime[reg], ddof=1)) if n_seeds > 1 else np.nan
            row[f"cost_{reg}_expected"] = rate * cost_vec[reg][-1]
        rows.append(row)
        store[rate] = oofs
    return pd.DataFrame(rows), store, boot_idx


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def matched_comparison(policy_oof, policy_cost, policy_cost_sd, y, folds, n_seeds, boot_idx, horizon):
    """Each policy vs random acquisition at the same expected cost, per regime.

    Policies that acquire nothing or everything are skipped: random acquisition
    at rate 0 or 1 IS that policy, so the comparison is identically zero and
    would only inflate the Holm family.
    """
    n = len(y)
    B = len(boot_idx)
    rows = []
    for reg in COST_REGIMES:
        total = subset_cost_vector(reg)[-1]
        reg_rows = []
        for name, p_pol in policy_oof.items():
            c = policy_cost[name][reg]
            if c <= 1e-9 or c >= total - 1e-9:
                continue
            rate = float(np.clip(c / total, 0.0, 1.0))
            draws = [random_oof(folds, n, rate, s_) for s_ in range(n_seeds)]
            oofs = np.vstack([d[0] for d in draws])
            rand_auc = np.mean([fast_auc(y, o) for o in oofs])
            point = fast_auc(y, p_pol) - rand_auc
            deltas = boot_auc(y, p_pol, boot_idx) - np.nanmean(
                np.vstack([boot_auc(y, o, boot_idx) for o in oofs]), axis=0)
            deltas = deltas[~np.isnan(deltas)]
            # two-sided bootstrap p with the +1 correction: never below 1/(B+1)
            p_two = 2 * min(((deltas <= 0).sum() + 1) / (len(deltas) + 1),
                            ((deltas >= 0).sum() + 1) / (len(deltas) + 1))
            seed_costs = np.array([subset_cost_vector(reg)[d[1]].mean() for d in draws])
            cvec = np.array([COST_REGIMES[reg][b] for b in BLOCKS])
            rand_pat_sd = float(np.sqrt(rate * (1 - rate) * np.sum(cvec ** 2)))
            reg_rows.append(dict(
                horizon=horizon, regime=reg, policy=name, policy_cost=c,
                policy_auc=fast_auc(y, p_pol), matched_rate=rate,
                random_cost_realised=seed_costs.mean(),
                random_cost_sd_seeds=seed_costs.std(ddof=1),
                random_cost_sd_patient=rand_pat_sd,
                policy_cost_sd_patient=policy_cost_sd[name][reg],
                cost_gap=seed_costs.mean() - c, random_auc=rand_auc,
                delta_auc=point, lo=np.percentile(deltas, 2.5), hi=np.percentile(deltas, 97.5),
                p_boot=min(1.0, p_two), n_boot=B,
            ))
        adj = holm(np.array([r["p_boot"] for r in reg_rows]))
        for r, a_ in zip(reg_rows, adj):
            r["family_size"] = len(reg_rows)
            r["p_holm"] = a_
            r["significant_ci"] = bool(r["lo"] > 0 or r["hi"] < 0)
            r["significant_holm"] = bool(a_ < 0.05)
        rows.extend(reg_rows)
    return pd.DataFrame(rows)


def load_dev(horizon: int, mock: bool):
    label_col = f"LABEL_{horizon}mo"
    if mock:
        from src.preprocessing.mock_adnimerge import generate_mock_adnimerge

        visits = generate_mock_adnimerge(n_subjects=3000, seed=0)
    else:
        visits = build_subject_visits()
    cohort = build_cohort(visits)
    labelled = cohort[cohort[label_col].notna()].copy()
    complete = subjects_with_blocks(labelled, BLOCKS)
    split = subject_level_split(complete, label_col=label_col, random_state=RANDOM_STATE)
    assert_no_subject_leakage(split)
    dev = complete[complete["RID"].isin(split.train_rids | split.val_rids)].copy()
    extra_pool = labelled[
        (~labelled["RID"].isin(split.test_rids)) & (~labelled["RID"].isin(dev["RID"]))
    ].copy()
    return dev, extra_pool, label_col


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, nargs="+", default=[12, 24, 36])
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--boot", type=int, default=1000, help="resamples for the sweep CIs")
    ap.add_argument("--boot-matched", type=int, default=10000,
                    help="resamples for the matched comparison (p-values resolve to 1/(B+1))")
    ap.add_argument("--no-policies", action="store_true",
                    help="skip the matched comparison (no torch/RL needed)")
    ap.add_argument("--quick", action="store_true", help="fewer RL episodes")
    ap.add_argument("--mock", action="store_true", help="synthetic data, for smoke tests only")
    args = ap.parse_args()

    for horizon in args.horizon:
        dev, extra_pool, label_col = load_dev(horizon, args.mock)
        y = dev[label_col].astype(int).to_numpy()
        print(f"\n=== {horizon} mo: n={len(dev)}, converters={int(y.sum())} ===", flush=True)
        folds = fold_subset_probs(dev, label_col)

        table, _, boot_idx = sweep(folds, y, GRID, args.seeds, args.boot, RANDOM_STATE)
        tag = "_mock" if args.mock else ""
        out = f"paper/tables/random_sweep_{horizon}mo{tag}.csv"
        table.round(4).to_csv(out, index=False)
        show = ["rate", "auc_mean", "auc_sd", "auc_ci_lo", "auc_ci_hi", "cost_price_like",
                "cost_price_like_expected", "cost_flat", "cost_burden"]
        print(table[show].round(3).to_string(index=False))
        print(f"wrote {out}")

        if args.no_policies:
            continue
        from src.eval.run_policy_benchmark import evaluate_policy, policy_factories

        policy_oof, policy_cost, policy_cost_sd = {}, {}, {}
        for name, factory in policy_factories(quick=args.quick, extra_pool=extra_pool).items():
            if name.startswith("random_"):
                continue  # replaced by the sweep
            print(f"running {name} ...", flush=True)
            row, probs, acquired = evaluate_policy(name, factory, dev, label_col)
            policy_oof[name] = probs
            policy_cost[name] = {r: row[f"cost_{r}"] for r in COST_REGIMES}
            policy_cost_sd[name] = {
                r: float(np.std([sum(COST_REGIMES[r][b] for b in blk) for blk in acquired], ddof=1))
                for r in COST_REGIMES}
        boot_idx_m = np.random.default_rng(RANDOM_STATE + 1).integers(
            0, len(y), (args.boot_matched, len(y)))
        matched = matched_comparison(
            policy_oof, policy_cost, policy_cost_sd, y, folds, args.seeds, boot_idx_m, horizon
        )
        out = f"paper/tables/random_matched_{horizon}mo.csv"
        matched.round(4).to_csv(out, index=False)
        print(matched[matched.regime == "price_like"][
            ["policy", "policy_cost", "policy_auc", "random_cost_realised",
             "random_auc", "delta_auc", "lo", "hi", "p_holm"]].round(3).to_string(index=False))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
