"""
Experiment 3: benchmark the published acquisition policies against each
other and against the frontier's reference endpoints.

Policies compared (all re-implementations -- see each module's docstring for
the adaptations made and why):
  cheap_only / acquire_all / fixed_* / random_*  -- frontier endpoints and
      null comparisons that any learned policy must beat to mean anything
  umos      -- uncertainty-threshold cascade (Zheng et al. 2022/2023)
  promuse   -- staged evidential + Dempster-Shafer fusion (arXiv:2606.19371)
  dqn       -- RL test-ordering policy (ACM ICBDSIC 2025)

PROTOCOL
  * Evaluated on a COMPLETE-CASE cohort for the blocks in play, so every
    policy could in principle have acquired anything. Without this a policy
    that happens to prefer the most-available modality wins for the wrong
    reason.
  * Cross-validated on the development set with pooled out-of-fold
    predictions. The TEST SET IS NOT TOUCHED -- it stays locked for the
    final confirmatory comparison, so these numbers cannot be
    cherry-picked into a favourable story.
  * Each policy is refitted inside every fold, including its internal
    thresholds, so no threshold is ever chosen on the data it is scored on.
  * TAU is excluded from the block set (only ~105-125 labelled subjects hold
    it, and 14-24 converters) and PLASMA is excluded (almost no plasma
    subject has conversion follow-up yet). Both are documented in the
    README rather than silently dropped.

Run:  python -m src.eval.run_policy_benchmark [--horizon 24] [--quick]
"""
from __future__ import annotations

import argparse
import time
import warnings

# sklearn's finiteness check sums the input array before imputation, which
# numpy warns about when NaNs are present. NaNs there are expected -- the
# imputer exists to handle them -- so this one warning is silenced narrowly
# rather than filtering everything and hiding real problems.
warnings.filterwarnings(
    "ignore", message="invalid value encountered in reduce", category=RuntimeWarning
)

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.baselines.dqn import DQNPolicy
from src.baselines.ours import OursFixed
from src.baselines.promuse import ProMUSE
from src.baselines.reference import AcquireAll, CheapOnly, FixedOrder, RandomSubset
from src.baselines.umos import UMoS
from src.eval.metrics import auc, bootstrap_ci, bootstrap_delta_ci, brier
from src.models.features import COST_REGIMES, subjects_with_blocks
from src.preprocessing.cohort import build_cohort
from src.preprocessing.leakage_checks import assert_no_subject_leakage
from src.preprocessing.load_adni import build_subject_visits
from src.preprocessing.splits import subject_level_split

RANDOM_STATE = 42
BLOCKS = ["MRI", "FDG", "AMYLOID", "CSF"]
N_FOLDS = 5


def policy_factories(quick: bool = False, extra_pool=None) -> dict:
    """Each entry builds a FRESH policy, so every CV fold refits from scratch."""
    episodes = 800 if quick else 4000
    factories = {
        "cheap_only": lambda: CheapOnly(),
        "acquire_all": lambda: AcquireAll(BLOCKS),
        "fixed_MRI": lambda: FixedOrder(BLOCKS, 1),
        "fixed_MRI+FDG": lambda: FixedOrder(BLOCKS, 2),
        "random_rate0.5": lambda: RandomSubset(BLOCKS, 0.5),
        # UMoS as published: MRI first, escalate the uncertain to amyloid PET.
        "umos_mri_to_amyloid": lambda: UMoS(["MRI"], ["AMYLOID"], target_escalation_rate=0.538),
        # A cheaper UMoS variant, to show the mechanism is not tied to one rate.
        "umos_mri_to_amyloid_at0.25": lambda: UMoS(["MRI"], ["AMYLOID"], target_escalation_rate=0.25),
        "promuse_mri_to_amyloid": lambda: ProMUSE(["MRI", "AMYLOID"], target_escalation_rate=0.5),
        "promuse_full_ladder": lambda: ProMUSE(BLOCKS, target_escalation_rate=0.5),
        # Same escalation rule, but predictions from the shared LR bank, so
        # ProMUSE's DECISION rule is compared without also penalising its
        # neural predictor (weaker than LR at this sample size).
        "promuse_full_ladder_sharedpred": lambda: ProMUSE(
            BLOCKS, target_escalation_rate=0.5, predictor="shared"
        ),
        "promuse_mri_to_amyloid_sharedpred": lambda: ProMUSE(
            ["MRI", "AMYLOID"], target_escalation_rate=0.5, predictor="shared"
        ),
    }
    # Our configuration: trivial fixed policy, better inputs (see
    # src/baselines/ours.py). Only available when the runner supplies the
    # pool of partially-observed subjects to train on.
    if extra_pool is not None:
        factories["ours_MRI"] = lambda: OursFixed(["MRI"], extra_pool=extra_pool)
        factories["ours_MRI+FDG"] = lambda: OursFixed(["MRI", "FDG"], extra_pool=extra_pool)

    # Sweeping the DQN's cost weight traces the frontier from one policy.
    for lam in ([0.05] if quick else [0.01, 0.05, 0.2]):
        factories[f"dqn_lambda{lam:g}"] = (
            lambda lam=lam: DQNPolicy(BLOCKS, cost_weight=lam, episodes=episodes)
        )
    return factories


def evaluate_policy(name, factory, dev: pd.DataFrame, label_col: str) -> dict:
    """Cross-validate one policy on the development set, pooling OOF results."""
    y_all = dev[label_col].astype(int).to_numpy()
    probs = np.full(len(dev), np.nan)
    acquired: list[list[str]] = [[] for _ in range(len(dev))]

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    t0 = time.time()
    for tr_idx, te_idx in cv.split(np.zeros(len(dev)), y_all):
        train_fold = dev.iloc[tr_idx]
        test_fold = dev.iloc[te_idx]
        policy = factory().fit(train_fold, label_col)
        res = policy.run(test_fold, label_col)
        probs[te_idx] = res.y_prob
        for pos, blocks in zip(te_idx, res.acquired):
            acquired[pos] = blocks
    elapsed = time.time() - t0

    lo, hi = bootstrap_ci(y_all, probs, auc, seed=RANDOM_STATE)
    row = {
        "policy": name,
        "n": len(dev),
        "n_pos": int(y_all.sum()),
        "auc": round(auc(y_all, probs), 4),
        "auc_lo": round(lo, 4),
        "auc_hi": round(hi, 4),
        "brier": round(brier(y_all, probs), 4),
        "fit_seconds": round(elapsed, 1),
    }
    for regime, costs in COST_REGIMES.items():
        row[f"cost_{regime}"] = round(
            float(np.mean([sum(costs[b] for b in blk) for blk in acquired])), 3
        )
    for b in BLOCKS:
        row[f"acq_{b}"] = round(float(np.mean([b in blk for blk in acquired])), 3)
    row["mean_n_blocks"] = round(float(np.mean([len(blk) for blk in acquired])), 3)
    return row, probs, acquired


def paired_comparisons(
    oof: dict, y: np.ndarray, references: list[str], results: pd.DataFrame
) -> pd.DataFrame:
    """Paired bootstrap of every policy against chosen reference policies.

    Comparing two policies by whether their independent CIs overlap is a weak
    test -- overlapping intervals routinely hide a real paired difference and
    vice versa. Since every policy is scored on the SAME subjects here, the
    paired delta is the right comparison and is far more sensitive.

    Two references matter for the project's central question:
      acquire_all    -- can a selective policy match scanning everyone?
      random_rate0.5 -- does the policy beat acquiring at random? If not,
                        it has not demonstrated that its *decisions* carry
                        information, only that more data helps.
    """
    cost_by_policy = dict(zip(results["policy"], results["cost_price_like"]))
    rows = []
    for ref in references:
        if ref not in oof:
            continue
        for name, probs in oof.items():
            if name == ref:
                continue
            d, lo, hi = bootstrap_delta_ci(y, oof[ref], probs, auc, seed=RANDOM_STATE)
            rows.append(
                dict(
                    policy=name,
                    reference=ref,
                    delta_auc=round(d, 4),
                    lo=round(lo, 4),
                    hi=round(hi, 4),
                    significant=bool(lo > 0 or hi < 0),
                    cost_policy=cost_by_policy.get(name),
                    cost_reference=cost_by_policy.get(ref),
                    cheaper_than_reference=bool(
                        cost_by_policy.get(name, np.inf) < cost_by_policy.get(ref, np.inf)
                    ),
                )
            )
    return pd.DataFrame(rows)


def cost_matched_table(results: pd.DataFrame) -> pd.DataFrame:
    """For each policy, the cheapest NON-LEARNED policy that is at least as good.

    This is the comparison that decides whether personalised acquisition is
    worth anything: a learned policy must beat a dumb fixed or random rule at
    equal-or-lower cost. If a fixed rule matches it for less, the
    personalisation is not buying anything.
    """
    dumb = results[results["policy"].str.startswith(("fixed_", "random_", "cheap_only", "acquire_all"))]
    rows = []
    for _, r in results.iterrows():
        if r["policy"] in set(dumb["policy"]):
            continue
        beaten_by = dumb[
            (dumb["auc"] >= r["auc"]) & (dumb["cost_price_like"] <= r["cost_price_like"])
        ].sort_values("cost_price_like")
        rows.append(
            dict(
                policy=r["policy"],
                auc=r["auc"],
                cost=r["cost_price_like"],
                dominated_by=beaten_by.iloc[0]["policy"] if len(beaten_by) else "",
                dominator_auc=beaten_by.iloc[0]["auc"] if len(beaten_by) else np.nan,
                dominator_cost=beaten_by.iloc[0]["cost_price_like"] if len(beaten_by) else np.nan,
            )
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--quick", action="store_true", help="fewer RL episodes, one lambda")
    ap.add_argument("--only", type=str, default=None, help="comma-separated policy names")
    args = ap.parse_args()

    label_col = f"LABEL_{args.horizon}mo"
    print("Loading real ADNI cohort...")
    cohort = build_cohort(build_subject_visits())
    labelled = cohort[cohort[label_col].notna()].copy()

    # Complete case over the blocks in play, so every policy could acquire anything.
    complete = subjects_with_blocks(labelled, BLOCKS)
    print(f"  {len(labelled)} labelled at {args.horizon}mo; "
          f"{len(complete)} hold all of {BLOCKS} (the comparison cohort)")

    split = subject_level_split(complete, label_col=label_col, random_state=RANDOM_STATE)
    assert_no_subject_leakage(split)
    dev = complete[complete["RID"].isin(split.train_rids | split.val_rids)].copy()
    print(f"  development set: {len(dev)}, converters: {int(dev[label_col].sum())} "
          f"(test of {len(split.test_rids)} locked away, untouched)")
    print()

    # Subjects who are labelled and outside the locked test set but lack some
    # modality -- exactly the ones complete-case analysis discards, and what
    # our pooled configuration trains on.
    extra_pool = labelled[
        (~labelled["RID"].isin(split.test_rids)) & (~labelled["RID"].isin(dev["RID"]))
    ].copy()

    factories = policy_factories(quick=args.quick, extra_pool=extra_pool)
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        factories = {k: v for k, v in factories.items() if k in wanted}

    rows = []
    oof: dict = {}
    for name, factory in factories.items():
        print(f"running {name} ...", flush=True)
        row, probs, _ = evaluate_policy(name, factory, dev, label_col)
        rows.append(row)
        oof[name] = probs
        print(f"   AUC={row['auc']:.4f} [{row['auc_lo']:.4f},{row['auc_hi']:.4f}]  "
              f"cost(price_like)={row['cost_price_like']:.2f}  "
              f"blocks={row['mean_n_blocks']:.2f}  ({row['fit_seconds']}s)")

    results = pd.DataFrame(rows).sort_values("cost_price_like")
    print()
    print(f"=== POLICY BENCHMARK, {args.horizon}-month horizon "
          f"(CV on development set; test untouched) ===")
    show = ["policy", "n", "n_pos", "auc", "auc_lo", "auc_hi", "brier",
            "cost_price_like", "cost_flat", "mean_n_blocks"] + [f"acq_{b}" for b in BLOCKS]
    print(results[show].to_string(index=False))

    y_dev = dev[label_col].astype(int).to_numpy()
    paired = paired_comparisons(oof, y_dev, ["acquire_all", "random_rate0.5"], results)
    if not paired.empty:
        print()
        print("=== PAIRED COMPARISONS (same subjects; positive = policy beats reference) ===")
        print(paired.to_string(index=False))
        paired.to_csv(f"paper/tables/policy_paired_{args.horizon}mo.csv", index=False)

    dom = cost_matched_table(results)
    if not dom.empty:
        print()
        print("=== COST DOMINATION: is a learned policy beaten by a dumb rule that costs no more? ===")
        print(dom.to_string(index=False))
        dom.to_csv(f"paper/tables/policy_domination_{args.horizon}mo.csv", index=False)

    out = f"paper/tables/policy_benchmark_{args.horizon}mo.csv"
    results.to_csv(out, index=False)
    print()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
