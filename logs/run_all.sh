#!/bin/bash
# Re-run the full pipeline on the Oct-2026 ADNI release. Logs per step.
cd /d/Journal/adni_project
export PYTHONWARNINGS=ignore PYTHONUTF8=1
step() { name=$1; shift; echo "=== $name  $(date +%T)"; python -W ignore -m "$@" > logs/$name.log 2>&1; echo "    exit $?"; }
step cohort        src.preprocessing.build_real_cohort
for h in 12 24 36; do step bench_$h src.eval.run_policy_benchmark --horizon $h; done
for h in 12 24 36; do step ablation_$h src.eval.run_ablation --horizon $h; done
step robustness    src.eval.run_robustness --horizon 24 --check all
step clinical      src.eval.run_clinical_utility --horizon 24
step fairness      src.eval.run_fairness_audit --horizon 24
step horizon_eff   src.eval.run_horizon_effect
step horizon_pol   src.eval.run_horizon_policy
step modality      src.eval.run_modality_value
step dominance     src.eval.run_dominance
step improved      src.eval.run_improved --horizon 24
step domination_rg src.eval.domination_by_regime
step sweep         src.eval.run_random_sweep
step train_final   src.models.train_final --all-horizons
step nested        src.models.nested_cv --all-horizons
step summarize     src.eval.summarize_results
echo ALL_DONE
