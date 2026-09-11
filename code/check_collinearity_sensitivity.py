#!/usr/bin/env python
"""
Robustness check: NC Ridge and NC LASSO predictions are ~0.93 correlated
(measured directly on data/firm_level_NC_ridge_predictions_ALL_firms.csv vs
data/firm_level_NC_lasso_predictions_ALL_firms.csv earlier this session).
That doesn't bias the joint-OLS spanning regression's residual/Sharpe ratio
(OLS point estimates stay unbiased under collinearity, only individual
coefficient variances inflate) -- but it's still worth confirming directly:
does the Variant A (6-benchmark) Sharpe ratio move much if we drop one of
the two near-redundant regressors?

Three variants compared, same GNN predictions / OOS window / decile-sort
logic as table5_replication.py's run_depth(), just different benchmark
column sets:
  - full6:  nn2, nn3, pca, ff5, ncridge, nclasso  (current Variant A)
  - drop_lasso: nn2, nn3, pca, ff5, ncridge          (NC LASSO removed)
  - drop_ridge: nn2, nn3, pca, ff5, nclasso          (NC Ridge removed)

If the Sharpe ratios are similar across all three, the negative Variant A
result is stable and not an artifact of NC Ridge/LASSO collinearity. If they
diverge a lot, that's a real caveat worth flagging.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

import table5_replication as t5

DEPTHS = t5.DEPTHS
MIN_FIRMS = 10

full = t5.build_benchmark_panel_full()
realized = t5.load_realized_returns()

variants = {
    "full6": ["nn2_pred", "nn3_pred", "pca_pred", "ff5_pred", "ncridge_pred", "nclasso_pred"],
    "drop_lasso": ["nn2_pred", "nn3_pred", "pca_pred", "ff5_pred", "ncridge_pred"],
    "drop_ridge": ["nn2_pred", "nn3_pred", "pca_pred", "ff5_pred", "nclasso_pred"],
}

results = {}
for name, cols in variants.items():
    rows = []
    for d in DEPTHS:
        try:
            r = t5.run_depth(d, full, realized, cols, min_firms_for_sort=MIN_FIRMS)
            rows.append({"depth": d, "sharpe": r["sharpe"], "a_hat": r["a_hat"], "R2": r["R2"]})
        except Exception as e:
            print(f"{name} depth={d} FAILED: {e}")
            rows.append({"depth": d, "sharpe": np.nan, "a_hat": np.nan, "R2": np.nan})
    results[name] = pd.DataFrame(rows).set_index("depth")

print("=" * 100)
print("Sharpe ratio comparison: full 6-benchmark vs. dropping one of NC Ridge/NC LASSO")
print("=" * 100)
comparison = pd.DataFrame({name: df["sharpe"] for name, df in results.items()})
comparison["max_abs_diff_from_full6"] = (
    comparison[["drop_lasso", "drop_ridge"]].sub(comparison["full6"], axis=0).abs().max(axis=1)
)
print(comparison)

print()
print("Interpretation:")
print("  max_abs_diff_from_full6 small (<0.2) at every depth -> negative Variant A result is")
print("  stable, not a collinearity artifact of including both NC Ridge and NC LASSO together.")
print("  max_abs_diff_from_full6 large at some depth -> that depth's result is sensitive to")
print("  which near-redundant benchmark is included, worth flagging as a caveat.")

comparison.to_csv("../data/table5_collinearity_sensitivity_check.csv")
print("\nSaved: ../data/table5_collinearity_sensitivity_check.csv")
