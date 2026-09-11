#!/usr/bin/env python
"""
Computes out-of-sample R2 (paper's Table 1 definition) and long-short portfolio
ER/Sharpe (paper's Table 4 definition, Eq. 23) for the GNN at each depth and for
NC Ridge / NC LASSO, over the OOS window Dtest = Jan 2017 - Jan 2024.

R2 definition (paper_text.txt lines 928-937): R2 = 1 - sum((Y - Yhat)^2) / sum(Y^2)
  -- benchmarked against a forecast of zero, not the historical mean.

ER/Sharpe definition (paper_text.txt, Eq. 23 / Table 4): Ft = R10,t - R1,t, the
equal-weighted top-minus-bottom decile spread of a monthly sort on model-predicted
excess returns. ER = mean(Ft), SR = mean(Ft)/std(Ft), both monthly (non-annualized,
matching the paper's Table 4 reporting convention already confirmed in
table5_replication.py's "monthly, non-annualized" comparison block).

Paper's actual numbers (Table 1 R2, Table 4 ER/SR) are hardcoded below as
PAPER_TABLE1 / PAPER_TABLE4 for direct side-by-side comparison.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

import table5_replication as t5

DEPTHS = t5.DEPTHS  # [1, 2, 3, 4, 6, 10]

# Paper Table 1: out-of-sample R2 by depth (paper_text.txt lines 948-953, 1006-1009)
PAPER_TABLE1_GNN = {1: 0.1493, 2: 0.1512, 3: 0.1527, 4: 0.1525, 10: -0.0030}
PAPER_NC_RIDGE_R2 = 0.0663
PAPER_NC_LASSO_R2 = 0.0660
PAPER_NN3_R2 = 0.1273

# Paper Table 4: out-of-sample ER / Sharpe by model (paper_text.txt, Table 4 block)
PAPER_TABLE4 = {
    "GNN (d=1)": (0.0648, 0.8562),
    "GNN (d=2)": (0.0682, 0.8988),
    "GNN (d=3)": (0.1106, 1.3923),
    "GNN (d=4)": (0.1078, 1.3914),
    "GNN (d=10)": (-0.0003, -0.0095),
    "NC Ridge": (0.1156, 1.1513),
    "NC LASSO": (0.1148, 1.1946),
}


def oos_r2(pred_col: str, pred_df: pd.DataFrame, realized: pd.DataFrame) -> tuple[float, int]:
    """R2 = 1 - sum((y-yhat)^2)/sum(y^2), paper's zero-benchmark definition."""
    m = pred_df.merge(realized, on=["gvkey", "yyyymm"], how="inner")
    m = m[(m["yyyymm"] >= t5.OOS_START) & (m["yyyymm"] <= t5.OOS_END)]
    m = m.dropna(subset=[pred_col, "mthret"])
    num = ((m["mthret"] - m[pred_col]) ** 2).sum()
    denom = (m["mthret"] ** 2).sum()
    return 1 - num / denom, len(m)


def main():
    realized = t5.load_realized_returns()
    realized["gvkey"] = realized["gvkey"].astype(str)

    print("=" * 100)
    print("OUT-OF-SAMPLE R2 (paper's Table 1 definition: 1 - sum((Y-Yhat)^2)/sum(Y^2))")
    print("=" * 100)
    rows = []
    for d in DEPTHS:
        gnn = t5.load_gnn_predictions(d)
        gnn["gvkey"] = gnn["gvkey"].astype(str)
        r2, n = oos_r2("gnn_pred", gnn, realized)
        paper_r2 = PAPER_TABLE1_GNN.get(d, np.nan)
        rows.append({"depth": d, "our_R2": r2, "paper_R2": paper_r2, "diff": r2 - paper_r2, "n_obs": n})
    r2_df = pd.DataFrame(rows).set_index("depth")
    print(r2_df.to_string(float_format=lambda x: f"{x:.4f}"))

    # NC Ridge / NC LASSO R2
    print()
    for name, path, paper_r2 in [
        ("NC Ridge", "../data/firm_level_NC_ridge_predictions_ALL_firms.csv", PAPER_NC_RIDGE_R2),
        ("NC LASSO", "../data/firm_level_NC_lasso_predictions_ALL_firms.csv", PAPER_NC_LASSO_R2),
    ]:
        wide = pd.read_csv(path)
        long = wide.melt(id_vars=["yyyymm"], var_name="gvkey", value_name="pred").dropna(subset=["pred"])
        long["gvkey"] = long["gvkey"].astype(str)
        r2, n = oos_r2("pred", long, realized)
        print(f"{name}: our_R2={r2:.4f}, paper_R2={paper_r2:.4f}, diff={r2-paper_r2:+.4f}, n_obs={n}")

    print()
    print("=" * 100)
    print("LONG-SHORT PORTFOLIO ER / SHARPE (paper's Table 4 definition, Eq. 23: F_t = R10,t - R1,t)")
    print("Monthly, non-annualized, equal-weighted decile spread on model-predicted returns")
    print("=" * 100)
    rows = []
    for d in DEPTHS:
        gnn = t5.load_gnn_predictions(d)
        gnn["gvkey"] = gnn["gvkey"].astype(str)
        merged = gnn.merge(realized, on=["gvkey", "yyyymm"], how="inner")
        merged = merged[(merged["yyyymm"] >= t5.OOS_START) & (merged["yyyymm"] <= t5.OOS_END)]
        ft = t5.decile_long_short(merged, sorting_var="gnn_pred", min_firms=10)
        er, sr = ft.mean(), ft.mean() / ft.std()
        paper_er, paper_sr = PAPER_TABLE4.get(f"GNN (d={d})", (np.nan, np.nan))
        rows.append({"depth": d, "our_ER": er, "paper_ER": paper_er,
                     "our_SR": sr, "paper_SR": paper_sr, "SR_diff": sr - paper_sr})
    table4_df = pd.DataFrame(rows).set_index("depth")
    print(table4_df.to_string(float_format=lambda x: f"{x:.4f}"))

    print()
    for name, path, paper_key in [
        ("NC Ridge", "../data/firm_level_NC_ridge_predictions_ALL_firms.csv", "NC Ridge"),
        ("NC LASSO", "../data/firm_level_NC_lasso_predictions_ALL_firms.csv", "NC LASSO"),
    ]:
        wide = pd.read_csv(path)
        long = wide.melt(id_vars=["yyyymm"], var_name="gvkey", value_name="pred").dropna(subset=["pred"])
        long["gvkey"] = long["gvkey"].astype(str)
        merged = long.merge(realized, on=["gvkey", "yyyymm"], how="inner")
        merged = merged[(merged["yyyymm"] >= t5.OOS_START) & (merged["yyyymm"] <= t5.OOS_END)]
        ft = t5.decile_long_short(merged, sorting_var="pred", min_firms=10)
        er, sr = ft.mean(), ft.mean() / ft.std()
        paper_er, paper_sr = PAPER_TABLE4[paper_key]
        print(f"{name}: our_ER={er:.4f} (paper {paper_er:.4f}), our_SR={sr:.4f} (paper {paper_sr:.4f}), "
              f"SR_diff={sr-paper_sr:+.4f}")

    r2_df.to_csv("../data/paper_metrics_r2_comparison.csv")
    table4_df.to_csv("../data/paper_metrics_table4_comparison.csv")
    print("\nSaved: ../data/paper_metrics_r2_comparison.csv, ../data/paper_metrics_table4_comparison.csv")


if __name__ == "__main__":
    main()
