#!/usr/bin/env python
"""
Table 5 replication restricted to S&P 500 constituents during the OOS window.

Requires data/sp500_membership_oos_panel.csv, produced by running
sp500_membership_extraction.ipynb (a WRDS pull of crsp.msp500list, point-in-time
correct: a firm only counts as S&P 500 in the months it was actually a member).

Reuses table5_replication.py's existing benchmark-panel builders, realized-return
loader, and run_depth() regression/decile-sort logic unchanged -- the only
difference is an inner join against the S&P 500 membership panel before running
each depth, restricting both the regression sample and the decile sort to
S&P 500 firm-months only.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

import table5_replication as t5

SP500_PANEL_PATH = "../data/sp500_membership_oos_panel.csv"


def load_sp500_panel() -> pd.DataFrame:
    panel = pd.read_csv(SP500_PANEL_PATH, dtype={"gvkey": str})
    panel = panel.drop_duplicates(subset=["gvkey", "yyyymm"])
    return panel


def run_depth_sp500(depth, benchmark_panel, realized, bench_cols, sp500_panel, min_firms_for_sort):
    """Same logic as table5_replication.run_depth, restricted to S&P 500 firm-months."""
    gnn = t5.load_gnn_predictions(depth)
    gnn["gvkey"] = gnn["gvkey"].astype(str)

    reg_df = gnn.merge(benchmark_panel, on=["gvkey", "yyyymm"], how="inner")
    reg_df = reg_df[(reg_df["yyyymm"] >= t5.OOS_START) & (reg_df["yyyymm"] <= t5.OOS_END)]
    reg_df = reg_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["gnn_pred"] + bench_cols)

    # Restrict to S&P 500 membership for that gvkey-month
    reg_df = reg_df.merge(sp500_panel, on=["gvkey", "yyyymm"], how="inner")

    n_obs = len(reg_df)
    n_months = reg_df["yyyymm"].nunique()
    n_firms = reg_df["gvkey"].nunique()
    avg_firms_per_month = n_obs / n_months if n_months else np.nan

    if n_obs == 0:
        raise ValueError(f"depth={depth}: no overlapping S&P 500 firm-month observations in OOS window")

    import statsmodels.api as sm

    X = sm.add_constant(reg_df[bench_cols])
    y = reg_df["gnn_pred"]

    ols = sm.OLS(y, X).fit()
    hac = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": t5.NW_LAGS})

    a_hat = hac.params["const"]
    a_t = hac.tvalues["const"]
    r2 = hac.rsquared

    beta = ols.params[bench_cols]
    reg_df = reg_df.copy()
    reg_df["gnn_perp"] = reg_df["gnn_pred"] - (reg_df[bench_cols].to_numpy() @ beta.to_numpy())

    port = reg_df.merge(realized, on=["gvkey", "yyyymm"], how="inner")
    port = port.dropna(subset=["mthret", "rf"])

    try:
        F = t5.decile_long_short(port, "gnn_perp", min_firms=min_firms_for_sort)
        F = F.dropna()
        mean_ret = F.mean()
        vol_ret = F.std()
        sharpe = mean_ret / vol_ret if vol_ret > 0 else np.nan
        f_n_months = len(F)
    except Exception as e:
        print(f"  depth={depth}: decile sort failed ({e})")
        f_n_months, mean_ret, vol_ret, sharpe = 0, np.nan, np.nan, np.nan

    return {
        "depth": depth,
        "n_obs": n_obs,
        "n_months": n_months,
        "n_firms": n_firms,
        "avg_firms_per_month": avg_firms_per_month,
        "a_hat": a_hat,
        "a_tstat_NW": a_t,
        "nw_lags": t5.NW_LAGS,
        "R2": r2,
        "F_n_months": f_n_months,
        "mean": mean_ret,
        "vol": vol_ret,
        "sharpe": sharpe,
    }


if __name__ == "__main__":
    # S&P 500 membership is a much smaller universe (~500 firms/month vs ~440-1660
    # avg firms/month in the full-universe results) -- lower min_firms_for_sort so
    # decile buckets (needs >= min_firms firms to form deciles at all) don't get
    # starved out in thin months.
    MIN_FIRMS = 5

    sp500_panel = load_sp500_panel()
    print(f"S&P 500 panel: {len(sp500_panel)} gvkey-months, "
          f"{sp500_panel['gvkey'].nunique()} distinct gvkeys, "
          f"{sp500_panel['yyyymm'].nunique()} months")

    realized = t5.load_realized_returns()

    for variant_name, bench_cols, panel_builder, out_path in [
        ("Variant A (full 6-benchmark)", t5.BENCH_COLS_FULL, t5.build_benchmark_panel_full,
         "../data/table5_sp500_variantA_full6bench.csv"),
        ("Variant B (PCA + FF5 only)", t5.BENCH_COLS_BROAD, t5.build_benchmark_panel_broad,
         "../data/table5_sp500_variantB_pca_ff5.csv"),
    ]:
        print("=" * 100)
        print(variant_name, "-- S&P 500 only")
        print("=" * 100)
        benchmark_panel = panel_builder()
        benchmark_panel["gvkey"] = benchmark_panel["gvkey"].astype(str)

        rows = []
        for d in t5.DEPTHS:
            try:
                r = run_depth_sp500(d, benchmark_panel, realized, bench_cols, sp500_panel, MIN_FIRMS)
                rows.append(r)
                print(f"depth={d}: n_firms={r['n_firms']}, avg_firms/mo={r['avg_firms_per_month']:.1f}, "
                      f"sharpe={r['sharpe']:.4f}")
            except Exception as e:
                print(f"depth={d} FAILED: {e}")

        out_df = pd.DataFrame(rows)
        out_df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}\n")
