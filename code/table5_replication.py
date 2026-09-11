"""
Table 5 replication: joint spanning regression of GNN predicted returns on
non-graph benchmarks, and the Sharpe ratio of the orthogonalized long-short
portfolio.

This is a standalone script (not touching GNN model.ipynb or
GNN_code_additional.ipynb). It reads only already-produced artifacts:

  - data/temp/rppca_{train,test}_corrections_Transformer_All_firms_layers_{d}_dim_40.csv
      firm-month GNN predicted returns (Y_hat_GNN(d)_it), for d in {1,2,3,4,6,10}.
      This is the raw model output ("predicted_correction" = model(x, edge_index)),
      i.e. the actual firm-level predicted excess return, NOT a decile-differenced
      portfolio return and NOT a PCA-transformed factor score. See notes below.

  - data/firm_level_reg_pca_predictions_ALL_firms.csv   -> PCA (k=5) benchmark
  - data/firm_level_reg_ff5_predictions_ALL_firms.csv   -> FF5 benchmark
  - data/firm_level_NC_ridge_predictions_ALL_firms.csv  -> NC Ridge benchmark
  - data/firm_level_NC_lasso_predictions_ALL_firms.csv  -> NC LASSO benchmark

  - data/features.csv  -> realized mthret, rf (for decile-sort portfolio returns)

NN2 / NN3 benchmarks are NOT available in this pipeline (see notes below) and
are omitted from the spanning regression menu; this is documented explicitly
in the printed output rather than silently substituted.

Paper reference (paper_text.txt):
  - Eq. (19), lines 1041-1044:  Y_hat_GNN(d)_it = a_d + beta_d' z_it + u_it
  - Residual definition, lines 1050-1054: Y_hat_GNN(d),perp_it = Y_hat_GNN(d)_it - beta_d' z_it
    (intercept a_d is NOT subtracted -- confirmed against GNN_code_additional.ipynb's
    reference cell 45, which computes the residual "NO_intercept": gnn_pred - b_hat*mlp_pred,
    consistent with the paper equation directly under Table 2, and Table 5's caption which
    repeats the identical formula.)
  - Newey-West lag 12, OOS window Jan 2017 - Jan 2024: lines 1026-1028.
  - Eq. (23) decile long-short construction, lines 1213-1240 (reused from GNN model.ipynb's
    decile_portfolio function, same logic, equal-weighted decile spread R10 - R1).
  - Table 5 target values (lines 1262-1293, 1309-1310):
        d=1:  mean 0.0525, vol 0.0037, SR 0.8660
        d=2:  mean 0.0546, vol 0.0034, SR 0.9354
        d=3:  mean 0.0633, vol 0.0035, SR 1.0682   <- paper's headline 1.07
        d=4:  mean 0.0568, vol 0.0039, SR 0.9115
        d=6:  mean 0.0431, vol 0.0045, SR 0.6422
        d=10: mean 0.0024, vol 0.0010, SR 0.0766
    No "annualiz" language appears anywhere near Table 4 or Table 5 in the paper text,
    so these are reported as monthly (non-annualized) Sharpe ratios: mean/std of the
    monthly long-short return series. We follow the same convention here.
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(NOTEBOOK_DIR)

DATA = "../data"
DEPTHS = [1, 2, 3, 4, 6, 10]
EMBED_DIM = 40  # matches the config used to produce pca_asset_pricing_factors_*_dim_40 files
OOS_START, OOS_END = 201701, 202401  # inclusive, per paper lines 1027-1028 / 1233
NW_LAGS = 12  # paper line 1026: "Newey-West t-statistic (with lag length 12)"


# ---------------------------------------------------------------------------
# 1. Load GNN firm-month predicted returns per depth (train+test corrections)
# ---------------------------------------------------------------------------
def load_gnn_predictions(depth: int) -> pd.DataFrame:
    train_path = f"{DATA}/temp/rppca_train_corrections_Transformer_ALL_firms_layers_{depth}_dim_{EMBED_DIM}.csv"
    test_path = f"{DATA}/temp/rppca_test_corrections_Transformer_All_firms_layers_{depth}_dim_{EMBED_DIM}.csv"
    train = pd.read_csv(train_path, index_col=0)
    test = pd.read_csv(test_path, index_col=0)
    both = pd.concat([train, test], ignore_index=True)
    both = both.rename(columns={"predicted_correction": "gnn_pred"})
    both["gvkey"] = both["gvkey"].astype(str)
    both["yyyymm"] = both["yyyymm"].astype(int)
    return both[["gvkey", "yyyymm", "gnn_pred"]]


# ---------------------------------------------------------------------------
# 2. Load the four available non-graph benchmark predicted-return panels
#    (wide: rows=yyyymm, cols=gvkey) and melt to long firm-month format.
#    NN2 / NN3 are NOT included: this pipeline never trains/saves NN2 or NN3
#    firm-level predictions. GNN_code_additional.ipynb cells 3-7 define an MLP
#    ("NN Benchmark: Multi-Layer Perceptron") with 2-hidden-layer and
#    3-hidden-layer configurations that would be the natural NN2/NN3 analogues,
#    but that notebook (a) depends on `df`/`feature_cols` state that is never
#    defined within it or within GNN model.ipynb (it's a leftover from the
#    original FactSet-based version of the pipeline per that notebook's own
#    header comment), and (b) never persists train_pred/test_pred to a CSV --
#    the outputs are held only as in-memory variables and overwritten by the
#    next cell. No real, already-computed NN2/NN3 firm-level prediction file
#    exists anywhere in this repo. Rather than run an untested 80-epoch MLP
#    training with fabricated/guessed inputs, we omit NN2/NN3 here and
#    document the gap explicitly (see report).
# ---------------------------------------------------------------------------
def load_benchmark_long(path: str, value_name: str) -> pd.DataFrame:
    wide = pd.read_csv(path)
    long = wide.melt(id_vars="yyyymm", var_name="gvkey", value_name=value_name)
    long["gvkey"] = long["gvkey"].astype(str).str.replace(r"\.0$", "", regex=True)
    long["yyyymm"] = long["yyyymm"].astype(int)
    return long


def build_benchmark_panel_full() -> pd.DataFrame:
    """NN2 + NN3 + FF5 + PCA + NC Ridge + NC LASSO, inner-joined (Variant A) -- the
    paper's full Table 5 benchmark menu (six non-graph benchmarks). This restricts the
    universe to the ~37 firms NC Ridge/LASSO ever cover, and within the OOS window
    those files are non-missing for only ~2 firms/month on average.

    NN2/NN3 (2- and 3-hidden-layer feed-forward MLPs over the 64 firm characteristics,
    target = mthret - rf, train/test split at 201612, 40 neurons/layer, 80 epochs) are
    produced by code/build_nn_benchmarks.py -> data/firm_level_NN{2,3}_predictions_ALL_firms.csv,
    wide format identical to the PCA/FF5 files."""
    nn2 = load_benchmark_long(f"{DATA}/firm_level_NN2_predictions_ALL_firms.csv", "nn2_pred")
    nn3 = load_benchmark_long(f"{DATA}/firm_level_NN3_predictions_ALL_firms.csv", "nn3_pred")
    pca = load_benchmark_long(f"{DATA}/firm_level_reg_pca_predictions_ALL_firms.csv", "pca_pred")
    ff5 = load_benchmark_long(f"{DATA}/firm_level_reg_ff5_predictions_ALL_firms.csv", "ff5_pred")
    ridge = load_benchmark_long(f"{DATA}/firm_level_NC_ridge_predictions_ALL_firms.csv", "ncridge_pred")
    lasso = load_benchmark_long(f"{DATA}/firm_level_NC_lasso_predictions_ALL_firms.csv", "nclasso_pred")

    z = nn2.merge(nn3, on=["gvkey", "yyyymm"], how="inner")
    z = z.merge(pca, on=["gvkey", "yyyymm"], how="inner")
    z = z.merge(ff5, on=["gvkey", "yyyymm"], how="inner")
    z = z.merge(ridge, on=["gvkey", "yyyymm"], how="inner")
    z = z.merge(lasso, on=["gvkey", "yyyymm"], how="inner")
    return z


def build_benchmark_panel_broad() -> pd.DataFrame:
    """PCA + FF5 only (Variant B), covering the full ~3500-4400 firm universe per month
    -- broad enough to support a real decile sort."""
    pca = load_benchmark_long(f"{DATA}/firm_level_reg_pca_predictions_ALL_firms.csv", "pca_pred")
    ff5 = load_benchmark_long(f"{DATA}/firm_level_reg_ff5_predictions_ALL_firms.csv", "ff5_pred")
    return pca.merge(ff5, on=["gvkey", "yyyymm"], how="inner")


BENCH_COLS_FULL = ["nn2_pred", "nn3_pred", "pca_pred", "ff5_pred", "ncridge_pred", "nclasso_pred"]
BENCH_COLS_BROAD = ["pca_pred", "ff5_pred"]


# ---------------------------------------------------------------------------
# 3. Realized returns for decile-sort portfolio construction
# ---------------------------------------------------------------------------
def load_realized_returns() -> pd.DataFrame:
    ret = pd.read_csv(f"{DATA}/features.csv", usecols=["gvkey", "yyyymm", "mthret", "rf"])
    ret["gvkey"] = ret["gvkey"].astype(str)
    ret["yyyymm"] = ret["yyyymm"].astype(int)
    return ret


# ---------------------------------------------------------------------------
# 4. Decile portfolio / long-short spread (same logic as GNN model.ipynb's
#    decile_portfolio, reused directly)
# ---------------------------------------------------------------------------
def decile_long_short(df: pd.DataFrame, sorting_var: str, min_firms: int = 10) -> pd.Series:
    """
    Same logic as GNN model.ipynb's decile_portfolio (equal-weighted decile spread,
    R10 - R1, on ranked sorting_var, risk-free rate subtracted). The spanning-regression
    residual is only defined on the NC Ridge/LASSO-covered universe (37 firms total,
    ~11-22 per month after intersecting with GNN train/test coverage in the OOS window),
    so exact decile (10-quantile) breakpoints are not always achievable with unique bin
    edges in low-count months. We use pd.qcut(..., duplicates="drop") to get the best
    available approximately-equal-sized grouping, then take the extreme top vs bottom
    group as the long-short spread -- this preserves "top decile minus bottom decile"
    in spirit when the firm count does not support exactly ten unique breakpoints.
    Months with fewer than `min_firms` observations are dropped entirely (too few firms
    to form a meaningful long-short spread).
    """
    d = df.copy()
    counts = d.groupby("yyyymm")[sorting_var].transform("count")
    d = d[counts >= min_firms]

    d["ranked_sorting_var"] = d.groupby("yyyymm")[sorting_var].rank(method="first")

    def bucket(x):
        nbins = min(10, x.nunique())
        return pd.qcut(x, nbins, labels=False, duplicates="drop")

    d["decile"] = d.groupby("yyyymm")["ranked_sorting_var"].transform(bucket)

    port = d.groupby(["yyyymm", "decile"])["mthret"].mean().reset_index(name="ew_ret")
    mat = port.pivot(index="yyyymm", columns="decile", values="ew_ret")
    rf_rate = d.groupby("yyyymm")["rf"].first()
    mat = mat.subtract(rf_rate, axis=0)

    top_col = mat.columns.max()
    bot_col = mat.columns.min()
    return mat[top_col] - mat[bot_col]  # F_t = R_top,t - R_bottom,t


# ---------------------------------------------------------------------------
# 5. Joint spanning regression (Eq. 19) + residualization + Sharpe ratio
# ---------------------------------------------------------------------------
def run_depth(depth: int, benchmark_panel: pd.DataFrame, realized: pd.DataFrame,
              bench_cols: list, min_firms_for_sort: int) -> dict:
    gnn = load_gnn_predictions(depth)

    reg_df = gnn.merge(benchmark_panel, on=["gvkey", "yyyymm"], how="inner")
    reg_df = reg_df[(reg_df["yyyymm"] >= OOS_START) & (reg_df["yyyymm"] <= OOS_END)]
    reg_df = reg_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["gnn_pred"] + bench_cols)

    n_obs = len(reg_df)
    n_months = reg_df["yyyymm"].nunique()
    n_firms = reg_df["gvkey"].nunique()
    avg_firms_per_month = n_obs / n_months if n_months else np.nan

    if n_obs == 0:
        raise ValueError(f"depth={depth}: no overlapping firm-month observations in OOS window")

    X = sm.add_constant(reg_df[bench_cols])
    y = reg_df["gnn_pred"]

    ols = sm.OLS(y, X).fit()
    hac = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})

    a_hat = hac.params["const"]
    a_t = hac.tvalues["const"]
    r2 = hac.rsquared

    beta = ols.params[bench_cols]  # slope coefficients (point estimates same in OLS/HAC; only SEs differ)
    # Residual WITHOUT the intercept, per Eq. just after (19): Y_perp = Y - beta'z (a_d not subtracted)
    reg_df = reg_df.copy()
    reg_df["gnn_perp"] = reg_df["gnn_pred"] - (reg_df[bench_cols].to_numpy() @ beta.to_numpy())

    port = reg_df.merge(realized, on=["gvkey", "yyyymm"], how="inner")
    port = port.dropna(subset=["mthret", "rf"])

    try:
        F = decile_long_short(port, "gnn_perp", min_firms=min_firms_for_sort)
        F = F.dropna()
        mean_ret = F.mean()
        vol_ret = F.std()
        sharpe = mean_ret / vol_ret if vol_ret > 0 else np.nan
        f_n_months = len(F)
    except Exception:
        f_n_months, mean_ret, vol_ret, sharpe = 0, np.nan, np.nan, np.nan

    return {
        "depth": depth,
        "n_obs": n_obs,
        "n_months": n_months,
        "n_firms": n_firms,
        "avg_firms_per_month": avg_firms_per_month,
        "a_hat": a_hat,
        "a_tstat_NW12": a_t,
        "R2": r2,
        "F_n_months": f_n_months,
        "mean": mean_ret,
        "vol": vol_ret,
        "sharpe": sharpe,
    }


def main():
    benchmark_panel_full = build_benchmark_panel_full()
    benchmark_panel_broad = build_benchmark_panel_broad()
    realized = load_realized_returns()

    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

    paper_table5 = {1: 0.8660, 2: 0.9354, 3: 1.0682, 4: 0.9115, 6: 0.6422, 10: 0.0766}

    # ------------------------------------------------------------------
    # Variant A: full paper-spec benchmark menu minus NN2/NN3
    #   z_it = {PCA, FF5, NC Ridge, NC LASSO}
    # NC Ridge/LASSO's own saved prediction files are EXTREMELY sparse in
    # the OOS window (avg ~2.2 non-missing firms/month out of a possible 37
    # -- a genuine upstream data-coverage limitation in
    # firm_level_NC_{ridge,lasso}_predictions_ALL_firms.csv, not a bug
    # introduced here). This is too few firms for a decile sort (need >=10
    # per month), so this variant reports the spanning-regression intercept
    # and R^2 (which only need firm-month pairs, not deciles) but the
    # portfolio Sharpe ratio is not computable / not meaningful here.
    # ------------------------------------------------------------------
    print("=" * 100)
    print("VARIANT A: joint spanning regression on the FULL paper Table 5 benchmark menu")
    print("z_it = {NN2, NN3, FF5, PCA, NC Ridge, NC LASSO}  (all six benchmarks -- matches the paper)")
    print("=" * 100)
    rows_a = []
    for d in DEPTHS:
        try:
            rows_a.append(run_depth(d, benchmark_panel_full, realized, BENCH_COLS_FULL, min_firms_for_sort=10))
        except Exception as e:
            print(f"depth={d} FAILED: {e}")
    results_a = pd.DataFrame(rows_a).set_index("depth")
    print(results_a[["n_obs", "n_months", "avg_firms_per_month", "a_hat", "a_tstat_NW12", "R2"]])
    print("\nPortfolio sort feasibility (avg firms/month vs 10 needed for deciles):")
    print(results_a[["avg_firms_per_month", "F_n_months", "sharpe"]])
    # This used to be a hardcoded "coverage is too sparse (~2 firms/month), Sharpe ratios
    # are NaN" message, written back when GNN model.ipynb's neighbor-lookup bug (cells 8/9)
    # capped NC Ridge/LASSO's usable universe at ~2 firms/month. That bug is now fixed
    # (avg_firms_per_month reflects the real, current NC Ridge/LASSO coverage), so the
    # message is derived from the actual data instead of restating a stale number.
    if (results_a["avg_firms_per_month"] < 10).any():
        print("--> NC Ridge/LASSO coverage is too sparse for a decile sort in some depths;")
        print("    Sharpe ratios above may be NaN/unreliable for those rows. See Variant B.")
    else:
        print("--> NC Ridge/LASSO coverage now supports a real decile sort (>=10 firms/month);")
        print("    Sharpe ratios above are a genuine full 4-benchmark Table 5 estimate.")

    # ------------------------------------------------------------------
    # Variant B: broad-coverage benchmark subset that actually supports a
    # real decile sort: z_it = {PCA, FF5} (full ~3500-4400 firm universe
    # per month). This is a further-reduced menu vs the paper's 6 (or even
    # our own NN2/NN3-omitted 4), but it is the only variant in this
    # pipeline's real, already-computed outputs that supports Eq. (23)'s
    # decile methodology, so it is the one we report as our best estimate
    # of the paper's Table 5 Sharpe ratios.
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("VARIANT B: joint spanning regression on BROAD-COVERAGE subset (real decile sort possible)")
    print("z_it = {PCA, FF5}  (NN2, NN3, NC Ridge, NC LASSO omitted -- restrict universe too much)")
    print("=" * 100)
    rows_b = []
    for d in DEPTHS:
        try:
            rows_b.append(run_depth(d, benchmark_panel_broad, realized, BENCH_COLS_BROAD, min_firms_for_sort=10))
        except Exception as e:
            print(f"depth={d} FAILED: {e}")
    results_b = pd.DataFrame(rows_b).set_index("depth")
    print(results_b[["n_obs", "n_months", "avg_firms_per_month", "a_hat", "a_tstat_NW12", "R2"]])
    print("\nTable 5 analogue -- Sharpe ratio of long-short portfolio on residualized GNN pred:")
    print(results_b[["F_n_months", "mean", "vol", "sharpe"]])

    print("\n=== Comparison vs paper Table 5 Sharpe ratios (monthly, non-annualized) ===")
    cmp = pd.DataFrame({
        "computed_sharpe_variantB": results_b["sharpe"],
        "paper_sharpe": pd.Series(paper_table5),
    })
    cmp["diff"] = cmp["computed_sharpe_variantB"] - cmp["paper_sharpe"]
    print(cmp)

    results_a.to_csv(f"{DATA}/table5_replication_variantA_full4bench.csv")
    results_b.to_csv(f"{DATA}/table5_replication_variantB_pca_ff5.csv")
    print(f"\nSaved: {DATA}/table5_replication_variantA_full4bench.csv")
    print(f"Saved: {DATA}/table5_replication_variantB_pca_ff5.csv")


if __name__ == "__main__":
    main()
