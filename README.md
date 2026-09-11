# Asset Pricing with a Supply-Chain GNN — Replication

Replication of **"Graph Machine Learning for Asset Pricing: Traversing the Supply Chain"**
(Capponi, Sidaoui & Zou, 2026). Paper: [`GNN_230P.pdf`](./GNN_230P.pdf).
Authors' reference code: <https://github.com/agcappo/SupplyChainAssetPricing>.

## Status

The full pipeline runs end to end and is reproducible (seeded). The paper's
headline Table 5 result — an out-of-sample Sharpe ratio of 1.07 that peaks at
message-passing depth `d=3` and collapses to `0.08` by `d=10` — is **not
reproduced** under any benchmark configuration built here. The gap traces mostly
to data: the paper's supply-chain graph is FactSet Revere (unavailable), and two
of six spanning-regression benchmarks (NN2, NN3) are not yet built.

| depth | paper | ours (PCA+FF5) | ours (4-benchmark) |
|------:|------:|---------------:|-------------------:|
| 1     | 0.87  | 1.23           | −0.07              |
| 3     | 1.07  | 1.97           | 0.06               |
| 6     | 0.64  | 0.91           | −0.05              |
| 10    | 0.08  | 0.68           | 0.09               |

Monthly, non-annualized Sharpe of the long-short decile portfolio on the
residualized GNN predicted return. See the replication memo for the full table
and the reasoning behind each divergence.

## Pipeline

1. **`code/FirmFeatures_Calculations.ipynb`** — builds `data/features.csv`
   (~1.07M firm-months, 2003–2024, 65 Freyberger et al. 2020 characteristics)
   from a WRDS CRSP + Compustat pull.
2. **`code/build_supply_chain_features.py`** — merges the supply-chain edge
   table into the feature panel → `data/features_and_supply_chain.csv`.
   Uses WRDS "Supply Chain with IDs (Compustat Segment)" as a substitute for the
   paper's FactSet Revere.
3. **`code/GNN model.ipynb`** — trains the Transformer GNN at depths `d=1…10`
   (dim 40, 2 heads, 80 epochs, dropout 0.3, seeded) and the PCA / RPPCA / FF5 /
   NC-Ridge / NC-LASSO benchmarks; writes per-config prediction and embedding
   CSVs to `data/`.
4. **`code/table5_replication.py`** — the paper's Table 5: joint spanning
   regression of the GNN predicted return on the benchmark menu, then the Sharpe
   ratio of the long-short portfolio sorted on the residual. Runs two variants
   (PCA+FF5, and the full 4-benchmark set).

### Supplementary

- **`code/merge_sec_supply_chain.py`** — enriches the supply-chain graph with
  customer/supplier edges extracted from SEC 10-K/10-Q filings (an external
  knowledge-graph project). Adds ~39 firms beyond the WRDS graph; bottlenecked
  on counterparty-name → gvkey resolution.
- **`code/GNN_code_additional.ipynb`** — MLP benchmark, cross-sectional R²,
  oversmoothing diagnostics. Several sections depend on FactSet-based reference
  files and are disabled.
- **`code/fetch_ff_portfolios.py`**, **`code/probe_factset_revere.py`** —
  Fama-French portfolio pull and a WRDS FactSet-access probe.

## Fixes applied during replication

| # | Issue | Effect |
|--:|-------|--------|
| 1 | Train/test split was Dec 2006, not the paper's Dec 2016 | Model was training on 4 yrs / testing on 17, vs the paper's 14 / 7 |
| 2 | 6 epochs / dropout 0.5 instead of the paper's tuned 80 / 0.3 | Sharpe went from a flat −0.16…0.11 to a plausible 0.2…1.4 |
| 3 | GNN block missing the residual + LayerNorm + FFN of Appendix B.1 | Rebuilding to match made results *worse* — reverted |
| 4 | Neighbor-feature lookup searched the edge-only subset, not the full panel | NC Ridge/LASSO coverage: 37 firms → 641 |
| 5 | No random seed — identical runs gave different Sharpe ratios | Now reproducible to 4 dp across retrains |
| 6 | `gvkey` compared as `"024856"` vs `24856` — most edges silently unmatched | Firms with an edge: 651 → 2,006 (3.1×) |

## Data

`data/` is **not tracked** (large, WRDS-licensed). To reproduce:

- Pull CRSP monthly + daily, Compustat annual + quarterly, the CCM link table,
  and the Compustat Segment supply-chain table from WRDS; place them in `data/`
  as the parquet files `FirmFeatures_Calculations.ipynb` expects.
- `data/ff_mon.csv` (Fama-French 5-factor monthly) and the French 25 / 30
  benchmark portfolio files are pulled by `fetch_ff_portfolios.py`.
- Working copies also live in the project Google Drive folder.

## Known gaps

- **NN2 / NN3 benchmarks not built** — the spanning regression uses 4 of the
  paper's 6 controls. NN2/NN3 need a self-contained 2- and 3-layer MLP over the
  existing panel.
- **Supply-chain graph covers 19% of firms** (2,006 / 10,587), and unlike the
  paper's graph it retains only listed firms — no unlisted counterparty nodes,
  so multi-hop structure is thinner.
- **`generate_oos_sr` cell** in `GNN model.ipynb` still throws `LinAlgError` on a
  near-singular covariance matrix; cosmetic, runs with `--allow-errors`.
