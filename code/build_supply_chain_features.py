#!/usr/bin/env python
"""
Build data/features_and_supply_chain.csv from data/scg_edges_yearly.parquet +
data/features.csv, for GNN model.ipynb (cell 37 onward).

Source: data/scg_edges_yearly.parquet -- the user's own WRDS supply-chain pull, one row
per (year, supplier_permno, customer_permno) disclosed relationship, with a sales-to-customer
weight. This replaces the previous data/wrds_supply_chain.csv ("Supply Chain with IDs
(Compustat Segment)") this script used to read.

Schema differences from the old wrds_supply_chain.csv this script previously targeted:
  * Keyed by PERMNO on both sides (supplier_permno, customer_permno), not by gvkey/cgvkey.
    Only supplier_gvkey is given directly; customer_gvkey is not, so we map
    customer_permno -> customer_gvkey via ccm_link.parquet (the standard CRSP-Compustat
    merged link table: keep only LU/LC link types and P/C primary-link flags, matched by
    the edge's disclosure year falling inside [linkdt, linkenddt]).
  * Dated by calendar `year`, not a YYYYMM `srcdate` int -- see the lag/hold-fixed
    discussion below for how the month-expansion logic is adapted for this.
  * Carries a `sales_to_customer` weight (not available in the old source); kept as an
    extra optional column since GNN model.ipynb's prepare_graph_data() doesn't consume
    it, but it's cheap to carry through for anyone doing weighted analysis later.
  * customer_name is given (a free-text company name) but not otherwise used here.

Lag/hold-fixed convention: matches lag_annual() in FirmFeatures_Calculations.ipynb in
spirit (7-month lag for annual data), but that function lags a YYYYMM int, and here we
only have a `year`. We pick a fixed convention: treat each edge's `year` disclosure as if
it occurred at that calendar year's fiscal year-end (December, i.e. YYYYMM = year*100+12),
mirroring the fact that FirmFeatures_Calculations.ipynb's own annual-frequency Compustat
matching also keys off each firm's most recent annual datadate treated as a single
year-end event. From that year-end, the relationship becomes usable starting
(year*100+12) + 7 months = (year+1)*100+7, and -- per the paper's own text, "we hold each
year's supply chain relationships fixed across all months within that year" -- stays
active for the following 12 months.

Direction: sales_to_customer is a directional flag -- supplier_permno is always the
selling/reporting firm (the supplier) and customer_permno is always the customer named in
the disclosure (analogous structure to the Compustat Segment Customer disclosures this
script previously assumed). So supplier_or_customer_flag is a fixed constant (1) for
every row, matching GNN model.ipynb's prepare_graph_data() "supplier -> customer" edge
branch (gvkey_idx -> partner_idx when flag != 0).

Written as an add-on; it does not modify any existing file in this project.
"""
import pandas as pd

RAW = 'scg_edges_yearly.parquet'
CCM_LINK = 'ccm_link.parquet'
FEATURES = 'features.csv'
OUT = 'features_and_supply_chain.csv'


def add_months(yyyymm: int, n: int) -> int:
    """Add n months to a YYYYMM int, with correct year rollover."""
    year = yyyymm // 100
    month = yyyymm % 100
    month += n
    year += (month - 1) // 12
    month = (month - 1) % 12 + 1
    return year * 100 + month


def main():
    print("Loading raw supply chain edges (PERMNO-keyed, yearly)...")
    raw = pd.read_parquet(RAW, columns=['year', 'supplier_permno', 'customer_permno',
                                         'supplier_gvkey', 'sales_to_customer'])
    print(f"  {len(raw):,} raw rows")

    before = len(raw)
    raw = raw.drop_duplicates(subset=['supplier_permno', 'customer_permno', 'year'])
    print(f"  dropped {before - len(raw):,} exact duplicate (supplier_permno,customer_permno,year) rows -> {len(raw):,} remain")

    # Map customer_permno -> customer_gvkey via the CCM link, since only supplier_gvkey is
    # given directly. Treat the edge's disclosure as occurring at that year's fiscal
    # year-end (December) for the purpose of picking the applicable CCM link row (see
    # module docstring for why December is the chosen convention).
    print("Loading ccm_link.parquet and mapping customer_permno -> customer_gvkey...")
    ccm = pd.read_parquet(CCM_LINK)
    ccm = ccm[ccm['linktype'].isin(['LU', 'LC']) & ccm['linkprim'].isin(['P', 'C'])].copy()
    ccm['permno'] = ccm['permno'].astype(int)

    raw['_year_end'] = pd.to_datetime(raw['year'].astype(str) + '-12-31')
    merged_customer = raw.merge(
        ccm[['permno', 'gvkey', 'linkdt', 'linkenddt']].rename(columns={'permno': 'customer_permno'}),
        on='customer_permno', how='left',
    )
    mask = (merged_customer['_year_end'] >= merged_customer['linkdt']) & (
        merged_customer['linkenddt'].isna() | (merged_customer['_year_end'] <= merged_customer['linkenddt'])
    )
    merged_customer = merged_customer[mask].drop(columns=['linkdt', 'linkenddt', '_year_end'])
    merged_customer = merged_customer.rename(columns={'gvkey': 'customer_gvkey'})
    merged_customer = merged_customer.drop_duplicates(
        subset=['supplier_permno', 'customer_permno', 'year', 'customer_gvkey']
    )
    print(f"  {len(merged_customer):,} rows after CCM customer_permno -> customer_gvkey match"
          f"  ({merged_customer['customer_gvkey'].notna().mean()*100:.1f}% matched)")

    raw = merged_customer.dropna(subset=['customer_gvkey', 'supplier_gvkey'])
    raw = raw.rename(columns={'supplier_gvkey': 'gvkey', 'customer_gvkey': 'supplier_or_customer_gvkey'})

    # active_start_yyyymm: disclosure treated as occurring at that year's Dec year-end,
    # then lagged 7 months per lag_annual()'s convention (see module docstring).
    raw['active_start_yyyymm'] = raw['year'].apply(lambda y: add_months(y * 100 + 12, 7))

    print("Expanding each edge to its 12 active months (7-month lag, held fixed for the year)...")
    expanded_rows = []
    for offset in range(12):
        chunk = raw[['gvkey', 'supplier_or_customer_gvkey', 'sales_to_customer', 'active_start_yyyymm']].copy()
        chunk['yyyymm'] = chunk['active_start_yyyymm'].apply(lambda ym: add_months(ym, offset))
        expanded_rows.append(chunk[['gvkey', 'supplier_or_customer_gvkey', 'sales_to_customer', 'yyyymm']])
    edges_monthly = pd.concat(expanded_rows, ignore_index=True)
    del expanded_rows

    before = len(edges_monthly)
    edges_monthly = edges_monthly.drop_duplicates(subset=['gvkey', 'supplier_or_customer_gvkey', 'yyyymm'])
    print(f"  {before:,} expanded rows -> {len(edges_monthly):,} after dropping overlap duplicates")

    # gvkey is always the supplier, supplier_or_customer_gvkey is always the customer
    # (fixed direction in this source table -- see module docstring).
    edges_monthly['supplier_or_customer_flag'] = 1

    print("Loading features.csv and left-joining supply chain edges on (gvkey, yyyymm)...")
    features = pd.read_csv(FEATURES)
    if 'Unnamed: 0' in features.columns:
        features = features.drop(columns=['Unnamed: 0'])
    # ccm_link.parquet (and everything derived from it, including this edge table's gvkey
    # column) uses Compustat's canonical zero-padded 6-digit gvkey format (e.g. '001013'),
    # while features.csv's gvkey came through as a plain int (1013) and str()'s to '1013'
    # with no padding. '001013' != '1013', so this merge silently matched almost nothing --
    # confirmed 67.9% of edges had a zero-padded supplier_gvkey that could never match.
    # Strip leading zeros on the edge side instead of just str()-casting both sides blind.
    features['gvkey'] = features['gvkey'].astype(str)
    edges_monthly['gvkey'] = edges_monthly['gvkey'].astype(str).str.lstrip('0')

    merged = features.merge(edges_monthly, on=['gvkey', 'yyyymm'], how='left')
    print(f"  features.csv: {len(features):,} rows -> merged: {len(merged):,} rows")
    print(f"  rows with a known supply chain edge: {merged['supplier_or_customer_gvkey'].notna().sum():,}"
          f"  ({merged['supplier_or_customer_gvkey'].notna().mean()*100:.1f}%)")
    print(f"  unique gvkeys with at least one edge: {merged.loc[merged['supplier_or_customer_gvkey'].notna(), 'gvkey'].nunique():,}"
          f"  / {merged['gvkey'].nunique():,} total gvkeys in features.csv")

    merged.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}: {len(merged):,} rows, {len(merged.columns)} columns")


if __name__ == '__main__':
    main()
