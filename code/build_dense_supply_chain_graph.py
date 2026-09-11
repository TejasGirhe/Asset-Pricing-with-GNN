#!/usr/bin/env python
"""
Rebuild features_and_supply_chain.csv using the denser supply-chain edge list
found in G:\\My Drive\\HFHFT\\230ZA\\features_and_supply_chain.csv (firm_a/firm_b/
srcdate/available_from columns), instead of this project's original WRDS Compustat
Segment-derived graph (scg_edges_yearly.parquet, median degree 2, 940 counterparties).

The Drive version already encodes point-in-time correctness itself (every edge row's
yyyymm >= available_from, confirmed by direct inspection), so we only need to extract
the (gvkey, yyyymm, counterparty_gvkey) edge triples and merge them onto our own,
larger features.csv universe (10,587 gvkeys vs. the Drive file's 5,541) -- this keeps
our full firm panel while adopting the denser graph structure.

Output: data/features_and_supply_chain.csv (overwrites the WRDS-only version;
the original is backed up first).
"""
import shutil
from pathlib import Path

import pandas as pd

DRIVE_PATH = r"G:\My Drive\HFHFT\230ZA\features_and_supply_chain.csv"
FEATURES_PATH = "../data/features.csv"
OUTPUT_PATH = "../data/features_and_supply_chain.csv"


def main():
    # Back up the existing (WRDS-only) merged file before overwriting
    existing = Path(OUTPUT_PATH)
    if existing.exists():
        backup = existing.with_name(existing.stem + "_wrds_only_backup" + existing.suffix)
        if not backup.exists():
            shutil.copy(existing, backup)
            print(f"Backed up existing file to {backup}")

    print("Loading dense edge list from Google Drive...")
    edges = pd.read_csv(DRIVE_PATH, usecols=["gvkey", "yyyymm", "firm_b"])
    edges = edges.dropna(subset=["firm_b"])
    edges["gvkey"] = edges["gvkey"].astype(int).astype(str)
    edges["firm_b"] = edges["firm_b"].astype(int).astype(str)
    edges = edges.rename(columns={"firm_b": "supplier_or_customer_gvkey"})
    print(f"  {len(edges)} edge rows, {edges['gvkey'].nunique()} distinct disclosing gvkeys, "
          f"{edges['supplier_or_customer_gvkey'].nunique()} distinct counterparties")

    print("Loading our features.csv universe...")
    features = pd.read_csv(FEATURES_PATH)
    # features.csv carries a stray pandas index column from an earlier to_csv() call
    # upstream; the original WRDS-based features_and_supply_chain.csv never had it, and
    # GNN model.ipynb's feature_cols = data_with_graph.columns[7:71] positional slice
    # assumes that same (no index column) layout -- keeping it here shifts every column
    # by one and silently feeds sic2..Tan into the model as "characteristics" instead of
    # AT..Total_vol. Drop it to match the original schema exactly.
    if "Unnamed: 0" in features.columns:
        features = features.drop(columns=["Unnamed: 0"])
    features["gvkey"] = features["gvkey"].astype(str)
    print(f"  {len(features)} rows, {features['gvkey'].nunique()} distinct gvkeys")

    # A firm can have multiple counterparties in the same month -- keep all edges,
    # matching the original schema's convention (one row per gvkey-yyyymm-counterparty).
    merged = features.merge(edges, on=["gvkey", "yyyymm"], how="left")

    # supplier_or_customer_flag / sales_to_customer existed in the old schema but aren't
    # available from this edge source -- keep the columns for downstream compatibility
    # (GNN model.ipynb references supplier_or_customer_flag when building edge direction)
    # but fill with a neutral default: flag=1 (treat all edges as undirected "customer"
    # link per this source's convention; the Drive file provided no directionality info).
    if "supplier_or_customer_flag" not in merged.columns:
        merged["supplier_or_customer_flag"] = merged["supplier_or_customer_gvkey"].notna().astype(int)
    if "sales_to_customer" not in merged.columns:
        merged["sales_to_customer"] = pd.NA

    merged = merged.drop_duplicates()
    merged.to_csv(OUTPUT_PATH, index=False)

    n_edge_rows = merged["supplier_or_customer_gvkey"].notna().sum()
    n_counterparties = merged["supplier_or_customer_gvkey"].nunique()
    print(f"\nSaved {OUTPUT_PATH}: {len(merged)} rows, {n_edge_rows} carry an edge, "
          f"{n_counterparties} distinct counterparties")


if __name__ == "__main__":
    main()
