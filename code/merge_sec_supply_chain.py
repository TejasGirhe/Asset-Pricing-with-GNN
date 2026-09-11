#!/usr/bin/env python
"""
Merge SEC 10-K/10-Q-derived supplier/customer edges (extracted from the user's separate
AB IP knowledge-graph project) with the existing WRDS/Compustat-Segment-Customer supply
chain edges (data/scg_edges_yearly.parquet), and rebuild an ENRICHED version of
data/features_and_supply_chain.csv for validation.

Does NOT overwrite data/features_and_supply_chain.csv or any existing parquet -- writes to
new files only:
  - data/scg_edges_sec.parquet              -- the new SEC-derived directed edges alone
  - data/scg_edges_combined.parquet         -- union of WRDS + SEC directed edges, deduped,
                                                with a `source` column ('WRDS' / 'SEC_10K10Q'
                                                / 'BOTH')
  - data/features_and_supply_chain_enriched.csv -- features.csv left-joined against the
                                                combined edge set, mirroring
                                                build_supply_chain_features.py's month
                                                expansion / lag convention

=== Source: SEC 10-K/10-Q triplets (D:\\UCB MFE\\AB IP\\cleaned_triplets_final.csv) ===

Extraction pipeline (see scratchpad scripts this was developed with -- summarized here):
  1. Filtered the 377MB triplets file (873,575 rows total) to relationship in
     {Supplies, Depends_On, Partners_With, Has_Stake_In, Involved_In} AND both
     entity_type and target_type in {ORG, COMP} (firm-to-firm candidates only,
     excluding firm-to-product/concept/person edges) -> 24,042 candidate rows.
  2. Applied confidence >= 0.80 (the dataset's confidence values range 0.70-0.99 with
     median ~0.81-0.88 depending on relationship type; 0.80 keeps roughly the upper half
     of each relationship type while remaining a meaningful bar above the observed floor)
     -> 17,446 rows.
  3. Kept only Supplies (2,307 rows) and Depends_On (1,359 rows) as directional supply-chain
     edges; Partners_With (6,548 rows) was extracted separately as an UNDIRECTED pair list
     (not merged into the directional supplier->customer edge set, since "partners with"
     does not imply a supply direction) and is NOT included in the combined output below --
     see scg_edges_sec_partners.parquet if needed later. Has_Stake_In and Involved_In were
     inspected and dropped: spot-checking showed these are dominated by ownership stakes,
     subsidiaries, business units, and internal committees (e.g. "Accenture -> Has_Stake_In
     -> accenture capital", "Fortinet -> Involved_In -> risk management committee"), not
     firm-to-firm supply relationships.
  4. Name/ticker resolution to gvkey:
       - entity (the filer's own name) -> gvkey: resolved via the filer's OWN ticker
         (the triplets file's `ticker` column), not via fuzzy name matching. Built a
         ticker -> gvkey table for the 507 unique SEC filer tickers by (a) finding each
         ticker's most common self-referential `entity` string within its own filings,
         (b) normalizing company-name suffixes (Inc/Corp/Ltd/plc/the/etc.), (c) exact
         matching against normalized data/crsp_monthly.parquet `comnam` values, with a
         conservative fuzzy fallback (rapidfuzz token_set_ratio >= 88, first-token-must-
         match guard) for near-misses caused by suffix/spacing differences, each of which
         was manually spot-checked, (d) chaining permno -> gvkey via data/ccm_link.parquet.
         Result: 399/507 tickers (78.7%) resolved to a gvkey; 395/507 (77.9%) resolved to
         a gvkey that is actually present in data/features.csv. The unresolved ~22% were
         inspected and are overwhelmingly NOT a matching failure: data/crsp_monthly.parquet
         is pre-filtered to shrcd 10/11 common shares only, which structurally excludes
         REITs (AvalonBay, Prologis, Public Storage, Digital Realty, Welltower, etc.) and
         foreign-incorporated firms (Arch Capital, Aptiv, Amcor, Linde, TE Connectivity,
         NXP, Garmin, Schlumberger, etc.) that are legitimately outside this replication's
         CRSP panel, not just outside this script's matching ability.
       - target (the counterparty) -> gvkey: resolved via EXACT normalized-name matching
         against data/crsp_monthly.parquet `comnam` ONLY (no fuzzy fallback). A fuzzy
         fallback was tried and rejected: spot-checking showed it produced false-positive
         matches on short/generic free-text phrases extracted by the NER pipeline (e.g.
         "watson", "banks and credit card issuers", "u.s. postal service" all matched
         *something* in CRSP via token-subset overlap despite not being the named firm).
         Given counterparty names cannot be cross-checked against a known ticker the way
         the filer side can, precision was prioritized over recall. Result: 784/4,524
         (17.3%) unique counterparty names resolved to a gvkey. A small generic-term
         blocklist (~40 terms: "suppliers", "retailers", "vendors", "franchisees", etc.)
         was also applied to avoid even attempting to match obviously non-company noun
         phrases that the NER extraction mislabeled as ORG/COMP.
  5. Direction: Supplies -> entity is supplier, target is customer. Depends_On -> target is
     supplier, entity is customer (entity depends on target for something it needs, i.e.
     target supplies entity). Both spot-checked against chunk_text for several examples
     (Skyworks->Google as a listed customer, BorgWarner->GM as an OEM customer, Best Buy
     sourcing from Hewlett-Packard, AbbVie/Amgen distributing via wholesalers, Align
     depending on UPS for freight, Corteva depending on Dow for IP licensing) and all
     checked out directionally correct.
  6. Self-loops (filer_gvkey == counterparty_gvkey, e.g. a company mentioning a subsidiary
     or product under its own corporate name) were dropped.
  7. Rows deduped on (supplier_gvkey, customer_gvkey, year), keeping the max-confidence
     instance when a pair/year was extracted from multiple filing chunks.

Result: 277 directed SEC-derived supplier->customer edges, spanning 154 unique gvkeys,
across fiscal years 2014-2026 (2026 reflects filings dated into the current fiscal year at
extraction time; treated identically to other years by the lag/hold-fixed logic below).

=== GDELT co-mention data: NOT used ===
D:\\UCB MFE\\AB IP\\SEC_10K10Q_GDELT\\gdelt_data\\sp500_comention_all.csv was inspected but
deliberately excluded from this merge. It records same-day news co-mentions between ticker
pairs, which is a proximity/attention signal, not a supplier/customer relationship -- two
firms can co-appear in news for any reason (M&A rumors, sector-wide stories, unrelated
events on the same day). Unlike the SEC triplets (which cite an explicit textual relation
with a directional verb and a confidence score from an extraction model), GDELT co-mentions
carry no directional or causal semantics, and folding them into the same edge type as
"Supplies"/"Depends_On" edges risks diluting the supply-chain graph with noise the NC Ridge/
NC LASSO neighboring-characteristics benchmarks would then treat as a real economic link.
If desired later, GDELT could be used as a supplementary EDGE WEIGHT or confidence booster
strictly for gvkey pairs that ALREADY have a SEC- or WRDS-derived edge (i.e. corroboration,
never as a standalone edge source) -- left as a follow-up, not implemented here.

=== Merge strategy ===
Union of WRDS edges (customer side already resolved gvkey via ccm_link, per
build_supply_chain_features.py) and the new SEC-derived directed edges, on
(supplier_gvkey, customer_gvkey, year). Where a pair/year appears in both sources, the row
is tagged source='BOTH' and the WRDS row's sales_to_customer weight is kept (WRDS is the
higher-precision, quantitatively-weighted source); rows unique to one source are tagged
accordingly. This is a UNION (not an intersection) -- the whole point of adding the SEC
source is to add coverage the WRDS pull is missing, so a SEC-only edge is kept even though
it lacks a dollar-sales weight.
"""
import pandas as pd

REPO = r'D:\UCB MFE\Sem3\230ZA\230ZA Replication'
SCRATCH = r'C:\Users\tejas\AppData\Local\Temp\claude\d--UCB-MFE-Sem3-230ZA-230ZA-Replication\4259a9f4-340e-4a79-bbcc-2d5ef4fe4caa\scratchpad'

WRDS_EDGES = f'{REPO}\\data\\scg_edges_yearly.parquet'
CCM_LINK = f'{REPO}\\data\\ccm_link.parquet'
FEATURES = f'{REPO}\\data\\features.csv'
SEC_EDGES = f'{SCRATCH}\\sec_directed_edges.parquet'
SEC_PARTNERS = f'{SCRATCH}\\sec_partners_edges.parquet'

OUT_SEC_ONLY = f'{REPO}\\data\\scg_edges_sec.parquet'
OUT_PARTNERS = f'{REPO}\\data\\scg_edges_sec_partners.parquet'
OUT_COMBINED = f'{REPO}\\data\\scg_edges_combined.parquet'
OUT_FEATURES = f'{REPO}\\data\\features_and_supply_chain_enriched.csv'


def add_months(yyyymm: int, n: int) -> int:
    year = yyyymm // 100
    month = yyyymm % 100
    month += n
    year += (month - 1) // 12
    month = (month - 1) % 12 + 1
    return year * 100 + month


def main():
    print('=' * 70)
    print('STEP 1: Load and prep WRDS edges (customer_permno -> customer_gvkey via CCM)')
    print('=' * 70)
    wrds = pd.read_parquet(WRDS_EDGES, columns=['year', 'supplier_permno', 'customer_permno',
                                                  'supplier_gvkey', 'sales_to_customer'])
    wrds = wrds.drop_duplicates(subset=['supplier_permno', 'customer_permno', 'year'])

    ccm = pd.read_parquet(CCM_LINK)
    ccm = ccm[ccm['linktype'].isin(['LU', 'LC']) & ccm['linkprim'].isin(['P', 'C'])].copy()
    ccm['permno'] = ccm['permno'].astype(int)

    wrds['_year_end'] = pd.to_datetime(wrds['year'].astype(str) + '-12-31')
    m = wrds.merge(
        ccm[['permno', 'gvkey', 'linkdt', 'linkenddt']].rename(columns={'permno': 'customer_permno'}),
        on='customer_permno', how='left',
    )
    mask = (m['_year_end'] >= m['linkdt']) & (m['linkenddt'].isna() | (m['_year_end'] <= m['linkenddt']))
    m = m[mask].drop(columns=['linkdt', 'linkenddt', '_year_end']).rename(columns={'gvkey': 'customer_gvkey'})
    m = m.drop_duplicates(subset=['supplier_permno', 'customer_permno', 'year', 'customer_gvkey'])
    m = m.dropna(subset=['customer_gvkey', 'supplier_gvkey'])

    wrds_edges = m.rename(columns={'supplier_gvkey': 'supplier_gvkey', 'customer_gvkey': 'customer_gvkey'})
    wrds_edges = wrds_edges[['supplier_gvkey', 'customer_gvkey', 'year', 'sales_to_customer']].copy()
    wrds_edges['source'] = 'WRDS'
    wrds_edges['confidence'] = pd.NA
    print(f'  WRDS directed edges (gvkey-gvkey): {len(wrds_edges):,}, '
          f'{wrds_edges["supplier_gvkey"].nunique() + 0} unique suppliers, '
          f'{len(set(wrds_edges.supplier_gvkey) | set(wrds_edges.customer_gvkey)):,} unique gvkeys w/ edge')

    print()
    print('=' * 70)
    print('STEP 2: Load SEC-derived directed edges')
    print('=' * 70)
    sec = pd.read_parquet(SEC_EDGES)
    sec_edges = sec[['supplier_gvkey', 'customer_gvkey', 'year', 'confidence']].copy()
    sec_edges['sales_to_customer'] = pd.NA
    sec_edges['source'] = 'SEC_10K10Q'
    print(f'  SEC directed edges: {len(sec_edges):,}, '
          f'{len(set(sec_edges.supplier_gvkey) | set(sec_edges.customer_gvkey)):,} unique gvkeys w/ edge')
    sec_edges.to_parquet(OUT_SEC_ONLY, index=False)
    print(f'  wrote {OUT_SEC_ONLY}')

    partners = pd.read_parquet(SEC_PARTNERS)
    partners.to_parquet(OUT_PARTNERS, index=False)
    print(f'  wrote {OUT_PARTNERS} ({len(partners):,} undirected Partners_With pairs, NOT merged below)')

    print()
    print('=' * 70)
    print('STEP 3: Union WRDS + SEC edges on (supplier_gvkey, customer_gvkey, year)')
    print('=' * 70)
    both = pd.concat([wrds_edges, sec_edges], ignore_index=True)
    key = ['supplier_gvkey', 'customer_gvkey', 'year']

    dupe_mask = both.duplicated(subset=key, keep=False)
    in_both_keys = set(map(tuple, both.loc[dupe_mask, key].drop_duplicates().values))
    print(f'  gvkey-pair/year combos found in BOTH sources: {len(in_both_keys):,}')

    # Keep WRDS row (has sales weight) when both agree; else keep whichever single source has it.
    both['_is_wrds'] = (both['source'] == 'WRDS').astype(int)
    combined = both.sort_values('_is_wrds', ascending=False).drop_duplicates(subset=key, keep='first').copy()
    combined = combined.drop(columns=['_is_wrds'])

    def tag_source(row):
        k = (row['supplier_gvkey'], row['customer_gvkey'], row['year'])
        return 'BOTH' if k in in_both_keys else row['source']
    combined['source'] = combined.apply(tag_source, axis=1)

    print(f'  Combined directed edges: {len(combined):,}')
    print(combined['source'].value_counts())
    gvkeys_before = set(wrds_edges['supplier_gvkey']) | set(wrds_edges['customer_gvkey'])
    gvkeys_after = set(combined['supplier_gvkey']) | set(combined['customer_gvkey'])
    print(f'  Unique gvkeys with >=1 edge -- WRDS only: {len(gvkeys_before):,}  ->  combined: {len(gvkeys_after):,}'
          f'  (+{len(gvkeys_after - gvkeys_before):,} new gvkeys from SEC source)')

    combined.to_parquet(OUT_COMBINED, index=False)
    print(f'  wrote {OUT_COMBINED}')

    print()
    print('=' * 70)
    print('STEP 4: Rebuild enriched features_and_supply_chain (monthly expansion, 7mo lag)')
    print('=' * 70)
    combined['active_start_yyyymm'] = combined['year'].apply(lambda y: add_months(y * 100 + 12, 7))

    expanded_rows = []
    for offset in range(12):
        chunk = combined[['supplier_gvkey', 'customer_gvkey', 'sales_to_customer', 'source',
                           'active_start_yyyymm']].copy()
        chunk['yyyymm'] = chunk['active_start_yyyymm'].apply(lambda ym: add_months(ym, offset))
        expanded_rows.append(chunk[['supplier_gvkey', 'customer_gvkey', 'sales_to_customer', 'source', 'yyyymm']])
    edges_monthly = pd.concat(expanded_rows, ignore_index=True)
    del expanded_rows

    before = len(edges_monthly)
    # if a pair has both WRDS+SEC rows in different years landing on the same month, prefer BOTH/WRDS row
    edges_monthly['_src_rank'] = edges_monthly['source'].map({'BOTH': 0, 'WRDS': 1, 'SEC_10K10Q': 2}).fillna(3)
    edges_monthly = edges_monthly.sort_values('_src_rank').drop_duplicates(
        subset=['supplier_gvkey', 'customer_gvkey', 'yyyymm'], keep='first'
    ).drop(columns=['_src_rank'])
    print(f'  {before:,} expanded rows -> {len(edges_monthly):,} after dropping overlap duplicates')

    edges_monthly = edges_monthly.rename(columns={'supplier_gvkey': 'gvkey',
                                                    'customer_gvkey': 'supplier_or_customer_gvkey'})
    edges_monthly['supplier_or_customer_flag'] = 1

    features = pd.read_csv(FEATURES)
    if 'Unnamed: 0' in features.columns:
        features = features.drop(columns=['Unnamed: 0'])
    # gvkey (the merge key against features.csv) is a mix of WRDS- and SEC-sourced values,
    # both of which carry Compustat's canonical zero-padded 6-digit format (e.g. '024856'),
    # while features.csv's gvkey is a plain int that str()'s to '24856' with no padding --
    # confirmed via build_supply_chain_features.py's identical bug: 67.9% of WRDS edges'
    # supplier_gvkey values were zero-padded and silently failed this merge, and only 4 of
    # 127 unique SEC gvkey-pairs happened to have neither side zero-padded (surviving by
    # coincidence). Strip leading zeros on the edge side so the merge actually works.
    # supplier_or_customer_gvkey (the customer side) is NOT touched here -- it's only ever
    # compared against features.csv's gvkey downstream in GNN model.ipynb's
    # extract_neighbor_features(), which does the same str-cast; leaving it zero-padded
    # would silently break neighbor lookups the same way, so strip it too for consistency.
    features['gvkey'] = features['gvkey'].astype(str)
    edges_monthly['gvkey'] = edges_monthly['gvkey'].astype(str).str.lstrip('0')
    edges_monthly['supplier_or_customer_gvkey'] = (
        edges_monthly['supplier_or_customer_gvkey'].astype(str).str.replace(r'\.0$', '', regex=True).str.lstrip('0')
    )

    merged = features.merge(edges_monthly, on=['gvkey', 'yyyymm'], how='left')
    print(f'  features.csv: {len(features):,} rows -> merged: {len(merged):,} rows')
    n_with_edge = merged['supplier_or_customer_gvkey'].notna().sum()
    print(f'  rows with a known supply chain edge: {n_with_edge:,} ({n_with_edge / len(merged) * 100:.2f}%)')
    gvkeys_with_edge = merged.loc[merged['supplier_or_customer_gvkey'].notna(), 'gvkey'].nunique()
    print(f'  unique gvkeys with at least one edge: {gvkeys_with_edge:,} / {merged["gvkey"].nunique():,} total gvkeys')

    merged.to_csv(OUT_FEATURES, index=False)
    print(f'\nWrote {OUT_FEATURES}: {len(merged):,} rows, {len(merged.columns)} columns')


if __name__ == '__main__':
    main()
