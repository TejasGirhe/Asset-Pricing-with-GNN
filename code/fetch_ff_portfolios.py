#!/usr/bin/env python
"""
Fetch FF25 and FF30 portfolio return files from Kenneth R. French's Data
Library and write them into data/ in the exact shape GNN_code_additional.ipynb
expects (french_25.csv, french_30_industry_eq_weighted.csv).

Run this on your own machine with normal internet access -- e.g. in
Terminal.app, NOT through the Claude session's device shell, which proxies
outbound requests through a restricted allowlist that blocks
mba.tuck.dartmouth.edu.

Usage:
    pip install pandas requests
    python fetch_ff_portfolios.py
    (run from anywhere -- it writes directly to the project's data/ folder
    via the OUT_DIR path below; edit OUT_DIR if your checkout lives elsewhere)
"""
import io
import re
import zipfile

import pandas as pd
import requests

OUT_DIR = "/Users/aaryenmehta/LocalFiles/MFE Readings/Hedge Fund/hfhf/code/gnn_230p_paper_strat/data"

FF_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
FF25_URL = f"{FF_BASE}/25_Portfolios_5x5_CSV.zip"
FF30_URL = f"{FF_BASE}/30_Industry_Portfolios_CSV.zip"


def _download_zip_csv(url: str) -> str:
    """Download a Ken French CSV.zip and return the inner CSV's raw text."""
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    inner_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
    return zf.read(inner_name).decode("latin-1")


def _extract_section(raw_text: str, section_header: str, next_header_hint: str = None) -> pd.DataFrame:
    """
    Ken French's CSV.zip files bundle several tables (value-weighted returns,
    equal-weighted returns, firm counts, avg firm size, ...) into one CSV,
    each preceded by its own text header line and separated by blank lines.
    This pulls out just the table whose header line contains `section_header`.
    """
    lines = raw_text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if section_header.lower() in line.lower():
            start = i + 1
            break
    if start is None:
        raise ValueError(f"Could not find section '{section_header}' in file")

    # Table runs until the next blank line or the next all-caps section header
    end = len(lines)
    for i in range(start, len(lines)):
        stripped = lines[i].strip()
        if stripped == "" or (i > start and re.match(r"^[A-Za-z].*Weighted|^[A-Za-z].*Firms|^[A-Za-z].*Size", stripped)):
            end = i
            break

    table_text = "\n".join(lines[start:end])
    df = pd.read_csv(io.StringIO(table_text))
    df.columns = [c.strip() for c in df.columns]
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "date"})
    df = df[pd.to_numeric(df["date"], errors="coerce").notna()].reset_index(drop=True)
    df["date"] = df["date"].astype(int)
    return df


def main():
    print("Downloading FF25 (25 Size/Book-to-Market portfolios)...")
    ff25_raw = _download_zip_csv(FF25_URL)
    # Standard convention for FF25 as regression test assets is the
    # value-weighted monthly series. If your project's notebook expects
    # equal-weighted instead, change the section string below to
    # "Average Equal Weighted Returns -- Monthly".
    ff25 = _extract_section(ff25_raw, "Average Value Weighted Returns -- Monthly")
    ff25_path = f"{OUT_DIR}/french_25.csv"
    ff25.to_csv(ff25_path, index=False)
    print(f"  wrote {ff25_path}: {ff25.shape}")

    print("Downloading FF30 (30 Industry portfolios, equal-weighted)...")
    ff30_raw = _download_zip_csv(FF30_URL)
    ff30 = _extract_section(ff30_raw, "Average Equal Weighted Returns -- Monthly")
    ff30_path = f"{OUT_DIR}/french_30_industry_eq_weighted.csv"
    ff30.to_csv(ff30_path, index=False)
    print(f"  wrote {ff30_path}: {ff30.shape}")

    print("\nDone. french_25.csv and french_30_industry_eq_weighted.csv are in data/.")
    print("NOTE: aipt_13_themes_factors.csv (JKP13 theme factors) is NOT handled by")
    print("this script -- see the accompanying note on why that one needs a different")
    print("approach.")


if __name__ == "__main__":
    main()
