#!/usr/bin/env python
"""
Probe for FactSet Revere (supply-chain relationship data) access on WRDS.

Usage:
    pip install wrds pandas
    python probe_factset_revere.py

Mirrors the probing approach in wrds_pull_crsp.py: library *visibility* isn't
proof of read access (crsp_q_stock showed up but was permission-denied), so
this actually attempts to read a row from every FactSet-ish library/table it
can find, rather than just listing schema names.

Written as an add-on; it does not modify any existing file in this project.
"""
import wrds

db = wrds.Connection()

print("All libraries visible to this account containing 'factset' or 'revere':")
candidates = [l for l in db.list_libraries() if 'factset' in l.lower() or 'revere' in l.lower()]
for l in candidates:
    print("   ", l)
if not candidates:
    print("   (none found -- your account likely does not have FactSet Revere)")

print("\nReadability probe (actual SELECT attempts, not just visibility):")
for lib in candidates:
    try:
        tables = db.list_tables(lib)
    except Exception as e:
        print(f"    could not list tables in {lib}: {e}")
        continue
    supply_tables = [t for t in tables if 'supply' in t.lower() or 'revere' in t.lower() or 'relationship' in t.lower()]
    tables_to_check = supply_tables if supply_tables else tables[:15]  # cap if nothing obviously named
    for tbl in tables_to_check:
        try:
            db.raw_sql(f"select * from {lib}.{tbl} limit 1")
            print(f"    OK       {lib}.{tbl}")
        except Exception as e:
            print(f"    no       {lib}.{tbl}  -- {str(e).strip().splitlines()[0][:90]}")

db.close()
