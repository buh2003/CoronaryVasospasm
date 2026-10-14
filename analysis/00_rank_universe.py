#!/usr/bin/env python3
"""Within-library percentile ranks on a fixed gene universe.

Usage
-----
    python analysis/00_rank_universe.py --proj .

Inputs
    data/counts_qc_pass.csv
    data/ziegler_tpm_matrix.csv

Outputs
    results/percentile_ranks_universe.csv
"""
import argparse
import os
import sys

import pandas as pd

DETECT_MIN = 1.0   # CPM (patient) or TPM (donor) at or above this counts as detected
PATIENT = ["1", "2a", "3"]


def keyed(df, cols):
    """Index on unversioned Ensembl id, keeping the most abundant copy of a duplicate."""
    d = df.copy()
    d["gid"] = d["gene_id"].astype(str).str.split(".").str[0]
    d["_m"] = d[cols].mean(axis=1)
    d = d.sort_values("_m", ascending=False).drop_duplicates("gid", keep="first")
    return d.set_index("gid")[["gene_name"] + cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", default=".")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    d = lambda *p: os.path.join(a.proj, "data", *p)
    out = a.outdir or os.path.join(a.proj, "results")
    os.makedirs(out, exist_ok=True)

    pat = pd.read_csv(d("counts_qc_pass.csv"))
    pat[PATIENT] = pat[PATIENT] / pat[PATIENT].sum() * 1e6

    zt = pd.read_csv(d("ziegler_tpm_matrix.csv"))
    zcols = [c for c in zt.columns if c.startswith("hSCG")]

    P, Z = keyed(pat, PATIENT), keyed(zt, zcols)
    shared = P.index.intersection(Z.index)
    X = pd.concat([P.loc[shared, PATIENT], Z.loc[shared, zcols]], axis=1)
    names = P.loc[shared, "gene_name"]

    universe = X.index[(X >= DETECT_MIN).all(axis=1)]
    R = X.loc[universe].rank(axis=0, pct=True) * 100.0

    assert R.shape[1] == 9, f"expected nine libraries, got {R.shape[1]}"
    assert R.max().max() <= 100.0 and R.min().min() > 0.0

    R.assign(gene_name=names.loc[universe]).to_csv(
        os.path.join(out, "percentile_ranks_universe.csv"))
    print(f"shared gene ids {len(shared):,}")
    print(f"detected at >= {DETECT_MIN} in all nine libraries {len(universe):,}")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
