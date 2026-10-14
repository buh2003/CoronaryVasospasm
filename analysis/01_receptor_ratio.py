#!/usr/bin/env python3
"""EDNRB:EDNRA within each of the nine libraries.

Usage
-----
    python analysis/01_receptor_ratio.py --proj .

Inputs
    data/counts_qc_pass.csv
    data/ziegler_tpm_matrix.csv

Outputs
    results/receptor_ratio.csv
"""
import argparse
import os
import sys

import pandas as pd

PATIENT = ["1", "2a", "3"]
RECEPTORS = ("EDNRB", "EDNRA")


def within_sample_ratio(frame, columns, to_cpm):
    """EDNRB divided by EDNRA inside each column.

    to_cpm=True for raw counts, False for values already on a per-million scale.
    """
    M = frame.drop_duplicates("gene_name").set_index("gene_name")[columns]
    for g in RECEPTORS:
        if g not in M.index:
            sys.exit(f"{g} absent from the matrix")
    if to_cpm:
        M = M.apply(lambda col: col / col.sum() * 1e6)
    return {c: M.loc[RECEPTORS[0], c] / M.loc[RECEPTORS[1], c] for c in columns}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", default=".")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    d = lambda *p: os.path.join(a.proj, "data", *p)
    out = a.outdir or os.path.join(a.proj, "results")
    os.makedirs(out, exist_ok=True)

    counts = pd.read_csv(d("counts_qc_pass.csv"))
    ziegler = pd.read_csv(d("ziegler_tpm_matrix.csv"))
    donors = [c for c in ziegler.columns if c.startswith("hSCG")]

    pat = within_sample_ratio(counts, PATIENT, to_cpm=True)
    ref = within_sample_ratio(ziegler, donors, to_cpm=False)

    rows = [{"group": "patient", "sample": s, "etb_eta_fold_linear": v}
            for s, v in pat.items()]
    rows += [{"group": "healthy" if "healthy" in s else "disease",
              "sample": s, "etb_eta_fold_linear": v} for s, v in ref.items()]
    R = pd.DataFrame(rows)

    assert len(R) == 9, f"expected nine libraries, got {len(R)}"
    assert R.etb_eta_fold_linear.gt(1).all(), "a library has EDNRA exceeding EDNRB"
    assert R.groupby("group").size().to_dict() == {"disease": 3, "healthy": 3, "patient": 3}

    R.to_csv(os.path.join(out, "receptor_ratio.csv"), index=False)
    print(R.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
    print(f"\nfold range by group:")
    for g, s in R.groupby("group").etb_eta_fold_linear:
        print(f"  {g:9} {s.min():6.1f} to {s.max():6.1f}")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
