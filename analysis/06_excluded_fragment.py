#!/usr/bin/env python3
"""Marker content of the excluded fragment 2b, in counts per million.

Usage
-----
    python analysis/06_excluded_fragment.py --proj .

Inputs
    data/counts_all_fragments.csv

Outputs
    results/excluded_fragment_markers.csv
"""
import argparse
import os

import pandas as pd

RETAINED = ["1", "2a", "3"]
EXCLUDED = "2b"

MARKERS = [
    ("TH",      "Sympathetic neuronal"),
    ("DBH",     "Sympathetic neuronal"),
    ("NPY",     "Sympathetic neuronal"),
    ("PHOX2B",  "Sympathetic neuronal"),
    ("PRPH",    "Sympathetic neuronal"),
    ("SNAP25",  "Sympathetic neuronal"),
    ("STMN2",   "Sympathetic neuronal"),
    ("S100B",   "Satellite glia / Schwann"),
    ("PLP1",    "Satellite glia / Schwann"),
    ("SOX10",   "Satellite glia / Schwann"),
    ("MPZ",     "Satellite glia / Schwann"),
    ("ACTA2",   "Mesenchymal / stromal"),
    ("MYH11",   "Mesenchymal / stromal"),
    ("DCN",     "Mesenchymal / stromal"),
    ("COL1A1",  "Mesenchymal / stromal"),
    ("PTPRC",   "Immune / endothelial"),
    ("PECAM1",  "Immune / endothelial"),
    ("CD3E",    "Immune / endothelial"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", default=".")
    a = ap.parse_args()

    counts = pd.read_csv(os.path.join(a.proj, "data", "counts_all_fragments.csv"))
    fragments = RETAINED + [EXCLUDED]
    missing = [f for f in fragments if f not in counts.columns]
    if missing:
        raise SystemExit(f"counts file is missing fragment columns: {missing}")

    cpm = counts.copy()
    for f in fragments:
        cpm[f] = counts[f] / counts[f].sum() * 1e6
    cpm = cpm.drop_duplicates("gene_name").set_index("gene_name")

    rows = []
    for gene, marker_class in MARKERS:
        if gene not in cpm.index:
            raise SystemExit(f"marker {gene} is not in the counts matrix")
        retained_mean = cpm.loc[gene, RETAINED].mean()
        excluded = cpm.loc[gene, EXCLUDED]
        rows.append({
            "gene": gene,
            "class": marker_class,
            "retained_mean_cpm": retained_mean,
            "fragment_2b_cpm": excluded,
            "fold_2b_vs_retained": excluded / retained_mean if retained_mean else float("nan"),
        })

    T = pd.DataFrame(rows)
    out = os.path.join(a.proj, "results")
    os.makedirs(out, exist_ok=True)
    T.to_csv(os.path.join(out, "excluded_fragment_markers.csv"), index=False)

    neuronal = T[T["class"] == "Sympathetic neuronal"]
    depletion = 1.0 / neuronal["fold_2b_vs_retained"]
    print(T.to_string(index=False))
    print(f"\nsympathetic neuronal markers: {len(neuronal)} of {len(neuronal)} depleted "
          f"{depletion.min():.0f} to {depletion.max():.0f}-fold in fragment {EXCLUDED}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
