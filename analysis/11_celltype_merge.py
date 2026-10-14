#!/usr/bin/env python3
"""Canonical cell-type merge, Fisher exact test, detection-floor sensitivity.

Usage
-----
    python analysis/11_celltype_merge.py --proj .

Inputs
    results/receptor_by_celltype_atlases.csv

Outputs
    results/celltype_merge.csv
    2x2 table, Fisher P and floor sensitivity on stdout
"""
import argparse, collections, csv, os
from math import comb


MURAL = "contractile/mural"

# raw atlas label -> canonical cell type
CANON = {
    # --- contractile / mural ---
    "regular atrial cardiac myocyte": "atrial cardiomyocyte",
    "regular ventricular cardiac myocyte": "ventricular cardiomyocyte",
    "vascular associated smooth muscle cell": "smooth muscle cell",
    "smooth muscle cell": "smooth muscle cell",
    "pericyte": "pericyte",
    # --- non-mural ---
    "mesothelial cell of epicardium": "mesothelial cell",
    "mast cell": "mast cell",
    "lymphocyte": "lymphocyte (unspecified)",
    "T cell": "T cell",
    "CD8-positive, alpha-beta T cell": "T cell",
    "adipocyte": "adipocyte",
    "cell": "unannotated",
    "cardiac neuron": "cardiac neuron",
    "fibroblast of cardiac tissue": "fibroblast",
    "fibroblast": "fibroblast",
    "macrophage": "macrophage",
    "monocyte": "monocyte",
    "plasma cell": "plasma cell",
    "dendritic cell": "dendritic cell",
    "endothelial cell of lymphatic vessel": "lymphatic endothelial cell",
    "endocardial cell": "endocardial cell",
    "cardiac endothelial cell": "endothelial cell (general)",
    "endothelial cell": "endothelial cell (general)",
    "capillary endothelial cell": "capillary endothelial cell",
    "vein endothelial cell": "vein endothelial cell",
    "endothelial cell of artery": "arterial endothelial cell",
}



def read_atlas_table(path):
    """Rows as tuples in file column order, numeric fields coerced to float."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    def cast(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return v
    return [tuple(cast(v) for v in r) for r in body]


def fisher(a, b, c, d):
    """Two-sided Fisher exact on [[a,b],[c,d]]."""
    n = a + b + c + d
    p = lambda a, b, c, d: comb(a + b, a) * comb(c + d, c) / comb(n, a + c)
    obs = p(a, b, c, d)
    r1, r2, c1 = a + b, c + d, a + c
    return sum(q for i in range(max(0, c1 - r2), min(r1, c1) + 1)
               for q in [p(i, r1 - i, c1 - i, r2 - c1 + i)] if q <= obs * (1 + 1e-9))


def detected_cells(row):
    """Cells detecting the SCARCER receptor, the quantity a floor should act on."""
    n, b_pct, a_pct = row[2], row[4], row[5]
    return min(n * b_pct / 100.0, n * a_pct / 100.0)


def table(types):
    mur = [t for t in types if t["is_mural"]]
    oth = [t for t in types if not t["is_mural"]]
    a = sum(1 for t in mur if t["call"] == "A")
    b = len(mur) - a
    c = sum(1 for t in oth if t["call"] == "A")
    d = len(oth) - c
    return a, b, c, d, mur, oth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", default=".")
    ap.add_argument("--atlases", default=None,
                    help="atlas cell-type table; defaults to "
                         "<proj>/results/receptor_by_celltype_atlases.csv")
    ap.add_argument("--outdir", default=None,
                    help="defaults to <proj>/results")
    ap.add_argument("--floors", default="0,5,10", help="detected-cell floors to test")
    a = ap.parse_args()

    src = a.atlases or os.path.join(a.proj, "results",
                                    "receptor_by_celltype_atlases.csv")
    a.outdir = a.outdir or os.path.join(a.proj, "results")
    raw = read_atlas_table(src)
    unknown = sorted({r[1] for r in raw} - set(CANON))
    if unknown:
        raise SystemExit("unmapped atlas labels, refusing to guess: %s" % unknown)

    groups = collections.defaultdict(list)
    for r in raw:
        groups[CANON[r[1]]].append(r)

    types = []
    for name, rows in sorted(groups.items()):
        calls = [1 if r[3] > 1 else 0 for r in rows]
        if sum(calls) * 2 == len(calls):
            call = "TIE"
        else:
            call = "B" if sum(calls) * 2 > len(calls) else "A"
        types.append({
            "merged_type": name,
            "is_mural": rows[0][7] == MURAL,
            "call": call,
            "n_entries": len(rows),
            "sources": "; ".join("%s:%s" % (r[0], r[1]) for r in rows),
            "ratios": "; ".join("%.4g" % r[3] for r in rows),
            "min_detected_cells": min(detected_cells(r) for r in rows),
        })

    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, "celltype_merge.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(types[0].keys()))
        w.writeheader()
        for t in types:
            w.writerow(t)

    print("%d raw entries -> %d canonical cell types" % (len(raw), len(types)))
    for label, sel in (("MURAL", True), ("NON-MURAL", False)):
        sub = [t for t in types if t["is_mural"] is sel]
        print("\n%s (%d)" % (label, len(sub)))
        for t in sub:
            print("  %-28s %s  n=%d  min_cells=%4.0f  ratios=%s"
                  % (t["merged_type"], t["call"], t["n_entries"],
                     t["min_detected_cells"], t["ratios"]))

    print("\n--- primary test (all entries) ---")
    A, B, C, D, mur, oth = table(types)
    print("mural EDNRA-weighted %d of %d ; non-mural EDNRB-weighted %d of %d"
          % (A, len(mur), D, len(oth)))
    print("exceptions among non-mural: %s"
          % [t["merged_type"] for t in oth if t["call"] != "B"])
    print("Fisher exact P = %.4g" % fisher(A, B, C, D))

    print("\n--- detection-floor sensitivity (floor on the scarcer receptor) ---")
    for fl in [float(x) for x in a.floors.split(",")]:
        kept = [t for t in types if t["min_detected_cells"] >= fl]
        A, B, C, D, mur, oth = table(kept)
        print("floor >= %-4.0f cells: %2d types (%d mural, %d non-mural)  "
              "%d/%d mural EDNRA, %d/%d non-mural EDNRB  P = %.4g"
              % (fl, len(kept), len(mur), len(oth), A, len(mur), D, len(oth),
                 fisher(A, B, C, D)))
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
