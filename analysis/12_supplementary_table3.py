#!/usr/bin/env python3
"""Per-dataset receptor values, detection counts and fold spread by cell type.

Usage
-----
    python analysis/12_supplementary_table3.py --proj .

Inputs
    results/receptor_by_celltype_ganglion.csv
    results/receptor_by_celltype_atlases.csv

Outputs
    results/Supplementary_Table_3_receptor_by_celltype.csv
"""
import argparse, csv, os
import numpy as np

LINEAGE = {
    "Endothelial": ["endothelial cell (general)", "arterial endothelial cell",
                    "vein endothelial cell", "capillary endothelial cell",
                    "lymphatic endothelial cell", "endocardial cell", "endothelial"],
    "Immune": ["monocyte", "macrophage", "myeloid / immune", "T cell",
               "dendritic cell", "plasma cell", "lymphocyte (unspecified)", "mast cell"],
    "Stromal": ["fibroblast", "adipocyte"],
    "Neural / glial": ["glial (satellite / Schwann)", "cardiac neuron",
                       "sympathetic neuron"],
    "Other": ["mesothelial cell", "unannotated"],
    "Mural / contractile": ["pericyte", "smooth muscle cell", "mural (SMC / pericyte)",
                            "ventricular cardiomyocyte", "atrial cardiomyocyte"],
}
NICE = {"mural (SMC / pericyte)": "Mural (SMC / pericyte)",
        "sympathetic neuron": "Sympathetic neuron",
        "glial (satellite / Schwann)": "Satellite glia / Schwann",
        "myeloid / immune": "Myeloid / immune", "endothelial": "Endothelial (ganglion)",
        "endothelial cell (general)": "Endothelial cell",
        "lymphocyte (unspecified)": "Lymphocyte", "unannotated": "Unannotated"}
nice = lambda u: NICE.get(u, u[:1].upper() + u[1:])
DS = {"GanglionSN": ("Ganglion", "GSE241386, full annotated count"),
      "HeartMap": ("Heart", "HeartMap, subsampled count"),
      "PlaqueAtlas": ("Coronary", "Plaque atlas coronary subset, full annotated count"),
      "TabulaSapiens": ("Vasculature", "Tabula Sapiens vasculature, full annotated count")}
ORDER = ["GanglionSN", "HeartMap", "PlaqueAtlas", "TabulaSapiens"]


def read(path, merged):
    """Read a deposited cell-type table. `merged` names the canonical-type column,
    or None when the native label is already the unit of analysis."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} is empty")
    nc = "n_cells_subsampled" if "n_cells_subsampled" in rows[0] else "n"
    def num(v):
        if v in (None, ""):
            return float("nan")
        f = float(v)
        return int(f) if f.is_integer() else f
    out = []
    for r in rows:
        out.append(dict(cohort=r["cohort"], native=r["cell_type"],
                        unit=r[merged] if merged else r["cell_type"],
                        n=num(r[nc]), ratio=num(r["ratio_B_over_A_pseudobulk_cpm"]),
                        bpct=num(r["pct_cells_detecting_EDNRB"]),
                        apct=num(r["pct_cells_detecting_EDNRA"]),
                        nb=num(r["n_detecting_EDNRB"]), na=num(r["n_detecting_EDNRA"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", default=".")
    ap.add_argument("--out", default=None,
                    help="defaults to <proj>/results/"
                         "Supplementary_Table_3_receptor_by_celltype.csv")
    a = ap.parse_args()
    res = os.path.join(a.proj, "results")
    a.out = a.out or os.path.join(res, "Supplementary_Table_3_receptor_by_celltype.csv")
    rows = (read(os.path.join(res, "receptor_by_celltype_ganglion.csv"), None)
            + read(os.path.join(res, "receptor_by_celltype_atlases.csv"), "merged_type"))

    by = {}
    for r in rows:
        by.setdefault(r["unit"], {}).setdefault(r["cohort"], []).append(r)

    fields = ["lineage_group", "cell_type", "dataset", "dataset_detail",
              "source_label_in_that_dataset", "n_cells", "n_cells_note",
              "pct_detecting_EDNRB", "pct_detecting_EDNRA",
              "n_detecting_EDNRB", "n_detecting_EDNRA",
              "ratio_EDNRB_over_EDNRA", "call",
              "n_datasets_for_this_cell_type", "median_ratio_across_datasets",
              "min_ratio", "max_ratio", "fold_spread_across_datasets"]
    out = []
    for grp, members in LINEAGE.items():
        ms = [u for u in members if u in by]
        ms.sort(key=lambda u: -float(np.median(
            [float(np.median([x["ratio"] for x in v])) for v in by[u].values()])))
        for u in ms:
            per = {c: float(np.median([x["ratio"] for x in v]))
                   for c, v in by[u].items()}
            med = float(np.median(list(per.values())))
            lo, hi = min(per.values()), max(per.values())
            for coh in ORDER:
                if coh not in by[u]:
                    continue
                for r in by[u][coh]:
                    out.append({
                        "lineage_group": grp, "cell_type": nice(u),
                        "dataset": DS[coh][0], "dataset_detail": DS[coh][1],
                        "source_label_in_that_dataset": r["native"],
                        "n_cells": r["n"],
                        "n_cells_note": ("subsampled" if coh == "HeartMap"
                                         else "annotated count"),
                        "pct_detecting_EDNRB": round(r["bpct"], 4),
                        "pct_detecting_EDNRA": round(r["apct"], 4),
                        "n_detecting_EDNRB": r["nb"], "n_detecting_EDNRA": r["na"],
                        "ratio_EDNRB_over_EDNRA": round(r["ratio"], 4),
                        "call": "EDNRB-weighted" if r["ratio"] > 1 else "EDNRA-weighted",
                        "n_datasets_for_this_cell_type": len(per),
                        "median_ratio_across_datasets": round(med, 4),
                        "min_ratio": round(lo, 4), "max_ratio": round(hi, 4),
                        "fold_spread_across_datasets": round(hi / lo, 2),
                    })
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(out)

    multi = [r for r in out if r["n_datasets_for_this_cell_type"] > 1]
    sp = sorted({(r["cell_type"], r["fold_spread_across_datasets"]) for r in multi},
                key=lambda t: -t[1])
    print("wrote %s  (%d rows, %d cell types)" % (a.out, len(out), len(by)))
    print("cell types in 1 dataset only: %d of %d"
          % (sum(1 for u in by if len(by[u]) == 1), len(by)))
    print("fold spread where >1 dataset:")
    for n, s in sp:
        print("   %-26s %.1fx" % (n, s))


if __name__ == "__main__":
    main()
