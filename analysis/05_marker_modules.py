#!/usr/bin/env python3
"""Marker module scores in the three retained fragments.

Usage
-----
    python analysis/05_marker_modules.py --proj .

Inputs
    data/vst_normalized.csv

Outputs
    results/marker_module_scores.csv
    results/marker_module_summary.csv
"""
import argparse
import os

import pandas as pd

FRAGMENTS = ["1", "2a", "3"]

MODULES = {
    "Neuron":        ["TH", "DBH", "NPY", "PHOX2B", "PRPH", "SNAP25", "STMN2"],
    "Glia":          ["S100B", "PLP1", "SOX10", "MPZ"],
    "Mesenchymal":   ["ACTA2", "MYH11", "DCN", "COL1A1"],
    "Immune":        ["PTPRC", "PECAM1", "CD3E"],
    "Cardiomyocyte": ["TNNT2", "MYH7", "NPPA", "ACTC1"],
    "Hepatocyte":    ["ALB", "APOA1", "TF", "SERPINA1"],
}
NEGATIVE_MODULES = {"Cardiomyocyte", "Hepatocyte"}


def module_scores(vst):
    V = vst.drop_duplicates("gene_name").set_index("gene_name")
    rows = []
    for module, genes in MODULES.items():
        present = [g for g in genes if g in V.index]
        if not present:
            raise SystemExit(f"module {module} has no genes in the VST matrix")
        for fragment in FRAGMENTS:
            rows.append({
                "module": module,
                "fragment": fragment,
                "score_mean_vst": V.loc[present, fragment].mean(),
                "n_genes_used": len(present),
                "n_genes_listed": len(genes),
                "is_negative_control": module in NEGATIVE_MODULES,
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", default=".")
    a = ap.parse_args()

    vst = pd.read_csv(os.path.join(a.proj, "data", "vst_normalized.csv"))
    per_fragment = module_scores(vst)

    summary = (per_fragment.groupby("module")
               .agg(mean_score=("score_mean_vst", "mean"),
                    min_score=("score_mean_vst", "min"),
                    max_score=("score_mean_vst", "max"),
                    n_genes_used=("n_genes_used", "first"),
                    n_genes_listed=("n_genes_listed", "first"),
                    is_negative_control=("is_negative_control", "first"))
               .sort_values("mean_score", ascending=False)
               .reset_index())

    out = os.path.join(a.proj, "results")
    os.makedirs(out, exist_ok=True)
    per_fragment.to_csv(os.path.join(out, "marker_module_scores.csv"), index=False)
    summary.to_csv(os.path.join(out, "marker_module_summary.csv"), index=False)

    print(summary.to_string(index=False))
    top = summary.iloc[0]["module"]
    print(f"\nhighest scoring module: {top}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
