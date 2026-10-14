#!/usr/bin/env python3
"""Receptor detection per cardiac cell type in HeartMap (Broad Single Cell Portal SCP3689).

Usage
-----
    python analysis/09_heart_celltypes.py --outdir results

Inputs
    Broad Single Cell Portal SCP3689, per-cell expression query (network access required)

Outputs
    results/heartmap_receptors_by_celltype.csv
"""
import argparse, json, os, sys, urllib.request
import numpy as np
import pandas as pd

BASE = "https://singlecell.broadinstitute.org/single_cell/api/v1"
ACC = "SCP3689"
GENES = ["EDNRB", "EDNRA"]
MIN_CELLS = 50


def fetch(gene):
    u = f"{BASE}/studies/{ACC}/expression/violin?genes={gene}"
    req = urllib.request.Request(u, headers={"Accept": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()

    V, axes = {}, set()
    for g in GENES:
        d = fetch(g)
        V[g] = d["values"]
        axes.add(d.get("y_axis_title", ""))
        print(f"{g}: {len(d['values'])} cell types, "
              f"{sum(len(v['y']) for v in d['values'].values()):,} cells, "
              f"y axis '{d.get('y_axis_title')}'")
    if not all("log" in x.lower() for x in axes):
        raise SystemExit(f"unexpected y axis {axes}; back-transformation assumption invalid")

    rows = []
    for ct in V["EDNRB"]:
        if ct not in V["EDNRA"]:
            continue
        b = np.asarray(V["EDNRB"][ct]["y"], dtype=float)
        aa = np.asarray(V["EDNRA"][ct]["y"], dtype=float)
        if len(b) < MIN_CELLS or len(aa) < MIN_CELLS:
            continue
        lb, la = float(np.expm1(b).mean()), float(np.expm1(aa).mean())
        rows.append({"cell_type": ct, "n_cells_subsampled": len(b),
                     "EDNRB_mean_linear": lb, "EDNRA_mean_linear": la,
                     "ratio_B_over_A": (lb / la if la > 0 else np.nan),
                     "EDNRB_pct_cells": 100 * float((b > 0).mean()),
                     "EDNRA_pct_cells": 100 * float((aa > 0).mean())})
    D = pd.DataFrame(rows).sort_values("ratio_B_over_A", ascending=False)

    os.makedirs(a.outdir, exist_ok=True)
    D.to_csv(os.path.join(a.outdir, "heartmap_receptors_by_celltype.csv"), index=False)

    print("\n" + D.to_string(index=False, float_format=lambda x: f"{x:9.2f}"))
    idx = D.set_index("cell_type")["ratio_B_over_A"]
    myo = [c for c in idx.index if "myocyte" in c]
    print("\ninternal positive control, cardiomyocytes should be EDNRA-dominant:")
    for c in myo:
        print(f"  {c:42} ratio {idx[c]:.3f} -> {'PASS' if idx[c] < 0.2 else 'FAIL'}")
    for c in ["cardiac neuron", "vascular associated smooth muscle cell",
              "cardiac endothelial cell"]:
        if c in idx:
            print(f"  {c:42} ratio {idx[c]:.2f}")
    if "cardiac neuron" in idx and "vascular associated smooth muscle cell" in idx:
        print(f"\nneuron vs vascular smooth muscle, same assay: "
              f"{idx['cardiac neuron'] / idx['vascular associated smooth muscle cell']:.1f}-fold "
              f"more EDNRB-weighted")
    print(f"\nwritten to {a.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
