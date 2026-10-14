#!/usr/bin/env python3
"""Receptor detection per cell type in two coronary artery single-cell atlases.

Usage
-----
    python analysis/08_coronary_celltypes.py \
        --files <tabula.h5ad> <plaque.h5ad> --outdir results

Inputs
    Tabula Sapiens Vasculature, CELLxGENE version 5d54a606-3cd9-4164-b862-a3c51e622790
    Plaque atlas coronary subset, CELLxGENE version ad9f52c0-054b-4dc5-944c-46c376a544cf

Outputs
    results/coronary_receptors_by_celltype.csv
"""
import argparse, os, sys
import numpy as np
import pandas as pd

GENES = ["EDNRB", "EDNRA"]
CONTEXT = ["ECE1", "EDN1", "PECAM1", "ACTA2", "MYH11"]


def load_counts(path, genes):
    """Return (obs, DataFrame of raw counts for `genes`, per-cell total raw counts, layer, missing).

    The CSR arrays are read with h5py rather than sliced through a backed AnnData: on these files
    `adata.raw[:, j].X` returns the ENTIRE matrix rather than one column, which silently produces a
    length-2.6e9 vector instead of one value per cell.
    """
    import anndata as ad
    import h5py
    from scipy.sparse import csr_matrix

    A = ad.read_h5ad(path, backed="r")
    obs = A.obs.copy()
    A.file.close()

    with h5py.File(path, "r") as f:
        layer = "raw/X" if "raw" in f and "X" in f["raw"] else "X"
        vgrp = "raw/var" if layer == "raw/X" and "feature_name" in f.get("raw/var", {}) else "var"
        fn = f[vgrp]["feature_name"]
        names = pd.Series(
            [s.decode() if isinstance(s, bytes) else str(s) for s in
             (fn["categories"][:][fn["codes"][:]] if isinstance(fn, h5py.Group) else fn[:])])
        idx = {g: int(np.where((names == g).values)[0][0]) for g in genes if (names == g).any()}
        missing = [g for g in genes if g not in idx]

        g = f[layer]
        enc = dict(g.attrs).get("encoding-type", "csr_matrix")
        if enc != "csr_matrix":
            raise SystemExit(f"{path}: {layer} is {enc}, expected csr_matrix")
        shape = tuple(dict(g.attrs)["shape"])
        M = csr_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=shape)

    tot = np.asarray(M.sum(axis=1)).ravel()
    cols = {gene: np.asarray(M[:, j].todense()).ravel() for gene, j in idx.items()}
    assert all(len(v) == shape[0] for v in cols.values()), "column slice length mismatch"
    return obs, pd.DataFrame(cols, index=obs.index), tot, layer, missing


def summarise(obs, cnt, tot, layer, label, tissue_filter):
    m = obs["tissue"].astype(str).str.contains(tissue_filter, case=False, na=False).values
    o, c, t = obs[m], cnt[m], tot[m]
    rows = []
    for ct, sub in o.groupby(o["cell_type"].astype(str), observed=True):
        k = o["cell_type"].astype(str).values == ct
        if k.sum() < 20:
            continue
        lib = t[k].sum()
        if lib <= 0:
            continue
        rec = {"dataset": label, "layer": layer, "cell_type": ct, "n_cells": int(k.sum()),
               "total_counts": float(lib)}
        for g in c.columns:
            s = float(c[g].values[k].sum())
            rec[f"{g}_cpm"] = 1e6 * s / lib
            rec[f"{g}_pct_cells"] = 100.0 * float((c[g].values[k] > 0).mean())
        if "EDNRB_cpm" in rec and "EDNRA_cpm" in rec:
            rec["ratio_B_over_A"] = (rec["EDNRB_cpm"] / rec["EDNRA_cpm"]
                                     if rec["EDNRA_cpm"] > 0 else np.inf)
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--tissue", default="coronary")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    labels = a.labels or [os.path.basename(f).replace(".h5ad", "") for f in a.files]

    out = []
    for f, lab in zip(a.files, labels):
        obs, cnt, tot, layer, missing = load_counts(f, GENES + CONTEXT)
        if missing:
            print(f"  {lab}: genes absent from var: {missing}")
        D = summarise(obs, cnt, tot, layer, lab, a.tissue)
        print(f"\n=== {lab} | layer={layer} | tissue filter '{a.tissue}' | "
              f"{D.n_cells.sum() if len(D) else 0} cells in {len(D)} cell types")
        if len(D):
            print(D[["cell_type", "n_cells", "EDNRB_cpm", "EDNRA_cpm", "ratio_B_over_A",
                     "EDNRB_pct_cells", "EDNRA_pct_cells"]]
                  .sort_values("ratio_B_over_A", ascending=False)
                  .to_string(index=False, float_format=lambda x: f"{x:9.2f}"))
        out.append(D)

    S = pd.concat(out, ignore_index=True)
    os.makedirs(a.outdir, exist_ok=True)
    S.to_csv(os.path.join(a.outdir, "coronary_receptors_by_celltype.csv"), index=False)

    vsmc = S[S.cell_type.str.contains("smooth muscle|myocyte", case=False, na=False)]
    endo = S[S.cell_type.str.contains("endothelial", case=False, na=False)]
    print("\n=== the premise, stated per compartment ===")
    for nm, df in [("smooth muscle", vsmc), ("endothelial", endo)]:
        if len(df):
            print(f"  {nm:14} EDNRB:EDNRA {df.ratio_B_over_A.min():.2f} to "
                  f"{df.ratio_B_over_A.max():.2f} across {len(df)} annotations "
                  f"({df.n_cells.sum()} cells)")
    print(f"\nwritten to {a.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
