#!/usr/bin/env python3
"""Receptor detection per cell class in single-nucleus sympathetic ganglion.

Usage
-----
    python analysis/07_ganglion_celltypes.py \
        --indir <directory containing GSM7728058_*.gz> --outdir results

Inputs
    GEO GSE241386, GSM7728058 (--indir)

Outputs
    results/ganglion_snrna_receptors.csv
    results/nucleus_assignments.csv
    results/cluster_marker_scores.csv
    results/cluster_marker_means.csv
    results/cluster_key_markers.csv
"""
import argparse, os, sys
import numpy as np
import pandas as pd

MARKERS = {
    "sympathetic neuron": ["TH", "DBH", "PHOX2B", "SLC18A2", "CHGA", "CHGB", "PRPH", "ELAVL4",
                           "SNAP25", "NPY"],
    "glial (satellite / Schwann)": ["S100B", "PLP1", "SOX10", "FABP7", "APOE", "MPZ", "PMP22",
                                    "MBP", "NGFR"],
    "endothelial": ["PECAM1", "CDH5", "VWF", "EGFL7"],
    "mural (SMC / pericyte)": ["ACTA2", "MYH11", "TAGLN", "RGS5", "NOTCH3", "PDGFRB"],
    "fibroblast": ["DCN", "COL1A1", "COL1A2", "PDGFRA", "LUM"],
    "myeloid / immune": ["PTPRC", "CD68", "CSF1R", "AIF1", "C1QA", "MRC1"],
    "adipocyte": ["ADIPOQ", "PLIN1", "LEP"],
    "lymphatic endothelial": ["PROX1", "PDPN", "FLT4"],
}
TARGETS = ["EDNRB", "EDNRA", "ECE1", "EDN1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--min-genes", type=int, default=500)
    ap.add_argument("--max-mito", type=float, default=10.0)
    ap.add_argument("--resolution", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import scanpy as sc
    import scipy.io as sio
    import gzip
    from scipy.sparse import csr_matrix
    import anndata as ad

    sc.settings.verbosity = 1
    np.random.seed(a.seed)

    with gzip.open(os.path.join(a.indir, "GSM7728058_matrix.mtx.gz"), "rb") as f:
        M = sio.mmread(f).T.tocsr()          # mtx is genes x cells; transpose to cells x genes
    bc = pd.read_csv(os.path.join(a.indir, "GSM7728058_barcodes.tsv.gz"), sep="\t", header=None)
    ft = pd.read_csv(os.path.join(a.indir, "GSM7728058_features.tsv.gz"), sep="\t", header=None)
    A = ad.AnnData(csr_matrix(M))
    A.obs_names = bc[0].astype(str).values
    A.var["gene_id"] = ft[0].astype(str).values
    A.var_names = pd.Index(ft[1].astype(str).values)
    A.var_names_make_unique()
    print(f"loaded {A.n_obs} nuclei x {A.n_vars} genes")

    A.var["mt"] = A.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(A, qc_vars=["mt"], inplace=True, percent_top=None, log1p=False)
    keep = (A.obs.n_genes_by_counts >= a.min_genes) & (A.obs.pct_counts_mt <= a.max_mito)
    print(f"QC: keeping {int(keep.sum())} of {A.n_obs} nuclei "
          f"(>= {a.min_genes} genes, <= {a.max_mito}% mito)")
    A = A[keep].copy()
    sc.pp.filter_genes(A, min_cells=3)

    A.layers["counts"] = A.X.copy()
    sc.pp.normalize_total(A, target_sum=1e4)
    sc.pp.log1p(A)
    sc.pp.highly_variable_genes(A, n_top_genes=2000)
    sc.pp.pca(A, n_comps=50, svd_solver="arpack", random_state=a.seed)
    sc.pp.neighbors(A, n_neighbors=15, random_state=a.seed)
    sc.tl.leiden(A, resolution=a.resolution, key_added="leiden",
                 flavor="igraph", n_iterations=2, directed=False, random_state=a.seed)
    nc = A.obs.leiden.nunique()
    print(f"clusters at resolution {a.resolution}: {nc}")

    # Marker score per cluster: mean log expression per cluster, then z across the
    # cluster means. Not z across cells, which puts the largest cluster at z ~ 0.
    present = {k: [g for g in v if g in A.var_names] for k, v in MARKERS.items()}
    clusters = sorted(A.obs.leiden.unique(), key=int)
    gene_means = {}
    allg = sorted({g for v in present.values() for g in v})
    sub = A[:, allg].X
    sub = np.asarray(sub.todense()) if hasattr(sub, "todense") else np.asarray(sub)
    GM = pd.DataFrame(sub, columns=allg, index=A.obs_names)
    GM["cl"] = A.obs.leiden.values
    cm = GM.groupby("cl", observed=True)[allg].mean().reindex(clusters)
    cz = (cm - cm.mean(0)) / (cm.std(0) + 1e-9)          # z across clusters, per gene
    MS = pd.DataFrame({k: cz[genes].mean(axis=1) for k, genes in present.items() if genes})
    MS.insert(0, "n_nuclei", [int((A.obs.leiden == c).sum()) for c in clusters])
    lab = MS.drop(columns=["n_nuclei"]).idxmax(axis=1)
    MS["label"] = lab
    MS["top_score"] = MS.drop(columns=["n_nuclei", "label"]).max(axis=1)
    MS["margin_over_2nd"] = MS.drop(columns=["n_nuclei", "label", "top_score"]).apply(
        lambda r: float(np.sort(r.values)[-1] - np.sort(r.values)[-2]), axis=1)
    MS["confident"] = (MS.top_score >= 0.5) & (MS.margin_over_2nd >= 0.25)
    A.obs["cell_type"] = A.obs.leiden.map(lab).astype(str)
    A.obs["label_confident"] = A.obs.leiden.map(MS.confident).values
    cm.to_csv(os.path.join(a.outdir, "cluster_marker_means.csv")) if os.path.isdir(a.outdir) \
        else None

    # receptor quantification from RAW counts, pseudobulk CPM within each label
    cnt = A.layers["counts"]
    tot = np.asarray(cnt.sum(axis=1)).ravel()
    tg = [g for g in TARGETS if g in A.var_names]
    vals = {}
    for g in tg:
        j = int(np.where(A.var_names == g)[0][0])
        vals[g] = np.asarray(cnt[:, j].todense()).ravel()
    out = []
    for ct, sub in A.obs.groupby("cell_type", observed=True):
        m = (A.obs.cell_type == ct).values
        lib = tot[m].sum()
        rec = {"cell_type": ct, "n_nuclei": int(m.sum()), "total_counts": float(lib)}
        for g in tg:
            rec[f"{g}_cpm"] = 1e6 * float(vals[g][m].sum()) / lib
            rec[f"{g}_pct_nuclei"] = 100 * float((vals[g][m] > 0).mean())
        if "EDNRB_cpm" in rec and "EDNRA_cpm" in rec:
            rec["ratio_B_over_A"] = (rec["EDNRB_cpm"] / rec["EDNRA_cpm"]
                                     if rec["EDNRA_cpm"] > 0 else np.nan)
        out.append(rec)
    R = pd.DataFrame(out).sort_values("ratio_B_over_A", ascending=False)

    os.makedirs(a.outdir, exist_ok=True)
    MS.to_csv(os.path.join(a.outdir, "cluster_marker_scores.csv"))
    cm.to_csv(os.path.join(a.outdir, "cluster_marker_means.csv"))
    KEY = [g for g in ["TH", "DBH", "PHOX2B", "S100B", "PLP1", "SOX10", "MPZ", "PECAM1", "VWF",
                       "ACTA2", "RGS5", "DCN", "COL1A1", "PTPRC", "CSF1R", "ADIPOQ"]
           if g in cm.columns]
    cm[KEY].to_csv(os.path.join(a.outdir, "cluster_key_markers.csv"))
    R.to_csv(os.path.join(a.outdir, "ganglion_snrna_receptors.csv"), index=False)
    A.obs[["leiden", "cell_type", "n_genes_by_counts", "pct_counts_mt"]].to_csv(
        os.path.join(a.outdir, "nucleus_assignments.csv"))

    print("\n=== marker z-score per cluster (check the labels against this) ===")
    print(MS.to_string(float_format=lambda x: f"{x:+6.2f}"))
    print("\n=== EDNRB and EDNRA by annotated cell type (raw-count pseudobulk CPM) ===")
    cols = ["cell_type", "n_nuclei"] + [f"{g}_cpm" for g in tg] + \
           ["ratio_B_over_A", "EDNRB_pct_nuclei", "EDNRA_pct_nuclei"]
    print(R[cols].to_string(index=False, float_format=lambda x: f"{x:9.2f}"))
    print(f"\nwritten to {a.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
