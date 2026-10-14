#!/usr/bin/env python3
"""Twenty gene sets against abundance- and length-matched null sets.

Usage
-----
    python analysis/02_gene_set_analysis.py --proj . \
        --lengths data/transcript_lengths_refseq_select.csv --nperm 5000

Inputs
    results/percentile_ranks_universe.csv
    data/transcript_lengths_refseq_select.csv

Outputs
    results/length_bias_by_decile.csv
    results/length_matched_sets.csv
"""
import argparse, hashlib, json, os, sys
import numpy as np
import pandas as pd
from scipy import stats as st

# All twenty sets, scored on one common footing and grouped by the question each one
# answers. The twentieth set is defined immediately below.
SYNAPTIC_ADHESION = ["NLGN1", "NRXN1", "NRXN2", "NRXN3", "SHANK2", "LIN7C",
                     "GRIN2B", "SLITRK5", "PTPRD", "SRCIN1"]

SETS = {
    # Is there neuroinflammation? (the project's original hypothesis)
    "IL-1 and inflammasome": ["IL1A", "IL1B", "IL1R1", "IL1RAP", "NLRP1", "NLRP3", "PYCARD",
                              "CASP1", "CASP4", "IL18", "AIM2", "NFKB1"],
    "Cytokine and chemokine signalling": ["TNF", "IL6", "CXCL8", "CCL2", "CCL3", "CXCL12",
                                          "IL10", "TGFB1", "CSF1", "IL1RN"],
    "Myeloid and leukocyte markers": ["PTPRC", "AIF1", "CD68", "CSF1R", "TYROBP", "ITGAM",
                                      "C1QA", "C1QB", "C1QC", "MRC1", "TREM2", "FCER1G", "LYZ"],
    # Is neuronal activity elevated?
    "Immediate-early (neuronal activity)": ["FOS", "FOSB", "JUNB", "EGR1", "EGR2", "EGR3",
                                            "NR4A1", "NR4A2", "NR4A3", "ARC", "NPAS4"],
    "Catecholamine machinery": ["TH", "DBH", "DDC", "PNMT", "SLC6A2", "SLC18A2", "CHGA",
                                "CHGB", "NPY"],
    "Ganglionic nicotinic receptors": ["CHRNA3", "CHRNB4", "CHRNA7", "CHRNB2", "CHRNA5"],
    "Synaptic vesicle machinery": ["SYN1", "SYT1", "SNAP25", "VAMP2", "RAB3A", "CPLX1", "CPLX2",
                                   "STX1A", "SV2A", "SYP"],
    "Active zone": ["RIMS1", "RIMS2", "UNC13A", "UNC13B", "BSN", "PCLO", "ERC2", "CACNA1B"],
    "Excitability": ["HCN1", "HCN2", "KCNQ2", "KCNQ3", "SCN3A", "SCN9A", "KCNN3", "CACNA1A"],
    # Is there injury, sprouting, or remodelling?
    "Regeneration-associated": ["ATF3", "JUN", "SOX11", "GAP43", "SPRR1A", "GADD45A", "KLF6",
                                "STAT3", "VGF", "TIMP1"],
    "Neurotrophin signalling": ["NGF", "NGFR", "NTRK1", "NTRK2", "NTRK3", "BDNF", "GFRA1",
                                "RET", "NTF3"],
    "Anterograde axonal transport": ["KIF1A", "KIF1B", "KIF5A", "KIF5B", "KIF5C", "KLC1",
                                     "KIF3A", "DYNC1H1"],
    "Neurexin/neuroligin family": ["NRXN1", "NRXN2", "NRXN3", "NLGN1", "NLGN2", "NLGN3",
                                   "NLGN4X", "CNTNAP2", "CNTNAP4"],
    # Neuronal content controls: identity markers outside the adhesion set.
    "Pan-neuronal identity": ["UCHL1", "ELAVL4", "TUBB3", "SNAP25", "NEFL", "STMN2", "MAP2",
                              "RBFOX3", "ENO2", "INA", "PRPH", "PHOX2B"],
    # Cellular composition — could a compartment difference explain the receptor result?
    "Endothelial": ["PECAM1", "CDH5", "VWF", "CLDN5", "EGFL7", "ERG", "TIE1"],
    "Mural and smooth muscle": ["ACTA2", "MYH11", "TAGLN", "PDGFRB", "RGS5", "NOTCH3"],
    "Satellite glia": ["S100B", "GFAP", "SOX10", "PLP1", "APOE", "MPZ", "PMP22", "FABP7"],
    # Oxidative stress, and the endothelin axis itself
    "Antioxidant response": ["NFE2L2", "NQO1", "SOD1", "SOD2", "CAT", "GPX1", "HMOX1", "TXN",
                             "PRDX1"],
    "Endothelin axis": ["EDN1", "EDN2", "EDN3", "ECE1", "ECE2", "EDNRA", "EDNRB"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", required=True)
    ap.add_argument("--lengths", required=True)
    ap.add_argument("--nperm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--family-size", type=int, default=30, dest="family_size",
                    help="Denominator for the Bonferroni threshold. Defaults to 30, the "
                         "total number of distinct gene sets tested across the analysis "
                         "(0.05/30 = 0.0017).")
    a = ap.parse_args()

    mo = os.path.join(a.proj, "results")
    R = pd.read_csv(os.path.join(mo, "percentile_ranks_universe.csv"), index_col=0)
    sym = R.pop("gene_name")
    sets = dict(SETS)
    sets["Synaptic adhesion"] = SYNAPTIC_ADHESION

    pat = [c for c in R.columns if c in ("1", "2a", "3")]
    hc = [c for c in R.columns if "healthy" in c]
    dc = [c for c in R.columns if "diseased" in c]

    def zscore(ref):
        s = R[pat].mean(axis=1) - R[ref].mean(axis=1)
        m = np.median(s)
        return (s - m) / (np.median(np.abs(s - m)) * 1.4826)

    z, zd = zscore(hc), zscore(dc)
    glen = sym.map(pd.read_csv(a.lengths, index_col=0).iloc[:, 0])
    ok = glen.notna().values
    abund = R[hc].mean(axis=1)

    rho, pv = st.spearmanr(np.log2(glen[ok]), z.values[ok])
    dec = pd.DataFrame({"len": glen[ok].values, "z": z.values[ok],
                        "zd": zd.values[ok]})
    dec["decile"] = pd.qcut(dec["len"], 10, labels=False, duplicates="drop")
    D = dec.groupby("decile").agg(n=("z", "size"), median_len=("len", "median"),
                                  median_z_vs_healthy=("z", "median"),
                                  median_z_vs_disease=("zd", "median")).reset_index()

    aq = pd.qcut(abund[ok], 5, labels=False, duplicates="drop")
    lq = pd.qcut(np.log2(glen[ok]), 5, labels=False, duplicates="drop")
    cell = pd.Series(aq.values * 5 + lq.values, index=np.where(ok)[0])
    pool = {c: cell.index[(cell == c).values].to_numpy() for c in sorted(cell.unique())}

    def joint(genes, zz, tag):
        # Seed per (set, contrast) so a set's draws do not depend on set order.
        # hashlib, not hash(): str hashing is randomised per process.
        key = f"{a.seed}|{tag}|{zz.name}".encode()
        rng = np.random.default_rng(
            int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))
        idx = np.array([i for i in np.where(sym.isin(genes).values)[0] if i in cell.index])
        if len(idx) < 3:
            return None
        want = cell.loc[idx].value_counts()
        if any(len(pool[c]) < n for c, n in want.items()):
            return None
        obs = float(np.median(zz.iloc[idx]))
        null = np.array([np.median(zz.iloc[np.concatenate(
            [rng.choice(pool[c], size=n, replace=False) for c, n in want.items()])])
            for _ in range(a.nperm)])
        return dict(n=len(idx), mean_log2_len=float(np.mean(np.log2(glen.iloc[idx]))),
                    obs_z=obs, null_p5=float(np.percentile(null, 5)),
                    null_p95=float(np.percentile(null, 95)), p=float((null >= obs).mean()))

    z.name, zd.name = "vs_healthy", "vs_disease"
    thresh = 0.05 / max(1, a.family_size or len(sets))
    rows = []
    for name, genes in sets.items():
        rh, rd = joint(genes, z, name), joint(genes, zd, name)
        if rh is None:
            continue
        rows.append({"gene_set": name, **{k + "_vs_healthy" if k in ("obs_z", "null_p5",
                     "null_p95", "p") else k: v for k, v in rh.items()},
                     "obs_z_vs_disease": rd["obs_z"], "p_vs_disease": rd["p"],
                     "bonferroni_threshold": thresh,
                     "survives_length_matched_bonferroni": bool(
                         rh["p"] < thresh and rd["p"] < thresh)})
    S = pd.DataFrame(rows).sort_values("obs_z_vs_healthy", ascending=False)

    out = os.path.join(a.proj, "results", ".")
    os.makedirs(out, exist_ok=True)
    D.to_csv(os.path.join(out, "length_bias_by_decile.csv"), index=False)
    S.to_csv(os.path.join(out, "length_matched_sets.csv"), index=False)

    print(f"lengths for {int(ok.sum())} of {len(sym)} universe rows | "
          f"universe median {int(np.median(glen[ok]))} nt")
    print(f"\nglobal Spearman rho(log2 length, rank-shift z) = {rho:+.3f}  p = {pv:.2e}")
    print("\nmedian rank-shift z by transcript-length decile:")
    print(D[["decile", "median_len", "median_z_vs_healthy"]].to_string(
        index=False, float_format=lambda x: f"{x:9.2f}"))
    print("\nsets vs the JOINT abundance x length matched null:")
    print(S[["gene_set", "n", "mean_log2_len", "obs_z_vs_healthy", "null_p95_vs_healthy",
             "p_vs_healthy", "p_vs_disease", "survives_length_matched_bonferroni"]]
          .to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
    print(f"\nBonferroni threshold across {a.family_size or len(sets)} sets: {thresh:.5f}")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
