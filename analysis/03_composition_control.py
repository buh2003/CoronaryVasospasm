#!/usr/bin/env python3
"""Receptor rank among abundance- and composition-matched control genes.

Usage
-----
    python analysis/03_composition_control.py --proj . \
        --lengths data/transcript_lengths_refseq_select.csv \
        --scdata <directory containing GSM7728058_*>

Inputs
    results/percentile_ranks_universe.csv
    data/transcript_lengths_refseq_select.csv
    GEO GSE241386 matrix (--scdata)

Outputs
    results/composition_adjusted_position.csv
    results/celltype_bulk_snrna_offset.csv
    results/patient_control_gap_null.csv
    results/calibration_gates.csv
    results/provenance.json
"""

import argparse
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import binomtest, rankdata

GATE_A = ["TH", "DBH", "SLC6A2", "SLC18A2", "PHOX2B", "SNAP25", "NEFL", "NEFM",
          "ACTB", "GAPDH", "TUBB", "RPL13A", "PPIA", "B2M", "PGK1", "TBP"]
GATE_B = ["ACTB", "GAPDH", "TUBB", "PPIA", "B2M", "PGK1", "TBP", "VCP", "PSMB2",
          "RAB7A", "VPS29", "CHMP2A", "EMC7", "REEP5", "SNRPD3"]
GATE_MAX_ABS_Z = 2.0


def snrna(scdata, assignments):
    """Return (per-cell-type CPM, tissue-level CPM) from the single-nucleus reference."""
    with gzip.open(os.path.join(scdata, "GSM7728058_matrix.mtx.gz"), "rb") as fh:
        X = sio.mmread(fh).tocsc()
    feat = pd.read_csv(os.path.join(scdata, "GSM7728058_features.tsv.gz"), sep="\t",
                       header=None)
    barc = pd.read_csv(os.path.join(scdata, "GSM7728058_barcodes.tsv.gz"), sep="\t",
                       header=None)
    lab = pd.read_csv(assignments).rename(columns={"Unnamed: 0": "barcode"})
    lab["barcode"] = lab["barcode"].astype(str)
    bcs = barc[0].astype(str).values
    keep = pd.Index(bcs).isin(set(lab["barcode"]))
    X = X[:, np.where(keep)[0]].tocsr()
    labels = lab.set_index("barcode").loc[bcs[keep], "cell_type"].values
    g = feat[1].astype(str).values
    cols = {}
    for t in sorted(pd.unique(labels)):
        s = np.asarray(X[:, np.where(labels == t)[0]].sum(axis=1)).ravel()
        cols[t] = s / s.sum() * 1e6
    S = pd.DataFrame(cols, index=g).groupby(level=0).sum()
    tot = np.asarray(X.sum(axis=1)).ravel()
    tis = pd.Series(tot / tot.sum() * 1e6, index=g).groupby(level=0).sum()
    return S, tis, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", default=".")
    ap.add_argument("--scdata", required=True)
    ap.add_argument("--lengths", required=True)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--targets", nargs="+", default=["EDNRB", "EDNRA"])
    ap.add_argument("--n-matched", type=int, default=300)
    ap.add_argument("--ab-tol", type=float, default=0.5)
    ap.add_argument("--len-tol", type=float, default=0.5)
    ap.add_argument("--n-null", type=int, default=400)
    ap.add_argument("--null-seed", type=int, default=0)
    a = ap.parse_args()
    out = a.outdir or os.path.join(a.proj, "results", ".")
    os.makedirs(out, exist_ok=True)

    # ---- GUARD 1: within-sample percentile ranks on one fixed universe, nothing else.
    R = pd.read_csv(os.path.join(a.proj, "results", "percentile_ranks_universe.csv"), index_col=0)
    sym = R.pop("gene_name")
    pat = [c for c in R.columns if c in ("1", "2a", "3")]
    ctl = [c for c in R.columns if "healthy" in c or "diseased" in c]
    hc = [c for c in R.columns if "healthy" in c]
    dc = [c for c in R.columns if "diseased" in c]
    assert len(pat) == 3 and len(ctl) == 6, (pat, ctl)

    def zrank(tgt, ref):
        sh = R[tgt].mean(axis=1) - R[ref].mean(axis=1)
        med = np.median(sh)
        return (sh - med) / (np.median(np.abs(sh - med)) * 1.4826)

    # ---- GUARD 2: gates first, exit non-zero if the marker-free gate fails.
    gate_rows = []
    for nm, gl in [("A_established", GATE_A), ("B_marker_free", GATE_B)]:
        for lab, zz in [("vs_healthy", zrank(pat, hc)), ("vs_disease", zrank(pat, dc))]:
            idx = np.where(sym.isin([g for g in gl if g in set(sym.values)]).values)[0]
            med = float(np.median(np.abs(zz.iloc[idx]))) if len(idx) else np.nan
            gate_rows.append({"gate": nm, "contrast": lab, "n_genes_found": len(idx),
                              "median_abs_z": med, "pass": bool(med < GATE_MAX_ABS_Z)})
    GT = pd.DataFrame(gate_rows)
    GT.to_csv(os.path.join(out, "calibration_gates.csv"), index=False)
    print(GT.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    if not bool(GT.loc[GT.gate == "B_marker_free", "pass"].all()):
        print("\nMARKER-FREE CALIBRATION GATE FAILED. Stopping.", file=sys.stderr)
        return 2
    print("\ncalibration: PASS\n")

    S, tis, labels = snrna(a.scdata, os.path.join(
        a.proj, "results", "nucleus_assignments.csv"))
    Pr = S.div(S.sum(axis=1), axis=0)
    glen = sym.map(pd.read_csv(a.lengths, index_col=0).iloc[:, 0])

    shared = [g for g in R.index if sym.loc[g] in set(Pr.index)]
    sh_idx = pd.Index(shared)
    gsym = sym.loc[shared].values
    prof = Pr.loc[gsym].values
    tcpm = np.log2(tis.loc[gsym].values + 1)
    glog = np.log2(glen[sh_idx].values)

    def match(target, use_length):
        """GUARD 5: receptors excluded from each other's matched sets."""
        p0 = Pr.loc[target].values
        a0 = np.log2(tis[target] + 1)
        ti = R.index[sym == target][0]
        l0 = np.log2(glen[ti])
        d = np.abs(prof - p0).sum(axis=1)
        ok = (np.abs(tcpm - a0) <= a.ab_tol) & ~np.isin(gsym, a.targets)
        if use_length:
            ok &= (np.abs(glog - l0) <= a.len_tol) & np.isfinite(glog)
        cand = np.where(ok)[0]
        if len(cand) < 30:
            return None
        return sh_idx[cand[np.argsort(d[cand])[:a.n_matched]]], ti

    # Null control for the patient-versus-control gap: random genes of matched
    # abundance and length through the identical procedure.
    rng = np.random.default_rng(a.null_seed)
    Rv = R.loc[sh_idx].values
    ccols = list(R.columns)
    p_i = [ccols.index(c) for c in pat]
    c_i = [ccols.index(c) for c in ctl]
    pool = np.where((tcpm > np.log2(5)) & np.isfinite(glog))[0]
    probe = rng.choice(pool, min(a.n_null, len(pool)), replace=False)
    nb = []
    for gi in probe:
        d = np.abs(prof - prof[gi]).sum(axis=1)
        ok = ((np.abs(tcpm - tcpm[gi]) <= a.ab_tol) & (np.abs(glog - glog[gi]) <= a.len_tol)
              & np.isfinite(glog) & (np.arange(len(gsym)) != gi))
        c2 = np.where(ok)[0]
        if len(c2) < 50:
            continue
        mi = c2[np.argsort(d[c2])[:a.n_matched]]
        pos = (Rv[mi] < Rv[gi][None, :]).mean(axis=0) * 100
        nb.append({"gene": gsym[gi], "patient_mean": pos[p_i].mean(),
                   "control_mean": pos[c_i].mean(),
                   "gap": pos[p_i].mean() - pos[c_i].mean()})
    NB = pd.DataFrame(nb)
    NB.to_csv(os.path.join(out, "patient_control_gap_null.csv"), index=False)

    rows = []
    for target in a.targets:
        for use_length in (False, True):
            m = match(target, use_length)
            if m is None:
                continue
            ids, ti = m
            for c in pat + ctl:
                obs = R.loc[ti, c]
                mv = R.loc[ids, c]
                rows.append({
                    "gene": target, "sample": c,
                    "group": "patient" if c in pat else "control",
                    "matching": "profile+abundance+length" if use_length
                                else "profile+abundance",
                    "n_matched": len(ids), "observed_rank": obs,
                    "matched_median": float(mv.median()),
                    "adj_position": 100.0 * float((mv < obs).mean())})
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(out, "composition_adjusted_position.csv"), index=False)

    # ---- cell-class bulk-versus-snRNA offset ladder (attribution by analogy)
    snr = pd.Series(rankdata(tcpm) / len(tcpm) * 100, index=sh_idx)
    dom = pd.Series(prof.argmax(axis=1), index=sh_idx).map(dict(enumerate(Pr.columns)))
    domfrac = pd.Series(prof.max(axis=1), index=sh_idx)
    sp = R.loc[sh_idx, pat].mean(axis=1) - snr
    sr = R.loc[sh_idx, ctl].mean(axis=1) - snr
    bp, br = sp.median(), sr.median()
    strong = domfrac >= 0.40
    cl = []
    for t in Pr.columns:
        msk = strong & (dom == t)
        if msk.sum() < 20:
            continue
        cl.append({"dominant": t, "n": int(msk.sum()),
                   "patient_corr": float(sp[msk].median() - bp),
                   "control_corr": float(sr[msk].median() - br)})
    CL = pd.DataFrame(cl).sort_values("patient_corr", ascending=False)
    for target in a.targets:
        ti = R.index[sym == target][0]
        CL[f"{target}_patient_corr"] = float(sp[ti] - bp)
    CL.to_csv(os.path.join(out, "celltype_bulk_snrna_offset.csv"), index=False)

    with open(os.path.join(out, "provenance.json"), "w") as f:
        json.dump({"bulk_substrate": "results/percentile_ranks_universe.csv",
                   "n_universe": int(R.shape[0]), "n_with_snrna_profile": len(shared),
                   "snrna_reference": "GSE241386 / GSM7728058, ONE sample, used as a "
                                      "cell-type lookup and NOT as a control group",
                   "controls": ctl, "patient_fragments": pat,
                   "absolute_expression_compared_across_platforms": False,
                   "deterministic": True, "n_matched": a.n_matched,
                   "ab_tol_log2": a.ab_tol, "len_tol_log2": a.len_tol,
                   "gates": gate_rows}, f, indent=1)

    print(f"universe {R.shape[0]} genes | {len(shared)} with an snRNA profile | "
          f"{a.n_matched} matched genes per target | deterministic\n")
    for target in a.targets:
        print(f"### {target}: composition-adjusted position, patient vs control")
        for mt in ("profile+abundance", "profile+abundance+length"):
            d = D[(D.gene == target) & (D.matching == mt)]
            if not len(d):
                continue
            pp = d[d.group == "patient"].adj_position
            cc = d[d.group == "control"].adj_position
            side = "above" if pp.mean() > cc.mean() else "below"
            k = int((cc < pp.mean()).sum()) if side == "above" else int((cc > pp.mean()).sum())
            print(f"  matched on {mt}")
            print(f"    patient  {', '.join(f'{v:.1f}' for v in pp)}   mean {pp.mean():.1f}%")
            print(f"    controls {', '.join(f'{v:.1f}' for v in sorted(cc))}   "
                  f"mean {cc.mean():.1f}%")
            print(f"    patient mean is {side} {k} of {len(cc)} controls | fragments beyond "
                  f"every control: "
                  f"{int((pp > cc.max()).sum()) if side == 'above' else int((pp < cc.min()).sum())}"
                  f" of {len(pp)} | ranges "
                  f"{'separate' if (pp.min() > cc.max() or pp.max() < cc.min()) else 'OVERLAP'}")
            # Consistency across individuals (3 fragments are one person, 6 donors are six).
            # Direction is deviation from expectation (adj_position 50), not patient vs control.
            dev = pd.concat([pp, cc]).mean() - 50.0
            sgn = 1 if dev > 0 else -1
            word = "above" if sgn > 0 else "below"
            u = [bool(((pp - 50) * sgn > 0).all())] + [bool((v - 50) * sgn > 0) for v in cc]
            print(f"    {word} composition expectation in {sum(u)} of {len(u)} independent "
                  f"individuals, sign-test P = "
                  f"{binomtest(sum(u), len(u), 0.5, alternative='greater').pvalue:.4f}")
            gap = pp.mean() - cc.mean()
            pc = float((NB.gap < gap).mean())
            print(f"    patient-minus-control gap {gap:+.1f} points -> {100 * pc:.1f}th "
                  f"percentile of {len(NB)} random matched genes, empirical P = {1 - pc:.3f}"
                  f"  [{'NOT ' if (1 - pc) >= 0.05 else ''}receptor-specific]")
        print()
    print("cell-class bulk-minus-snRNA offset, baseline-corrected:")
    print(CL.to_string(index=False, float_format=lambda v: f"{v:+8.1f}"))
    print(f"\nnull control: {len(NB)} random genes, patient-minus-control gap median "
          f"{NB.gap.median():+.2f}, 5-95% {NB.gap.quantile(.05):+.1f} to "
          f"{NB.gap.quantile(.95):+.1f}")
    print("  -> the statistic itself is unbiased between groups, but its sampling spread with "
          "one\n     patient against six donors is wide enough that an ordinary gene can "
          "produce a large gap.")
    print("\nNOTE: the patient is ONE individual. Three fragments are subsamples, not "
          "replicates.\nAny 3-versus-6 test is pseudoreplicated and none is reported above.")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
