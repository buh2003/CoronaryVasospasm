#!/usr/bin/env python3
"""Empirical null percentile for the composition-matched procedure.

Usage
-----
    python analysis/04_percentile_null.py --proj . \
        --lengths data/transcript_lengths_refseq_select.csv \
        --scdata <directory containing GSM7728058_*>

Inputs
    results/percentile_ranks_universe.csv
    data/transcript_lengths_refseq_select.csv
    GEO GSE241386 matrix (--scdata)

Outputs
    results/percentile_null/percentile_null_by_gene.csv
    results/percentile_null/percentile_null_summary.csv
    results/percentile_null/receptors_vs_null.csv
    results/percentile_null/provenance.json
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module

_c3 = import_module("03_composition_control")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", default=".")
    ap.add_argument("--scdata", required=True)
    ap.add_argument("--lengths", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--targets", nargs="+", default=["EDNRB", "EDNRA"])
    ap.add_argument("--n-matched", type=int, default=300)
    ap.add_argument("--ab-tol", type=float, default=0.5)
    ap.add_argument("--len-tol", type=float, default=0.5)
    ap.add_argument("--n-probe", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    R = pd.read_csv(os.path.join(a.proj, "results", "percentile_ranks_universe.csv"),
                    index_col=0)
    sym = R.pop("gene_name")
    pat = [c for c in R.columns if c in ("1", "2a", "3")]
    ctl = [c for c in R.columns if "healthy" in c or "diseased" in c]
    assert len(pat) == 3 and len(ctl) == 6, (pat, ctl)

    S, tis, _ = _c3.snrna(a.scdata, os.path.join(a.proj, "results",
                                                 "nucleus_assignments.csv"))
    Pr = S.div(S.sum(axis=1), axis=0)
    glen = sym.map(pd.read_csv(a.lengths, index_col=0).iloc[:, 0])

    shared = [g for g in R.index if sym.loc[g] in set(Pr.index)]
    sh_idx = pd.Index(shared)
    gsym = sym.loc[shared].values
    prof = Pr.loc[gsym].values
    tcpm = np.log2(tis.loc[gsym].values + 1)
    glog = np.log2(glen[sh_idx].values)
    Rv = R.loc[sh_idx].values
    cols = list(R.columns)

    def percentiles_for(gi):
        """Within-sample percentile of probe gene gi among its matched set, per library."""
        d = np.abs(prof - prof[gi]).sum(axis=1)
        ok = ((np.abs(tcpm - tcpm[gi]) <= a.ab_tol)
              & (np.abs(glog - glog[gi]) <= a.len_tol)
              & np.isfinite(glog) & (np.arange(len(gsym)) != gi))
        c2 = np.where(ok)[0]
        if len(c2) < 50:
            return None
        mi = c2[np.argsort(d[c2])[:a.n_matched]]
        return (Rv[mi] < Rv[gi][None, :]).mean(axis=0) * 100

    # abundance deciles over the matchable universe, so probe and receptor share a scale
    pool = np.where((tcpm > np.log2(5)) & np.isfinite(glog))[0]
    dec_edges = np.quantile(tcpm[pool], np.linspace(0, 1, 11))
    dec_edges[0] -= 1e-9

    def decile_of(x):
        return int(np.clip(np.searchsorted(dec_edges, x, side="left"), 1, 10))

    tgt_i = {t: int(np.where(gsym == t)[0][0]) for t in a.targets
             if len(np.where(gsym == t)[0])}
    missing = [t for t in a.targets if t not in tgt_i]
    assert not missing, f"target not in the matchable universe: {missing}"

    rng = np.random.default_rng(a.seed)
    probe = rng.choice(pool, min(a.n_probe, len(pool)), replace=False)
    probe = np.array([g for g in probe if gsym[g] not in set(a.targets)])

    rows = []
    for gi in probe:
        pc = percentiles_for(gi)
        if pc is None:
            continue
        for c, v in zip(cols, pc):
            rows.append({"gene": gsym[gi], "library": c,
                         "group": "patient" if c in pat else "control",
                         "abundance_log2cpm": float(tcpm[gi]),
                         "abundance_decile": decile_of(tcpm[gi]),
                         "percentile": float(v)})
    N = pd.DataFrame(rows)
    assert len(N), "no probe gene produced a matched set"
    N.to_csv(os.path.join(a.outdir, "percentile_null_by_gene.csv"), index=False)

    g = N.groupby("abundance_decile").percentile
    SU = pd.DataFrame({"n_obs": g.size(), "mean": g.mean(), "median": g.median(),
                       "sd": g.std(), "p5": g.quantile(.05),
                       "p95": g.quantile(.95)}).reset_index()
    SU.loc[len(SU)] = ["all", len(N), N.percentile.mean(), N.percentile.median(),
                       N.percentile.std(), N.percentile.quantile(.05),
                       N.percentile.quantile(.95)]
    SU.to_csv(os.path.join(a.outdir, "percentile_null_summary.csv"), index=False)

    # each receptor against the null of its own abundance decile
    out = []
    for t, gi in tgt_i.items():
        obs = percentiles_for(gi)
        dec = decile_of(tcpm[gi])
        nd = N[N.abundance_decile == dec].percentile.values
        for c, v in zip(cols, obs):
            p = (np.abs(nd - nd.mean()) >= abs(v - nd.mean())).mean() if len(nd) else np.nan
            out.append({"gene": t, "library": c,
                        "group": "patient" if c in pat else "control",
                        "abundance_decile": dec, "observed_percentile": float(v),
                        "null_mean": float(nd.mean()), "null_sd": float(nd.std()),
                        "null_p5": float(np.quantile(nd, .05)),
                        "null_p95": float(np.quantile(nd, .95)),
                        "n_null_obs": int(len(nd)), "empirical_p_two_sided": float(p)})
    TV = pd.DataFrame(out)
    TV.to_csv(os.path.join(a.outdir, "receptors_vs_null.csv"), index=False)

    # does the null drift with abundance?
    dd = N.groupby("abundance_decile").percentile.mean()
    from scipy import stats as _st
    rho, prho = _st.spearmanr(dd.index.values, dd.values)
    centered = abs(N.percentile.mean() - 50)

    checks = [
        ("probe genes produced matched sets", int(N.gene.nunique()) >= 200),
        ("null is centered within 5 points of 50", centered <= 5),
        ("no abundance trend in the null (|rho| < 0.6)", abs(rho) < 0.6),
        ("both receptors scored", set(TV.gene) == set(a.targets)),
        ("every library represented", TV.library.nunique() == len(cols)),
    ]
    json.dump({"n_probe_genes": int(N.gene.nunique()),
               "n_libraries": len(cols),
               "null_mean_percentile": float(N.percentile.mean()),
               "null_median_percentile": float(N.percentile.median()),
               "abundance_trend_spearman_rho": float(rho),
               "abundance_trend_p": float(prho),
               "receptor_deciles": {t: decile_of(tcpm[gi]) for t, gi in tgt_i.items()},
               "params": {"n_matched": a.n_matched, "ab_tol": a.ab_tol,
                          "len_tol": a.len_tol, "n_probe": a.n_probe, "seed": a.seed},
               "checks": {k: bool(v) for k, v in checks}},
              open(os.path.join(a.outdir, "provenance.json"), "w"), indent=1)

    print(f"probe genes {N.gene.nunique()} | observations {len(N)}")
    print(f"\nnull percentile by abundance decile:")
    print(SU.to_string(index=False, float_format=lambda v: f"{v:8.2f}"))
    print(f"\nabundance trend: Spearman rho = {rho:+.3f}, P = {prho:.3g}")
    print(f"\nreceptors against the null of their own decile:")
    print(TV.groupby(["gene", "abundance_decile"])
          .agg(obs_mean=("observed_percentile", "mean"),
               null_mean=("null_mean", "mean"),
               null_p5=("null_p5", "mean"), null_p95=("null_p95", "mean"),
               max_p=("empirical_p_two_sided", "max"))
          .to_string(float_format=lambda v: f"{v:8.2f}"))
    for k, v in checks:
        print(f"[{'PASS' if v else 'FAIL'}] {k}")
    return 0 if all(v for _, v in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
