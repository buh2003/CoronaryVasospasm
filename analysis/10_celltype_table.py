#!/usr/bin/env python3
"""Pool the four cohort tables onto common columns.

Usage
-----
    python analysis/10_celltype_table.py --proj .

Inputs
    results/ganglion_snrna_receptors.csv
    results/coronary_receptors_by_celltype.csv
    results/heartmap_receptors_by_celltype.csv

Outputs
    results/receptor_by_celltype.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

# cohort -> (file, n column, EDNRB detection column, EDNRA detection column)
SOURCES = {
    "HeartMap": ("heartmap_receptors_by_celltype.csv",
                 "n_cells_subsampled", "EDNRB_pct_cells", "EDNRA_pct_cells"),
    "coronary": ("coronary_receptors_by_celltype.csv",
                 "n_cells", "EDNRB_pct_cells", "EDNRA_pct_cells"),
    "GanglionSN": ("ganglion_snrna_receptors.csv",
                   "n_nuclei", "EDNRB_pct_nuclei", "EDNRA_pct_nuclei"),
}
CORONARY_COHORTS = {"TabulaSapiens_normal": "TabulaSapiens",
                    "PlaqueAtlas_athero": "PlaqueAtlas"}

MURAL_RE = r"myocyt|smooth muscle|SMC|pericyte|mural|myofibro"
MURAL, OTHER = "contractile/mural", "other"


def normalise(cohort, frame, ncol, bcol, acol):
    """One cohort reduced to common columns, with the contractile and mural label."""
    d = pd.DataFrame({
        "cohort": cohort,
        "cell_type": frame["cell_type"].astype(str).values,
        "n": frame[ncol].values,
        "ratio_B_over_A": frame["ratio_B_over_A"].values,
        "B_pct": frame[bcol].values,
        "A_pct": frame[acol].values,
    })
    d["class"] = np.where(d.cell_type.str.contains(MURAL_RE, case=False, regex=True),
                          MURAL, OTHER)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", default=".")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    res = os.path.join(a.proj, "results")
    out = a.outdir or res
    os.makedirs(out, exist_ok=True)

    parts = []
    for cohort, (fname, ncol, bcol, acol) in SOURCES.items():
        raw = pd.read_csv(os.path.join(res, fname))
        if cohort == "coronary":
            unknown = sorted(set(raw.dataset.unique()) - set(CORONARY_COHORTS))
            if unknown:
                sys.exit(f"unexpected dataset labels in {fname}: {unknown}")
            for src, name in CORONARY_COHORTS.items():
                parts.append(normalise(name, raw[raw.dataset == src].reset_index(drop=True),
                                       ncol, bcol, acol))
        else:
            parts.append(normalise(cohort, raw, ncol, bcol, acol))

    allrows = pd.concat(parts, ignore_index=True)
    finite = np.isfinite(pd.to_numeric(allrows.ratio_B_over_A, errors="coerce"))
    T = (allrows[finite]
         .sort_values(["cohort", "ratio_B_over_A"], kind="mergesort")
         .reset_index(drop=True))

    mural = T[T["class"] == MURAL]
    other = T[T["class"] == OTHER]
    checks = [
        ("every cohort contributes rows", T.cohort.nunique() == 4),
        ("no contractile or mural cell type is EDNRB-weighted outside the ganglion",
         (mural[mural.cohort != "GanglionSN"].ratio_B_over_A < 1).all()),
        ("every non-mural ganglion class is EDNRB-weighted",
         (other[other.cohort == "GanglionSN"].ratio_B_over_A >= 1).all()),
        ("the ganglion's own mural class is EDNRA-weighted",
         (mural[mural.cohort == "GanglionSN"].ratio_B_over_A < 1).all()),
    ]
    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")

    T.to_csv(os.path.join(out, "receptor_by_celltype.csv"), index=False)
    print(f"\n{len(T)} cell types: {T.groupby('cohort').size().to_dict()}")
    print(f"contractile and mural {len(mural)}, other {len(other)}")
    print(f"written to {out}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
