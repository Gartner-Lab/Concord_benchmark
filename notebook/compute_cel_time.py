"""
compute_cel_time.py
===============

Port of Qin Zhu's R function `compute_time` to Python working with `anndata.AnnData`.
The function estimates developmental time for single cells by correlating their
expression profile with a bulk time‑course reference.

Example
-------
>>> import scanpy as sc, pandas as pd
>>> from compute_time import compute_time
>>> adata = sc.read_h5ad("All_combined_cds_annotated.h5ad")
>>> bulk = pd.read_csv("whole.embryo.interval.timecourse.tsv", sep="\t", index_col=0)
>>> ref_time = bulk.columns.str.replace("mpfc_", "").astype(int)
>>> compute_time(
...     adata,
...     bulk,
...     ref_time,
...     name="BestTime_Yanai",
...     detect_low=1,
...     detect_high=7,
...     minAutoCor=0.6,
...     minSD=1.5,
...     span=0.5,
... )
>>> adata.obs[["BestTime_Yanai", "BestTime_Yanai_SD", "BestTime_Yanai_DR"]].head()
"""

from __future__ import annotations
from typing import Sequence, Optional, Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
from statsmodels.nonparametric.smoothers_lowess import lowess
import anndata as ad


def _search_peak(
    y: np.ndarray,
    t: Sequence[float],
    span: float,
    mode: Literal["max", "min"] = "max",
) -> int:
    """LOWESS‑smooth *y* vs *t* and return the index of its extremum."""
    y_smooth = lowess(y, t, frac=span, return_sorted=False)
    return int(np.argmax(y_smooth) if mode == "max" else np.argmin(y_smooth))


def compute_time(
    adata: ad.AnnData,
    bulk_expr: pd.DataFrame,
    ref_time: Sequence[float],
    *,
    name: str = "BestTime",
    layer: Optional[str] = None,
    pseudoCountSingle: float = 1.0,
    pseudoCountBulk: float = 1.0,
    minAutoCor: float = 0.5,
    minSD: float = 1.0,
    log: bool = True,
    detect_low: float = 0.0,
    detect_high: float = np.inf,
    warn_thresh: int = 50,
    span: float = 0.5,
    return_adata: bool = True,
):
    """Estimate the best matching time point for every cell in *adata*.

    Parameters
    ----------
    adata
        `AnnData` object with gene names in ``adata.var_names``.
    bulk_expr
        Bulk (genes × samples) expression matrix as `pandas.DataFrame`.
    ref_time
        Numeric time for each sample/column in *bulk_expr*.
    name
        Key to save the estimated time in ``adata.obs``.
    layer
        Which layer to read counts from (defaults to ``adata.X``).
    pseudoCountSingle, pseudoCountBulk
        Pseudocounts added before log‑transform.
    minAutoCor, minSD
        Gene filters based on bulk temporal autocorrelation and variance.
    log
        If *True*, apply ``log2(count + pseudocount)`` to both matrices.
    detect_low, detect_high
        Gene‑level thresholds used when calculating correlations for a cell.
    warn_thresh
        Warn if a cell uses fewer than this number of genes.
    span
        Fraction of points for LOWESS smoothing.
    return_adata
        If *True*, annotate and return *adata*; otherwise return NumPy array.

    Returns
    -------
    AnnData or numpy.ndarray
        Annotated AnnData or array of best time estimates.
    """

    ref_time = np.asarray(ref_time)
    if ref_time.size != bulk_expr.shape[1]:
        raise ValueError("`ref_time` must match number of bulk samples.")

    # 1. Align genes
    sc_mat = adata.layers[layer] if layer else adata.X
    if sp.issparse(sc_mat):
        sc_mat = sc_mat.T.todense()  # genes × cells
    else:
        sc_mat = sc_mat.T  # genes × cells
    sc_mat = np.asarray(sc_mat)

    bulk_expr = bulk_expr.loc[~bulk_expr.index.duplicated(keep="first")]
    shared_genes = bulk_expr.index.intersection(adata.var_names)
    if shared_genes.empty:
        raise ValueError("No overlapping genes between single‑cell and bulk data.")
    print(f"Keep {len(shared_genes)} shared genes.")

    bulk_mat = bulk_expr.loc[shared_genes].to_numpy()
    sc_mat = sc_mat[adata.var_names.get_indexer(shared_genes)]

    # 2. Log‑transform
    if log:
        bulk_mat = np.log2(bulk_mat + pseudoCountBulk)
        sc_mat = np.log2(sc_mat + pseudoCountSingle)

    # 3. Bulk gene filters
    g_autocor = np.array([
        np.corrcoef(g, np.roll(g, -1))[0, 1] for g in bulk_mat
    ])
    g_sd = bulk_mat.std(axis=1)
    keep = (g_sd > minSD) & (g_autocor > minAutoCor)
    if keep.sum() == 0:
        raise ValueError("No genes passed SD/autocorrelation thresholds.")
    print(f"{keep.sum()} genes passed bulk variance & autocorrelation filters.")

    bulk_mat = bulk_mat[keep]
    sc_mat = sc_mat[keep]

    # 4. Correlation matrix (timepoints × cells)
    n_cells = sc_mat.shape[1]
    n_tp = bulk_mat.shape[1]
    cor_mtx = np.empty((n_tp, n_cells), dtype=np.float32)
    warn_num = 0

    bulk_centered = bulk_mat - bulk_mat.mean(axis=0)
    bulk_norm = np.sqrt((bulk_centered ** 2).sum(axis=0))

    for i in range(n_cells):
        x = sc_mat[:, i]
        mask = (x >= detect_low) & (x <= detect_high)
        if mask.sum() < warn_thresh:
            warn_num += 1
        xm = x[mask] - x[mask].mean()
        denom = np.sqrt((xm ** 2).sum())
        if denom == 0:
            cor_mtx[:, i] = np.nan
            continue
        bc = bulk_centered[mask]
        cor_mtx[:, i] = (xm[:, None] * bc).sum(axis=0) / (denom * bulk_norm)
        if (i + 1) % 1000 == 0:
            print(f"Computed {i + 1} / {n_cells} cells")

    print(
        f"{warn_num} cells had fewer than {warn_thresh} genes for correlations."
    )

    # 5. Derive metrics per cell
    best_idx = np.fromiter(
        (_search_peak(cor_mtx[:, i], ref_time, span, mode="max") for i in range(n_cells)),
        dtype=int,
    )
    best_time = ref_time[best_idx]
    cor_sd = cor_mtx.std(axis=0)
    cor_range = cor_mtx.max(axis=0) - cor_mtx.min(axis=0)

    if return_adata:
        adata.obs[name] = best_time
        adata.obs[f"{name}_SD"] = cor_sd
        adata.obs[f"{name}_DR"] = cor_range
        return adata
    return best_time
