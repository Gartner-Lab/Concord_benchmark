

# misclass_tools.py
from __future__ import annotations
from typing import Literal, Mapping
import numpy as np, pandas as pd
from anndata import AnnData
from sklearn.metrics.pairwise import pairwise_distances
try:
    from sklearn.metrics.pairwise import cosine_distances          # ≥0.24
except ImportError:                                                # older skl
    from sklearn.metrics.pairwise import cosine_similarity
    def cosine_distances(A, B): return 1 - cosine_similarity(A, B)


# ──────────────────────────────────────────────────────────────────────
#  A. helpers
# ──────────────────────────────────────────────────────────────────────
def fetch_matrix(adata: AnnData, key: str | None = "X") -> np.ndarray:
    X = adata.X if key in (None, "X") else adata.obsm[key]
    return X.toarray() if not isinstance(X, np.ndarray) else X


def compute_centroids(
    adata: AnnData,
    label_key: str,               # column in adata.obs
    feature_key: str | None = "X"
) -> Mapping[str, np.ndarray]:
    """Mean vector of every label in a fixed feature space."""
    X = fetch_matrix(adata, feature_key)
    labels = adata.obs[label_key].to_numpy()
    centroids: dict[str, np.ndarray] = {}
    for lb in np.unique(labels):
        mask = labels == lb
        if mask.any():
            centroids[lb] = X[mask].mean(0)
    return centroids


def _single_distance(
    v: np.ndarray, c: np.ndarray, metric: str
) -> float:
    if metric == "cosine":
        return cosine_distances(v[None], c[None])[0, 0]
    return pairwise_distances(v[None], c[None], metric=metric)[0, 0]


# ──────────────────────────────────────────────────────────────────────
#  B. core routine (requires pre-computed centroids)
# ──────────────────────────────────────────────────────────────────────
def misclass_distance_analysis(
    adata: AnnData,
    pred_df: pd.DataFrame,            # y_true / y_pred DataFrame
    centroids: Mapping[str, np.ndarray],
    feature_key: str | None = "X",
    metric: Literal["euclidean", "cosine"] = "cosine",
) -> pd.DataFrame:
    """
    Distance of each mis-classified cell to (true, predicted) centroids.
    """
    if {"y_true", "y_pred"} - set(pred_df.columns):
        raise ValueError("pred_df must contain y_true & y_pred columns.")

    mis = pred_df[pred_df.y_true != pred_df.y_pred]
    if mis.empty:
        return pd.DataFrame()

    X      = fetch_matrix(adata, feature_key)
    ad_idx = pd.Index(adata.obs_names)

    rows = []
    for cid, row in mis.iterrows():
        xi   = X[ad_idx.get_loc(cid)]
        y_t, y_p = row.y_true, row.y_pred
        if y_t not in centroids or y_p not in centroids:
            continue
        d_t = _single_distance(xi, centroids[y_t], metric)
        d_p = _single_distance(xi, centroids[y_p], metric)
        rows.append(
            dict(cell=cid, y_true=y_t, y_pred=y_p,
                 dist_true=d_t, dist_pred=d_p,
                 margin=d_t - d_p, closer_to_pred=d_p < d_t)
        )
    return pd.DataFrame(rows).set_index("cell")


# ──────────────────────────────────────────────────────────────────────
#  C. high-level wrapper for an entire probe-bank
# ──────────────────────────────────────────────────────────────────────
def analyse_pred_bank(
    adata: AnnData,
    pred_bank: dict,
    *,
    probe_type: str = "KNN",        # "KNN" or "Linear"
    bank_target_key: str = "state", # outer-dict key in pred_bank
    obs_label_key: str,             # column name in adata.obs (e.g. "cell_type")
    feature_key: str | None = "X",
    metric: str = "cosine",
) -> dict[str, pd.DataFrame]:
    """
    Returns {embedding_key → misclass_df} using shared centroids.
    """
    centroids = compute_centroids(adata, obs_label_key, feature_key)
    result: dict[str, pd.DataFrame] = {}

    for emb_key, preds in pred_bank[probe_type][bank_target_key].items():
        df = misclass_distance_analysis(
            adata, preds, centroids,
            feature_key=feature_key, metric=metric
        )
        if not df.empty:
            result[emb_key] = df
    return result



from typing import Literal

def summarise_misclassification(
    mis_dict: dict[str, pd.DataFrame],
    pred_bank: dict,
    *,
    probe_type: str = "KNN",
    bank_target_key: str = "state",
    sort_by: Literal[None, "mis_rate", "frac_closer"] = None,
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Returns a tidy DataFrame with:
        embedding | mis_rate | frac_closer | closer_height | further_height
    """
    rows = []
    for emb_key, mis_df in mis_dict.items():
        total_cells = len(pred_bank[probe_type][bank_target_key][emb_key])
        mis_cells   = len(mis_df)
        if total_cells == 0:
            continue
        mis_rate    = mis_cells / total_cells
        frac_closer = mis_df["closer_to_pred"].mean()      # within mis-classified
        rows.append({
            "embedding": emb_key,
            "mis_rate": mis_rate,
            "frac_closer": frac_closer,
            "closer_height": mis_rate * frac_closer,
            "further_height": mis_rate * (1 - frac_closer),
        })

    df = pd.DataFrame(rows)
    if sort_by in {"mis_rate", "frac_closer"}:
        df = df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
    return df


def plot_misclassification_bars(
        summary_df: pd.DataFrame, 
        figsize=(6, 4),
        dpi=300,
        ) -> tuple[plt.Figure, plt.Axes]:
    """
    Stacked bar-plot with a % label on the green segment
    (percentage of mis-classified cells that lie closer to prediction).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(summary_df))
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # ── stacked bars ─────────────────────────────────────────────── #
    green_bars = ax.bar(
        x,
        summary_df["closer_height"],
        color="#4daf4a",
        label="Closer to predicted",
    )
    grey_bars = ax.bar(
        x,
        summary_df["further_height"],
        bottom=summary_df["closer_height"],
        color="lightgrey",
        label="Closer to label",
    )

    # ── add % labels to green segment ────────────────────────────── #
    for rect, frac in zip(green_bars, summary_df["frac_closer"]):
        height = rect.get_height()
        if height == 0:           # no mis-classified cells for this method
            continue
        # place text in the middle of the green block (or just above if tiny)
        y = rect.get_y() + height / 2
        if height < 0.02:         # small bar → move label above
            y = rect.get_y() + height + 0.005
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            y,
            f"{frac*100:.0f}%",
            ha="center",
            va="center",
            fontsize=8,
            color="white" if height >= 0.02 else "black",
            fontweight="bold",
        )

    # ── cosmetics ────────────────────────────────────────────────── #
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["embedding"], rotation=45, ha="right")
    ax.set_ylabel("Fraction of all cells")
    ax.set_ylim(0, summary_df["mis_rate"].max() * 1.1)
    ax.set_title("Misclassification rate and distance consistency")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax


import numpy as np, pandas as pd, matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────
#  1) Summarize: correct + salvageable (by embedding)
# ──────────────────────────────────────────────────────────────
def summarise_salvageable(
    mis_dict: dict[str, pd.DataFrame],
    pred_bank: dict,
    *,
    probe_type: str = "KNN",
    bank_target_key: str = "state",
    sort: bool = True,
    ascending: bool = False,   # sort by total_rate (desc by default)
) -> pd.DataFrame:
    """
    Builds a table with:
      embedding | total | correct_rate | salvage_rate | mis_rate | frac_closer | total_rate
    where:
      total_rate = correct_rate + salvage_rate
      salvage_rate = mis_rate * frac_closer
      mis_rate = 1 - correct_rate
    """
    rows = []
    for emb_key, preds_df in pred_bank[probe_type][bank_target_key].items():
        total = len(preds_df)
        if total == 0:
            continue

        # correct rate from the prediction table itself
        correct_rate = (preds_df["y_true"] == preds_df["y_pred"]).mean()

        # misclassified slice (may be empty if this emb got all right)
        mis_df = mis_dict.get(emb_key, pd.DataFrame(index=[]))
        mis_rate = 1.0 - correct_rate

        # among misclassified, what fraction are closer to predicted?
        frac_closer = 0.0 if len(mis_df) == 0 else float(mis_df["closer_to_pred"].mean())

        salvage_rate = mis_rate * frac_closer
        total_rate   = correct_rate + salvage_rate

        rows.append({
            "embedding": emb_key,
            "total": total,
            "correct_rate": correct_rate,
            "salvage_rate": salvage_rate,
            "mis_rate": mis_rate,
            "frac_closer": frac_closer,
            "total_rate": total_rate,
        })

    df = pd.DataFrame(rows)
    if sort and not df.empty:
        df = df.sort_values("total_rate", ascending=ascending, kind="mergesort").reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────
#  2) Plot: stacked bars of correct + salvageable
# ──────────────────────────────────────────────────────────────
def plot_salvageable_accuracy_bars(summary_df: pd.DataFrame,
                                   label_total: bool = True,
                                   label_salvage: bool = True,
                                   figsize=(6, 4)):
    """
    Stacked bars:
      bottom  = correct_rate
      top     = salvage_rate
      total   = correct + salvageable  (sorted beforehand)
    """
    if summary_df.empty:
        raise ValueError("summary_df is empty. Did you run summarise_salvageable()?")

    x = np.arange(len(summary_df))
    fig, ax = plt.subplots(figsize=figsize)

    bars_correct = ax.bar(
        x, summary_df["correct_rate"],
        label="Correct", color="#377eb8"
    )
    bars_salvage = ax.bar(
        x, summary_df["salvage_rate"],
        bottom=summary_df["correct_rate"],
        label="Miscls. but closer to predicted",
        color="#4daf4a"
    )

    # optional labels
    if label_salvage:
        for r, rect in zip(summary_df["salvage_rate"], bars_salvage):
            if r <= 0:
                continue
            ax.text(rect.get_x() + rect.get_width()/2,
                    rect.get_y() + rect.get_height()/2,
                    f"{r*100:.0f}%",
                    ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")

    if label_total:
        totals = summary_df["total_rate"].to_numpy()
        for xi, h in zip(x, totals):
            ax.text(xi, h + 0.01, f"{h*100:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["embedding"], rotation=45, ha="right")
    ax.set_ylabel("Fraction of all cells")
    ax.set_ylim(0, min(1.0, summary_df["total_rate"].max() * 1.10))
    ax.set_title("Correct + salvageable fraction (distance-consistent)")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Patch

# ─────────────────────────────────────────────────────────────
# Combine two summaries into a tidy long df and order methods
# summary_* must have: ['embedding','correct_rate','salvage_rate','total_rate']
# ─────────────────────────────────────────────────────────────
def combine_summaries_for_plot(
    summary_ct: pd.DataFrame,
    summary_lin: pd.DataFrame,
    *,
    method_filter: list[str] | None = None,
    order_by: str = "max",      # "Cell type" | "Lineage" | "max" | "mean"
):
    df_ct  = summary_ct.copy();  df_ct["target"] = "Cell type"
    df_lin = summary_lin.copy(); df_lin["target"] = "Lineage"
    df = pd.concat([df_ct, df_lin], ignore_index=True)

    if method_filter is not None:
        df = df[df["embedding"].isin(method_filter)]
        df["embedding"] = pd.Categorical(df["embedding"],
                                         categories=method_filter, ordered=True)

    if order_by in ("Cell type", "Lineage"):
        order = (df[df["target"] == order_by]
                 .sort_values("total_rate", ascending=False)
                 .drop_duplicates("embedding"))["embedding"].tolist()
    elif order_by == "max":
        order = (df.groupby("embedding")["total_rate"].max()
                   .sort_values(ascending=False).index.tolist())
    elif order_by == "mean":
        order = (df.groupby("embedding")["total_rate"].mean()
                   .sort_values(ascending=False).index.tolist())
    else:
        order = df["embedding"].unique().tolist()

    df["embedding"] = pd.Categorical(df["embedding"], categories=order, ordered=True)
    df = df.sort_values(["embedding", "target"]).reset_index(drop=True)
    return df


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Patch

def _lighten(hex_color: str, alpha: float = 0.6) -> str:
    """
    Blend a hex color toward white by `alpha` (0..1).
    alpha=0 → original; alpha=1 → white.
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0,2,4))
    r = int(r + (255 - r) * alpha)
    g = int(g + (255 - g) * alpha)
    b = int(b + (255 - b) * alpha)
    return f"#{r:02x}{g:02x}{b:02x}"

def plot_correct_plus_salvageable_shades(
    df_long: pd.DataFrame,
    *,
    # palette maps target → (dark, light). If only dark is provided,
    # we auto-generate a light shade.
    palette: dict[str, tuple[str, str] | str] | None = None,
    figsize=(5.2, 3.2),
    dpi=300,
    annotate_components=True,
    annotate_total=True,
    comp_label_thresh=0.03,   # hide tiny in-bar labels (<3%)
):
    """
    df_long columns required:
      ['embedding','target','correct_rate','salvage_rate','total_rate']
    Expects `embedding` as an ordered Categorical (use combine_summaries_for_plot).
    """
    if palette is None:
        # colorblind-friendly bases (dark); lights auto-computed
        palette = {
            "Cell type": "#1f77b4",   # dark blue
            "Lineage":   "#6a3d9a",   # dark purple
        }

    # normalize palette to (dark, light)
    norm_palette: dict[str, tuple[str, str]] = {}
    for t, col in palette.items():
        if isinstance(col, tuple):
            norm_palette[t] = col
        else:
            norm_palette[t] = (col, _lighten(col, alpha=0.55))

    methods = (df_long["embedding"].cat.categories.tolist()
               if hasattr(df_long["embedding"], "cat") else
               sorted(df_long["embedding"].unique()))
    pivot_corr  = df_long.pivot(index="embedding", columns="target",
                                values="correct_rate").reindex(methods)
    pivot_salv  = df_long.pivot(index="embedding", columns="target",
                                values="salvage_rate").reindex(methods)
    pivot_total = (pivot_corr.fillna(0) + pivot_salv.fillna(0))

    x = np.arange(len(methods), dtype=float)
    width = 0.36
    offsets = {"Cell type": -width/2, "Lineage": +width/2}

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    bars = {"correct": {}, "salvage": {}}
    for target in ["Cell type", "Lineage"]:
        corr = pivot_corr.get(target, pd.Series(0, index=methods)).to_numpy()
        salv = pivot_salv.get(target, pd.Series(0, index=methods)).to_numpy()
        dark, light = norm_palette[target]

        # bottom: Correct (dark)
        bc = ax.bar(x + offsets[target], corr, width=width,
                    color=dark, edgecolor="none")
        # top: Salvageable (light)
        bs = ax.bar(x + offsets[target], salv, width=width, bottom=corr,
                    color=light, edgecolor="none")
        bars["correct"][target] = bc
        bars["salvage"][target] = bs

        # in-segment percentage labels
        if annotate_components:
            for rc, rs, rect_c, rect_s in zip(corr, salv, bc, bs):
                if rc >= comp_label_thresh:
                    ax.text(rect_c.get_x() + rect_c.get_width()/2,
                            rc/2,
                            f"{rc*100:.0f}%",
                            ha="center", va="center",
                            color="white", fontsize=8, fontweight="bold")
                if rs >= comp_label_thresh:
                    y0 = rect_s.get_y()
                    ax.text(rect_s.get_x() + rect_s.get_width()/2,
                            y0 + rs/2,
                            f"{rs*100:.0f}%",
                            ha="center", va="center",
                            color="black", fontsize=8)

    # total labels on top of each bar
    if annotate_total:
        for j, m in enumerate(methods):
            for target, off in offsets.items():
                total_h = float(pivot_total.loc[m, target]) if target in pivot_total.columns else 0.0
                if total_h <= 0:
                    continue
                ax.text(x[j] + off, total_h + 0.012,
                        f"{total_h*100:.0f}%",
                        ha="center", va="bottom", fontsize=8)

    # publication-y cosmetics
    ax.set_xlim(-0.6, len(methods) - 0.4)
    ymax = min(1.0, float(pivot_total.max().max()) * 1.10 if not pivot_total.empty else 1.0)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("Fraction of all cells")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=0, ha="center")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_title("Correct + salvageable fractions by target", pad=8)

    # legend: 4 entries (dark/light per target)
    legend_handles = [
        Patch(facecolor=norm_palette["Cell type"][0], edgecolor="none", label="Cell type – Correct"),
        Patch(facecolor=norm_palette["Cell type"][1], edgecolor="none", label="Cell type – Salvageable"),
        Patch(facecolor=norm_palette["Lineage"][0],   edgecolor="none", label="Lineage – Correct"),
        Patch(facecolor=norm_palette["Lineage"][1],   edgecolor="none", label="Lineage – Salvageable"),
    ]

    # → place legend on the right in one column
    ax.legend(
        handles=legend_handles,
        frameon=False,
        ncols=1,                 # one column
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),  # just outside the axes on the right
        borderaxespad=0.0,
    )

    # leave space for the outside legend
    fig.tight_layout(rect=[0, 0, 0.82, 1])   # adjust if labels are long
    return fig, ax




import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from scipy.sparse import coo_matrix

# if you already have one, keep using your helper
def _get_X(adata, key):
    if key in (None, "X"):
        X = adata.X
        return X.toarray() if hasattr(X, "toarray") else X
    return adata.obsm[key]

def knn_refine_labels_fast(
    adata,
    *,
    label_col: str = "ct_or_lin",
    emb_key: str = "concord_hcl",
    k: int = 30,
    metric: str = "cosine",       # "cosine" or "euclidean"
    weight: str = "distance",     # "distance" or "uniform"
    bad_annotation = (np.nan, '', 'unknown', 'None', 'nan', 'NaN', 'NA', 'na', 'unannotated'),
    min_votes: int = 3,
    min_frac: float = 0.55,
    allow_flip: bool = True,
    max_iter: int = 1,
    self_exclude: bool = True,
    eps: float = 1e-8,
    dtype=np.float32,
):
    """
    Vectorized kNN label refinement / propagation.
    Returns: (refined_labels: pd.Series, vote_frac: pd.Series, votes_used: pd.Series)
    """
    n = adata.n_obs
    X = _get_X(adata, emb_key)
    X = np.asarray(X, dtype=dtype)

    # Speed trick for cosine: L2 normalize and use euclidean neighbors
    use_cosine = (metric == "cosine")
    X_nn = normalize(X) if use_cosine else X
    nn_metric = "euclidean" if use_cosine else "euclidean"  # Euclidean in both cases

    # ── kNN (computed once) ──────────────────────────────────────────
    n_neighbors = k + 1 if self_exclude else k
    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric=nn_metric,
        algorithm="auto",
        n_jobs=-1,
    ).fit(X_nn)
    dists, nbrs = nn.kneighbors(X_nn, return_distance=True)
    if self_exclude:
        dists, nbrs = dists[:, 1:], nbrs[:, 1:]  # drop self
    dists = dists.astype(dtype, copy=False)

    # ── labels → integer codes (−1 = invalid) ────────────────────────
    y = adata.obs[label_col].astype(object).to_numpy()
    bad_set = set(bad_annotation) - {np.nan}
    valid_mask = (~pd.isna(y)) & (~pd.Series(y).isin(bad_set).to_numpy())

    # factorize only valid labels to keep class count small
    valid_codes, uniques = pd.factorize(y[valid_mask], sort=False)
    y_codes = np.full(n, -1, dtype=np.int32)
    y_codes[valid_mask] = valid_codes.astype(np.int32)
    n_classes = int(valid_codes.max() + 1) if valid_codes.size else 0

    # Nothing to refine if no labeled cells
    if n_classes == 0:
        refined = pd.Series(y, index=adata.obs_names, dtype=object)
        return refined, pd.Series(0.0, index=adata.obs_names), pd.Series(0, index=adata.obs_names)

    # ── iterative refinement (labels participate in next round) ─────
    vote_frac = np.zeros(n, dtype=dtype)
    votes_used = np.zeros(n, dtype=np.int32)

    for _ in range(max_iter):
        prev_codes = y_codes.copy()

        # neighbor label codes per row
        L = y_codes[nbrs]                         # (n, k)
        valid_nb = (L != -1)                      # (n, k)
        cnt_valid = valid_nb.sum(axis=1).astype(np.int32)  # votes used per cell

        if weight == "distance":
            w = np.zeros_like(dists, dtype=dtype)
            # inverse distance on valid neighbors
            w[valid_nb] = 1.0 / (dists[valid_nb] + eps)
        else:
            w = valid_nb.astype(dtype)

        # Build sparse matrix of shape (n_cells × n_classes) with summed weights
        # rows: repeat row index for each valid neighbor
        rows = np.repeat(np.arange(n, dtype=np.int32), cnt_valid, axis=0)
        # columns: neighbor class codes for valid neighbors
        cols = L[valid_nb].astype(np.int32, copy=False)
        data = w[valid_nb].astype(dtype, copy=False)

        # Guard if there are rows without any valid neighbors
        if data.size == 0:
            break

        M = coo_matrix((data, (rows, cols)), shape=(n, n_classes), dtype=dtype).tocsr()

        # Top class per row, total weight, top weight
        # argmax along axis 1:
        top_cls = M.argmax(axis=1).A1.astype(np.int32)
        # row-wise max weight (need a quick way)
        # Using trick: take data per row via CSR:
        top_w = np.zeros(n, dtype=dtype)
        row_ptr = M.indptr
        for i in range(n):
            start, end = row_ptr[i], row_ptr[i + 1]
            if start < end:
                top_w[i] = M.data[start:end].max()
        tot_w = np.asarray(M.sum(axis=1)).ravel().astype(dtype, copy=False)

        frac = np.divide(top_w, tot_w, out=np.zeros_like(top_w), where=tot_w > 0)
        enough = cnt_valid >= min_votes

        cur_valid = y_codes != -1
        agree = (cur_valid) & (top_cls == y_codes)
        assign_mask = (~cur_valid) & enough & (frac >= min_frac)
        flip_mask   = (allow_flip) & cur_valid & (~agree) & enough & (frac >= min_frac)

        y_codes_new = y_codes.copy()
        y_codes_new[assign_mask | flip_mask] = top_cls[assign_mask | flip_mask]
        y_codes = y_codes_new

        vote_frac = frac
        votes_used = cnt_valid

        if np.array_equal(y_codes, prev_codes):
            break  # converged

    # ── map back to labels (object) ──────────────────────────────────
    refined = np.array(y, dtype=object)
    # rebuild full label array from codes (codes correspond to uniques)
    mask_final = y_codes != -1
    refined[mask_final] = uniques[y_codes[mask_final]]
    refined = pd.Series(refined, index=adata.obs_names, dtype=object)

    return refined, pd.Series(vote_frac, index=adata.obs_names), pd.Series(votes_used, index=adata.obs_names)




import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from pandas.api.types import is_categorical_dtype
import matplotlib.pyplot as plt

BAD_ANN_DEFAULT = (np.nan, '', 'unknown', 'None', 'nan', 'NaN', 'NA', 'na', 'unannotated')


def _compute_centroids_from_truth(adata, label_col, feature_key="X", bad_ann=BAD_ANN_DEFAULT):
    """Centroid per annotated label (uses only valid annotated rows)."""
    X = _get_X(adata, feature_key)
    y = adata.obs[label_col].copy()
    bad_set = set(bad_ann) - {np.nan}
    good = (~y.isna()) & (~y.isin(bad_set))
    if is_categorical_dtype(y):
        y = y.astype("category")
    labels = pd.Index(pd.unique(y[good]), name="label")
    # means per label
    cents = []
    for lb in labels:
        m = (y == lb).to_numpy()
        mu = X[m].mean(axis=0)
        mu = np.asarray(mu).ravel()
        cents.append(mu)
    C = np.vstack(cents) if len(cents) else np.zeros((0, _get_X(adata, feature_key).shape[1]))
    return labels, C  # labels: Index of class names in same order as rows of C

def compare_pred_vs_true_dist(
    adata,
    *,
    true_col="ct_or_lin",
    pred_col="concord_hcl_knn_refined_ct_or_lin",
    feature_key="X",
    metric="cosine",                # "cosine" or "euclidean"
    bad_ann=BAD_ANN_DEFAULT,
    changed_only=True,
    batch_size=50000,
):
    """
    Returns DataFrame with: y_true, y_pred, dist_true, dist_pred, margin, closer_to_pred
    Distances computed in `feature_key` space to centroids built from annotated labels.
    """
    # 0) masks & align categories for safe comparison
    y_true = adata.obs[true_col].copy()
    y_pred = adata.obs[pred_col].copy()
    bad_set = set(bad_ann) - {np.nan}
    has_true = (~y_true.isna()) & (~y_true.isin(bad_set))
    has_pred = (~y_pred.isna()) & (~y_pred.isin(bad_set))
    if is_categorical_dtype(y_true) or is_categorical_dtype(y_pred):
        y_true = y_true.astype("category"); y_pred = y_pred.astype("category")
        cats = y_true.cat.categories.union(y_pred.cat.categories)
        y_true = y_true.cat.set_categories(cats)
        y_pred = y_pred.cat.set_categories(cats)
    comparable = has_true & has_pred
    changed = comparable & (y_true != y_pred)
    use_mask = changed if changed_only else comparable
    if not use_mask.any():
        out = pd.DataFrame(columns=["y_true","y_pred","dist_true","dist_pred","margin","closer_to_pred"])
        out.index = pd.Index([], name="cell")
        return out

    # 1) centroids from annotated labels only
    cent_labels, C = _compute_centroids_from_truth(adata, true_col, feature_key, bad_ann=bad_ann)
    if C.shape[0] == 0:
        out = pd.DataFrame(columns=["y_true","y_pred","dist_true","dist_pred","margin","closer_to_pred"])
        out.index = pd.Index([], name="cell")
        return out

    # map labels to centroid indices
    label_to_idx = pd.Series(np.arange(len(cent_labels)), index=cent_labels).to_dict()
    # drop rows whose true/pred isn’t in centroid set
    valid_rows = []
    for i, (t, p) in enumerate(zip(y_true[use_mask], y_pred[use_mask])):
        if (t in label_to_idx) and (p in label_to_idx):
            valid_rows.append(True)
        else:
            valid_rows.append(False)
    valid_rows = np.array(valid_rows, dtype=bool)
    if not valid_rows.any():
        out = pd.DataFrame(columns=["y_true","y_pred","dist_true","dist_pred","margin","closer_to_pred"])
        out.index = pd.Index([], name="cell")
        return out

    idx_cells = np.where(use_mask.values)[0][valid_rows]
    y_true_sub = y_true.iloc[use_mask.values][valid_rows]
    y_pred_sub = y_pred.iloc[use_mask.values][valid_rows]

    # 2) compute distances to ALL centroids in batches, then gather needed cols
    X = _get_X(adata, feature_key)
    n_sub = len(idx_cells)
    d_true = np.empty(n_sub, dtype=np.float32)
    d_pred = np.empty(n_sub, dtype=np.float32)

    # class column indices
    t_col = np.array([label_to_idx[v] for v in y_true_sub], dtype=int)
    p_col = np.array([label_to_idx[v] for v in y_pred_sub], dtype=int)

    for start in range(0, n_sub, batch_size):
        end = min(start + batch_size, n_sub)
        rows = idx_cells[start:end]
        D = pairwise_distances(X[rows], C, metric=metric)   # shape (m, n_classes)
        d_true[start:end] = D[np.arange(end-start), t_col[start:end]]
        d_pred[start:end] = D[np.arange(end-start), p_col[start:end]]

    df = pd.DataFrame(
        {
            "y_true": y_true_sub.to_numpy(),
            "y_pred": y_pred_sub.to_numpy(),
            "dist_true": d_true,
            "dist_pred": d_pred,
        },
        index=adata.obs_names[idx_cells],
    )
    df.index.name = "cell"
    df["margin"] = df["dist_true"] - df["dist_pred"]
    df["closer_to_pred"] = df["dist_pred"] < df["dist_true"]
    return df





import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Patch

BAD_ANN_DEFAULT = (np.nan, '', 'unknown', 'None', 'nan', 'NaN', 'NA', 'na', 'unannotated')


# ---------- centroids from annotated labels only ----------
def _centroids_from_truth(adata, true_col, feature_key="X", bad_ann=BAD_ANN_DEFAULT):
    X = _get_X(adata, feature_key)
    y = adata.obs[true_col].copy()
    bad_set = set(bad_ann) - {np.nan}
    good = (~y.isna()) & (~y.isin(bad_set))
    if is_categorical_dtype(y): y = y.astype("category")
    labels = pd.Index(pd.unique(y[good]))
    cents = []
    for lb in labels:
        m = (y == lb).to_numpy()
        mu = X[m].mean(axis=0)
        cents.append(np.asarray(mu).ravel())
    C = np.vstack(cents) if len(cents) else np.zeros((0, X.shape[1]))
    return labels, C

def _count_changed_with_centroids(adata, true_col, pred_col, *, use_mask, labels, label_to_idx,
                                  feature_key="X", metric="cosine"):
    """# of annotated cells (mask) where pred!=true and both centroids exist."""
    y_true = adata.obs[true_col].copy()
    y_pred = adata.obs[pred_col].copy()
    # align for safe comparison
    if is_categorical_dtype(y_true) or is_categorical_dtype(y_pred):
        y_true = y_true.astype("category"); y_pred = y_pred.astype("category")
        cats = y_true.cat.categories.union(y_pred.cat.categories)
        y_true = y_true.cat.set_categories(cats)
        y_pred = y_pred.cat.set_categories(cats)

    changed = use_mask & (y_true != y_pred)
    if not changed.any():
        return 0, 0  # changed with cents, total changed

    t_vals = y_true[changed].to_numpy()
    p_vals = y_pred[changed].to_numpy()
    has_cents = np.array([(t in label_to_idx) and (p in label_to_idx) for t, p in zip(t_vals, p_vals)])
    return int(has_cents.sum()), int(changed.sum())

def _count_distance_consistent(adata, true_col, pred_col, *, use_mask, labels, C, label_to_idx,
                               feature_key="X", metric="cosine", batch_size=50000):
    """Among annotated changed cells with centroids, count how many are closer to predicted centroid."""
    y_true = adata.obs[true_col]; y_pred = adata.obs[pred_col]
    if is_categorical_dtype(y_true) or is_categorical_dtype(y_pred):
        y_true = y_true.astype("category"); y_pred = y_pred.astype("category")
        cats = y_true.cat.categories.union(y_pred.cat.categories)
        y_true = y_true.cat.set_categories(cats); y_pred = y_pred.cat.set_categories(cats)
    changed = use_mask & (y_true != y_pred)
    if not changed.any() or C.shape[0] == 0:
        return 0

    idx_cells = np.where(changed.to_numpy())[0]
    y_t = y_true.iloc[idx_cells].to_numpy()
    y_p = y_pred.iloc[idx_cells].to_numpy()
    keep = np.array([(t in label_to_idx) and (p in label_to_idx) for t, p in zip(y_t, y_p)])
    if not keep.any():
        return 0
    idx_cells = idx_cells[keep]
    t_cols = np.array([label_to_idx[v] for v in y_t[keep]], dtype=int)
    p_cols = np.array([label_to_idx[v] for v in y_p[keep]], dtype=int)

    X = _get_X(adata, feature_key)
    cons = 0
    for start in range(0, len(idx_cells), batch_size):
        end = min(start + batch_size, len(idx_cells))
        rows = idx_cells[start:end]
        D = pairwise_distances(X[rows], C, metric=metric)
        d_true = D[np.arange(end-start), t_cols[start:end]]
        d_pred = D[np.arange(end-start), p_cols[start:end]]
        cons += int((d_pred < d_true).sum())
    return cons

# ---------- overall labeled coverage ----------
def compute_labeled_rate(adata, true_col="ct_or_lin", bad_ann=BAD_ANN_DEFAULT) -> float:
    y = adata.obs[true_col]
    bad_set = set(bad_ann) - {np.nan}
    has_true = (~y.isna()) & (~y.isin(bad_set))
    return float(has_true.mean())

# ---------- summary with 5 components ----------
def summarise_refinement_components_v2(
    adata,
    specs,                        # list of dicts: {"method":..., "pred_col":..., "score_col":..., "score_min":...}
    true_col="ct_or_lin",
    feature_key="X",
    metric="cosine",
    bad_ann=BAD_ANN_DEFAULT
) -> pd.DataFrame:
    """
    Returns per-method fractions (of ALL cells):
      method | labeled_rate | correct_rate | incorrect_consistent_rate |
      incorrect_inconsistent_rate | unlabeled_salvaged_rate | total_bar
    where total_bar = correct + incorrect_consistent + incorrect_inconsistent + unlabeled_salvaged
    """
    n = adata.n_obs
    labeled_rate = compute_labeled_rate(adata, true_col, bad_ann)

    # precompute centroids from annotated labels
    labels, C = _centroids_from_truth(adata, true_col, feature_key, bad_ann)
    label_to_idx = pd.Series(np.arange(len(labels)), index=labels).to_dict()

    y_true_full = adata.obs[true_col].copy()
    bad_set = set(bad_ann) - {np.nan}
    has_true_full = (~y_true_full.isna()) & (~y_true_full.isin(bad_set))

    rows = []
    for spec in specs:
        method    = spec["method"]
        pred_col  = spec["pred_col"]
        score_col = spec.get("score_col")
        score_min = spec.get("score_min", None)

        y_pred_full = adata.obs[pred_col].copy()
        has_pred_full = (~y_pred_full.isna()) & (~y_pred_full.isin(bad_set))

        # optional gating
        valid_score = pd.Series(True, index=adata.obs_names)
        if score_col is not None and score_col in adata.obs.columns and score_min is not None:
            valid_score = pd.Series(adata.obs[score_col] >= score_min, index=adata.obs_names).fillna(False)

        # cells with both labels present and gated
        comparable = has_true_full & has_pred_full & valid_score

        # align to compare equality
        y_true = y_true_full.copy(); y_pred = y_pred_full.copy()
        if is_categorical_dtype(y_true) or is_categorical_dtype(y_pred):
            y_true = y_true.astype("category"); y_pred = y_pred.astype("category")
            cats = y_true.cat.categories.union(y_pred.cat.categories)
            y_true = y_true.cat.set_categories(cats); y_pred = y_pred.cat.set_categories(cats)

        correct_rate = (comparable & (y_true == y_pred)).sum() / n

        # Disagreeing annotated cells
        changed_with_cents, changed_total = _count_changed_with_centroids(
            adata, true_col, pred_col, use_mask=comparable,
            labels=labels, label_to_idx=label_to_idx,
            feature_key=feature_key, metric=metric
        )
        consistent_cnt = _count_distance_consistent(
            adata, true_col, pred_col, use_mask=comparable,
            labels=labels, C=C, label_to_idx=label_to_idx,
            feature_key=feature_key, metric=metric
        )
        inconsistent_cnt = max(changed_with_cents - consistent_cnt, 0)

        incorrect_consistent_rate   = consistent_cnt   / n
        incorrect_inconsistent_rate = inconsistent_cnt / n

        # unlabeled salvaged: no annotation but we got a (gated) prediction
        unlabeled = (~has_true_full) & has_pred_full & valid_score
        unlabeled_salvaged_rate = unlabeled.sum() / n

        rows.append({
            "method": method,
            "labeled_rate": labeled_rate,
            "correct_rate": correct_rate,
            "incorrect_consistent_rate": incorrect_consistent_rate,
            "incorrect_inconsistent_rate": incorrect_inconsistent_rate,
            "unlabeled_salvaged_rate": unlabeled_salvaged_rate,
            "total_bar": correct_rate + incorrect_consistent_rate + incorrect_inconsistent_rate + unlabeled_salvaged_rate,
            # (optional debug)
            # "changed_total_frac": changed_total / n,
            # "changed_with_centroids_frac": changed_with_cents / n,
        })

    df = pd.DataFrame(rows).sort_values("total_bar", ascending=False).reset_index(drop=True)
    return df

# ---------- plotting with 5 components + leftmost labeled bar ----------
def plot_refinement_components_v2(df, *,
                                  figsize=(6.2, 3.2), dpi=300,
                                  annotate=True, legend_right=True,
                                  labeled_color="#7f7f7f",
                                  colors=None):
    """
    df columns:
      method | labeled_rate | correct_rate | incorrect_consistent_rate |
      incorrect_inconsistent_rate | unlabeled_salvaged_rate | total_bar
    """
    if colors is None:
        colors = {
            "Correct": "#1f77b4",                 # blue
            "Incorrect-consistent": "#2ca02c",    # green
            "Incorrect-inconsistent": "#d62728",  # red
            "Unlabeled salvaged": "#9ecae1",      # light blue
        }

    methods = df["method"].tolist()
    m = len(methods)

    x_lbl = np.array([0.0])                 # leftmost single bar
    x_mth = np.arange(1, m + 1, dtype=float)
    width = 0.60

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # leftmost: originally labeled
    b0 = ax.bar(x_lbl, [df["labeled_rate"].iloc[0] if m > 0 else 0.0],
                width=width, color=labeled_color, label="Originally labeled")

    # per-method stacked bars (4 segments)
    bottom = np.zeros(m, dtype=float)
    segs = [
        ("correct_rate",               colors["Correct"],                 "Predicted correct"),
        ("incorrect_consistent_rate",  colors["Incorrect-consistent"],    "Pred. incorrect (distance-consistent)"),
        ("incorrect_inconsistent_rate",colors["Incorrect-inconsistent"],  "Pred. incorrect (distance-inconsistent)"),
        ("unlabeled_salvaged_rate",    colors["Unlabeled salvaged"],      "Unlabeled salvaged"),
    ]
    bars = []
    for col, col_color, label in segs:
        h = df[col].to_numpy()
        b = ax.bar(x_mth, h, bottom=bottom, width=width, color=col_color, label=label)
        bars.append((col, b))
        bottom += h

    # annotations
    if annotate:
        # totals on top of method bars
        for xi, h in zip(x_mth, df["total_bar"].to_numpy()):
            if h > 0:
                ax.text(xi, h + 0.012, f"{h*100:.0f}%", ha="center", va="bottom", fontsize=8)
        # sizable segment labels inside
        for _, b in bars:
            for rect in b:
                h = rect.get_height()
                if h >= 0.03:
                    ax.text(rect.get_x() + rect.get_width()/2,
                            rect.get_y() + h/2,
                            f"{h*100:.0f}%",
                            ha="center", va="center", fontsize=7, color="black")
        # labeled bar top
        lbl_h = float(b0[0].get_height()) if len(b0) else 0.0
        if lbl_h > 0:
            ax.text(x_lbl[0], lbl_h + 0.012, f"{lbl_h*100:.0f}%", ha="center", va="bottom", fontsize=8)

    # axes cosmetics
    xticks = np.concatenate([x_lbl, x_mth])
    xticklabels = ["Labeled"] + methods
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=0, ha="center")

    ax.set_ylabel("Fraction of all cells")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
    for s in ("top","right"): ax.spines[s].set_visible(False)

    ymax = max(float(df["total_bar"].max()) if len(df) else 0.0,
               float(df["labeled_rate"].iloc[0]) if len(df) else 0.0)
    ax.set_ylim(0, min(1.0, ymax * 1.12 if ymax > 0 else 0.15))
    ax.set_title("Outcomes vs. ground truth & salvage", pad=8)

    # legend to the right
    handles = [
        Patch(facecolor=labeled_color, label="Originally labeled"),
        Patch(facecolor=colors["Correct"], label="Predicted correct"),
        Patch(facecolor=colors["Incorrect-consistent"], label="Pred. incorrect (distance-consistent)"),
        Patch(facecolor=colors["Incorrect-inconsistent"], label="Pred. incorrect (distance-inconsistent)"),
        Patch(facecolor=colors["Unlabeled salvaged"], label="Unlabeled salvaged"),
    ]
    if legend_right:
        ax.legend(handles=handles, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
        fig.tight_layout(rect=[0, 0, 0.80, 1])
    else:
        ax.legend(handles=handles, frameon=False)
        fig.tight_layout()

    return fig, ax








import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from pandas.api.types import is_categorical_dtype

BAD_ANN = (np.nan, '', 'unknown', 'None', 'nan', 'NaN', 'NA', 'na', 'unannotated')

def _centroids_from_truth(adata, true_col, feature_key="X", bad_ann=BAD_ANN):
    X = _get_X(adata, feature_key)
    y = adata.obs[true_col].copy()
    bad_set = set(bad_ann) - {np.nan}
    good = (~y.isna()) & (~y.isin(bad_set))
    labels = pd.Index(pd.unique(y[good]))
    cents = []
    for lb in labels:
        m = (y == lb).to_numpy()
        mu = X[m].mean(axis=0)
        cents.append(np.asarray(mu).ravel())
    C = np.vstack(cents) if len(cents) else np.zeros((0, X.shape[1]))
    return labels, C  # row order aligns with `labels`

def cells_needing_correction(
    adata,
    subset_index,                # e.g. adata_subsub.obs.index
    true_col="lin_or_ct",
    pred_col="concord_hcl_knn_refined_ct_or_lin",
    feature_key="X",
    metric="cosine",
    bad_ann=BAD_ANN,
    batch_size=50000,
):
    # align categories so (y_true!=y_pred) works
    y_true = adata.obs[true_col].copy()
    y_pred = adata.obs[pred_col].copy()
    if is_categorical_dtype(y_true) or is_categorical_dtype(y_pred):
        y_true = y_true.astype("category"); y_pred = y_pred.astype("category")
        cats = y_true.cat.categories.union(y_pred.cat.categories)
        y_true = y_true.cat.set_categories(cats)
        y_pred = y_pred.cat.set_categories(cats)

    bad_set = set(bad_ann) - {np.nan}
    has_true = (~y_true.isna()) & (~y_true.isin(bad_set))
    has_pred = (~y_pred.isna()) & (~y_pred.isin(bad_set))

    # only consider your sub-subset
    in_subset = adata.obs_names.isin(subset_index)
    comparable = in_subset & has_true & has_pred
    changed = comparable & (y_true != y_pred)
    if not changed.any():
        return pd.Index([]), pd.DataFrame(columns=["dist_true","dist_pred","margin"])

    # centroids from annotated labels only
    labels, C = _centroids_from_truth(adata, true_col, feature_key, bad_ann)
    if C.shape[0] == 0:
        return pd.Index([]), pd.DataFrame(columns=["dist_true","dist_pred","margin"])

    label_to_idx = pd.Series(np.arange(len(labels)), index=labels).to_dict()

    idx_cells = np.where(changed.values)[0]
    t_vals = y_true.iloc[changed.values].to_numpy()
    p_vals = y_pred.iloc[changed.values].to_numpy()

    keep_mask = np.array([(t in label_to_idx) and (p in label_to_idx) for t, p in zip(t_vals, p_vals)])
    if not keep_mask.any():
        return pd.Index([]), pd.DataFrame(columns=["dist_true","dist_pred","margin"])

    idx_cells = idx_cells[keep_mask]
    t_cols = np.array([label_to_idx[v] for v in t_vals[keep_mask]], dtype=int)
    p_cols = np.array([label_to_idx[v] for v in p_vals[keep_mask]], dtype=int)

    X = _get_X(adata, feature_key)
    d_true = np.empty(len(idx_cells), dtype=np.float32)
    d_pred = np.empty(len(idx_cells), dtype=np.float32)

    for start in range(0, len(idx_cells), batch_size):
        end = min(start + batch_size, len(idx_cells))
        rows = idx_cells[start:end]
        D = pairwise_distances(X[rows], C, metric=metric)  # (m, n_classes)
        d_true[start:end] = D[np.arange(end-start), t_cols[start:end]]
        d_pred[start:end] = D[np.arange(end-start), p_cols[start:end]]

    margin = d_true - d_pred
    need_fix = margin > 0  # closer to predicted than to annotated

    corrected_idx = adata.obs_names[idx_cells[need_fix]]
    dist_df = pd.DataFrame(
        {"dist_true": d_true, "dist_pred": d_pred, "margin": margin},
        index=adata.obs_names[idx_cells]
    )
    return corrected_idx, dist_df
