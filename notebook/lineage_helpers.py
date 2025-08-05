

"""
lineage_helpers.py
~~~~~~~~~~~~~~~~~~

Utilities for

1.  Loading a *C. elegans* lineage‐edge table (from ➜ “from”, “to” columns)
2.  Reconciling fuzzy lineage / cell-type labels in an AnnData object
3.  Building a NetworkX DiGraph whose nodes carry the mapping results
"""

from __future__ import annotations

from pathlib import Path
import itertools
import pandas as pd
import numpy as np
import networkx as nx
from typing import Iterable, Mapping, Sequence, Any

SPECIAL_MAPPINGS = {
    # Examples you mentioned
    "OLQ": ["OLQDL", "OLQDR", "OLQVL", "OLQVR"],
    "OLL": ["OLLL", "OLLR"],
    "ILsh_OLLsh_OLQsh":["ILshDL","ILshDR", "ILshL", "ILshR", "ILshVL", "ILshVR", "OLLshL", "OLLshR",  "OLQshDL", "OLQshSR", "OLQshVL", "OLQshVR"],
    'ADE': ['ADEL', 'ADER'],
    "BWM_headrow1_in": [],
    "BWM_headrow2_in": [],
    "M_cell": ["M"],
    "P_cells": [],
    "B_F_K_Kp_U_Y": ["B", "F", "K", "Kp", "U", "Y"],
    "Seam_hyp_early_cells": [],
    "hyp7_AB_lineage": ["hyp7"],
    "hyp7_C_lineage": ["hyp7"],
    "hyp4_hyp5_hyp6": ["hyp4", "hyp5", "hyp6"],
    "mu_int_mu_anal": ["mu_int_L", "mu_int_R", "mu_anal"],
    "mu_int_mu_anal_related": [],
    # pm3_pm4_pm5c => explicit expansions
    "pm3_pm4_pm5c": [
        "pm3DL", "pm3DR", "pm3L", "pm3R", "pm3VL", "pm3VR",
        "pm4DL", "pm4DR", "pm4L", "pm4R", "pm4VL", "pm4VR",
        "pm5DL", "pm5DR", "pm5L", "pm5R", "pm5VL", "pm5VR"
    ],
    "pm3_pm4_pm5b": [
        "pm3DL", "pm3DR", "pm3L", "pm3R", "pm3VL", "pm3VR",
        "pm4DL", "pm4DR", "pm4L", "pm4R", "pm4VL", "pm4VR",
        "pm5DL", "pm5DR", "pm5L", "pm5R", "pm5VL", "pm5VR"
    ],
    "Coelomocytes": ['ccAL', 'ccAR', 'ccPL', 'ccPR'],
    #"early_arcade_cell": ['arc ant DL', 'arc ant DR', 'arc ant V', 'arc post DL', 'arc post DR', 'arc post V', 'arc post VR', 'arc post VL'],
    'mu_sph': ['mu_sph'],
    'Seam_cells': [],
    'Seam_cells_early': ['HOL', 'HOR', 'H1L', 'H1R', 'H2L', 'H2R', 'V1L', 'V1R', 'V2L', 'V2R', 'V3L', 'V3R', 'V4L', 'V4R', 'V5L', 'V5R', 'V6L', 'V6R'],
    'mc2b': ['mc2DL', 'mc2DR', 'mc2V'],
    'mc2a': ['mc2DL', 'mc2DR', 'mc2V'],
    'Tail_hypodermis': ['hyp8/9', 'hyp10'],
    'Rectal_gland': ['rect_D', 'rect_VL', 'rect_VR'],
    'Anterior_arcade_cell': ['arc ant DL', 'arc ant DR', 'arc ant V'],
    'Posterior_arcade_cell': ['arc post D', 'arc post DL', 'arc post DR', 'arc post V', 'arc post VR', 'arc post VL'],
    'Pharyngeal_intestinal_valve': ['vpi1', 'vpi2DL', 'vpi2DR', 'vpi2V', 'vpi3D', 'vpi3V'],
    'hyp1V_and_ant_arc_V': ['ant arc V'],
    'hyp1V': ['ant arc V'],
    'Excretory_cell': ['exc_cell'],
    'Excretory_duct_and_pore': ['exc_duct'],
    'Excretory_gland': ['exc_gl_L', 'exc_gl_R'],
    'mu_int_mu_anal_related': [],
    'CEP': ['CEPDL', 'CEPDR', 'CEPVL', 'CEPVR'],
    'Arcade_cell': ['arc ant DL', 'arc ant DR', 'arc ant V', 'arc post D', 'arc post DL', 'arc post DR', 'arc post V', 'arc post VR', 'arc post VL'],

    # etc. Add more as needed...
}


BROAD_LINEAGE_MAP = {
    'Cxa':   ['Cxa', 'Cpa', 'Caa'],
    'Cxp':   ['Cxp', 'Cpp', 'Cap'],
    'D':     ['D'],
    'E':     ['E'],
    'MSxpa': ['MSxpa', 'MSapa', 'MSppa'],
    'MSxaa': ['MSxaa', 'MSpaa', 'MSaaa'],
    'MSxap': ['MSxap', 'MSpap', 'MSaap'],
    'MSxpp': ['MSxpp', 'MSppp', 'MSapp'],
    'ABala': ['ABala'],
    'ABalp': ['ABalp'],
    'ABara': ['ABara'],
    'ABarp': ['ABarp'],
    'ABpla': ['ABpla'],
    'ABplp': ['ABplp'],
    'ABpra': ['ABpra'],
    'ABprp': ['ABprp'],
    'Z2/Z3': ['Z2', 'Z3'],
}


# ------------------------------------------------------------------------------
# 2) Optionally define a synonym/prefix dictionary for partial expansions
#    This can handle simpler patterns like "AWB" => search for "AWB" in canonical cells
#    or "BWM" => "mu_bod", etc.
# ------------------------------------------------------------------------------
SYNONYM_PREFIXES = {
    #"BWM": "mu_bod",  # if you want "BWM_..." => "mu_bod"
}



# -----------------------------------------------------------------------------#
# 1.  Tiny helpers
# -----------------------------------------------------------------------------#

def expand_x(name: str,
             alphabet: Sequence[str] = ("a", "p", "r", "l", "d", "v")
             ) -> list[str]:
    """All strings obtained by replacing every 'x' with letters in *alphabet*."""
    if "x" not in name:
        return [name]

    pools: list[Sequence[str]] = [(alphabet if ch == "x" else (ch,))
                                  for ch in name]
    return ["".join(chars) for chars in itertools.product(*pools)]


def map_lineage_name(query: str,
                     canonical: set[str]) -> list[str]:
    """
    Resolve a possibly wildcard / slash-delimited lineage *query* against
    *canonical* names taken from the lineage table (“to” column).
    """
    matches: set[str] = set()
    for token in query.split("/"):
        for guess in expand_x(token):
            if guess in canonical:
                matches.add(guess)
    return sorted(matches)


def _has_content(x) -> bool:
    if x is None:
        return False
    if isinstance(x, float) and np.isnan(x):
        return False
    if isinstance(x, (list, tuple, set)) and len(x) == 0:
        return False
    return True


# -----------------------------------------------------------------------------#
# 2.  Canonical-name lookup that handles your “special cases”
# -----------------------------------------------------------------------------#

def make_canonical_lookup(canonical_cells: Iterable[str],
                          special: Mapping[str, Sequence[str]] | None = None,
                          prefixes: Mapping[str, str] | None = None):
    """
    Return a single function *lookup(name) -> list[str]* that implements:

    • direct hits
    • slash expansion (union of results for each side)
    • SPECIAL_MAPPINGS overrides
    • synonym / prefix fall-back search
    """
    canonical_list = [str(c) for c in canonical_cells if str(c).lower() != "nan"]
    canonical_set  = set(canonical_list)
    special = special or {}
    prefixes = prefixes or {}

    def lookup(name: str) -> list[str]:
        name = (name or "").strip()
        if not name or name.lower() == "nan":
            return []

        # 1) hard-coded overrides
        if name in special:
            return list(special[name])

        # 2) slash logic
        if "/" in name:
            hits = set()
            for part in name.split("/"):
                hits.update(lookup(part))
            return sorted(hits)

        # 3) prefix / synonym logic
        hits: set[str] = set()
        for token in name.split("_"):
            token = prefixes.get(token, token)
            if token in canonical_set:
                hits.add(token)
            else:
                hits.update([c for c in canonical_list if c.startswith(token)])
        if hits:
            return sorted(hits)

        # 4) final fall-back
        return [name] if name in canonical_set else []

    return lookup




# lineage_helpers.py
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

# ----------------------------------------------------------------------
# 1)  broad-group assignment
# ----------------------------------------------------------------------
def assign_broad_groups(
    G: nx.DiGraph,
    broad_map: Mapping[str, Sequence[str]] | None = None  # ← default None
) -> dict:
    """
    Add a 'broad_group' attribute to every node and return {node: group_or_None}.
    """
    if broad_map is None:
        broad_map = BROAD_LINEAGE_MAP

    def _match(node: str) -> str | None:
        for grp, prefixes in broad_map.items():
            if any(node.startswith(p) for p in prefixes):
                return grp
        return None

    node2group = {}
    for n in G.nodes():
        grp = _match(n)
        G.nodes[n]["broad_group"] = grp
        node2group[n] = grp
    return node2group


# ----------------------------------------------------------------------
# 2)  quick plot
# ----------------------------------------------------------------------
# --------------------------------------------------------------------------- #
#  plot_lineage_graph   (palette-aware)
# --------------------------------------------------------------------------- #
def plot_lineage_graph(
    G: nx.DiGraph,
    node2group: dict[str, str | None],
    *,
    figsize: tuple[int, int] = (12, 2),
    palette: str | Sequence[str] | dict[str, str] = "tab20",
    dpi: int = 600,
    save_path: str | Path | None = None,
):
    """
    Visualise the lineage tree coloured by *node2group*.

    Parameters
    ----------
    palette
        • str         → any Matplotlib colormap name  
        • sequence    → list/tuple of colour hex-codes / names  
        • dict        → explicit ``{group: colour}`` mapping
    """

    # 1 ── determine palette --------------------------------------------------
    groups = sorted({g for g in node2group.values() if g is not None})

    if isinstance(palette, dict):
        group2colour = palette.copy()
        # auto-assign colours for any group not in the dict
        missing = [g for g in groups if g not in group2colour]
        if missing:
            cmap = cm.get_cmap("tab20", len(missing))
            group2colour.update({g: cmap(i) for i, g in enumerate(missing)})
    else:
        # palette is str or sequence  → build a colormap iterator
        if isinstance(palette, str):
            cmap = cm.get_cmap(palette, len(groups))
            colours = [cmap(i) for i in range(len(groups))]
        else:                           # assume list / tuple of colours
            colours = list(palette)
            if len(colours) < len(groups):
                # repeat colours if too few
                colours *= int(np.ceil(len(groups) / len(colours)))
        group2colour = dict(zip(groups, colours))

    # 2 ── node colours -------------------------------------------------------
    node_colours = []
    for n in G.nodes():
        grp = node2group.get(n)
        mapped = G.nodes[n].get("mapped", False)
        if grp is not None and mapped:
            node_colours.append(group2colour.get(grp, "lightgrey"))
        else:
            node_colours.append("black" if mapped else "lightgrey")

    # 3 ── layout + draw ------------------------------------------------------
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")   # needs pygraphviz
    plt.figure(figsize=figsize, dpi=dpi)
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=20,
        font_size=4,
        arrowsize=4,
        width=0.4,
        node_color=node_colours,
    )
    plt.title("Lineage tree with broad-group colouring")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()



# -----------------------------------------------------------------------------#
# 3.  All-in-one front end
# -----------------------------------------------------------------------------#

def build_lineage_graph(
    adata,
    tbl_df: pd.DataFrame | str | Path,                 # <-- now dataframe-first
    *,
    special_mappings: Mapping[str, Sequence[str]] | None = None,
    synonym_prefixes: Mapping[str, str] | None = None,
    lineage_col: str = "lineage",
    celltype_col: str = "plot.cell.type",
    broad_lineage_groups: dict | None = None,
    add_broad_group: bool = True,
    plot: bool = False,
    plot_path=None,
):
    """
    Parameters
    ----------
    adata : AnnData
        Your annotated data object.
    tbl_df : pandas.DataFrame  OR  str/Path
        The lineage edge list.  If a path is given, it will be read with
        ``pd.read_csv``.

    All other arguments behave as before.

    Returns
    -------
    G : networkx.DiGraph
    tbl : pandas.DataFrame
        Copy of *tbl_df* with extra mapping columns.
    """

    # -- 0.  Load table if the caller accidentally gives us a file ------------
    if isinstance(tbl_df, (str, Path)):
        tbl_df = pd.read_csv(tbl_df, index_col=0)

    # -- 1.  Fall back on module-level defaults if None -----------------------
    if special_mappings is None:
        special_mappings = SPECIAL_MAPPINGS
    if synonym_prefixes is None:
        synonym_prefixes = SYNONYM_PREFIXES

    # --------------  everything below here is IDENTICAL to the prior version --
    canonical_lineage = set(tbl_df["to"].astype(str).values)
    canonical_cells   = tbl_df["Cell"].astype(str).values

    canonical_lookup = make_canonical_lookup(
        canonical_cells,
        special=special_mappings,
        prefixes=synonym_prefixes,
    )

    fuzzy_vals = adata.obs[celltype_col].unique()
    celltype_to_cellname = {v: canonical_lookup(str(v)) for v in fuzzy_vals}

    cell_to_lin = dict(zip(tbl_df["Cell"], tbl_df["to"]))
    celltype_to_lineage = {
        k: [cell_to_lin[c] for c in v if c in cell_to_lin]
        for k, v in celltype_to_cellname.items()
    }

    unique_linannot = adata.obs[lineage_col].unique()
    linannot_to_actual = {
        ln: map_lineage_name(str(ln), canonical_lineage) for ln in unique_linannot
    }
    lin_actual_to_annot = {}
    for annot, actuals in linannot_to_actual.items():
        for act in actuals:
            lin_actual_to_annot.setdefault(act, []).append(annot)

    tbl = tbl_df.copy()
    tbl["celltype_annot"]  = tbl["to"].map(
        lambda ln: [ct for ct, lst in celltype_to_lineage.items() if ln in lst]
    )
    tbl["lineage_annot"]  = tbl["to"].map(lin_actual_to_annot)

    tbl["lin_or_ct_annot"] = tbl.apply(
        lambda r: r["lineage_annot"] if _has_content(r["lineage_annot"])
        else r["celltype_annot"],
        axis=1,
    )

    tbl["mapped"] = tbl["lin_or_ct_annot"].apply(_has_content)

    G = nx.DiGraph()
    for _, row in tbl.iterrows():
        parent, child = row["from"], row["to"]
        if pd.isna(parent) or pd.isna(child):
            continue
        if not G.has_node(parent):
            G.add_node(parent, mapped=False)

        G.add_node(child,
                   mapped=row["mapped"],
                   celltype_annot=row["celltype_annot"],
                   lineage_annot=row["lineage_annot"],
                   linorct=row["lin_or_ct_annot"])

        G.add_edge(parent, child)

    node2group = {}
    if add_broad_group:
        node2group = assign_broad_groups(G, broad_lineage_groups)

    if plot and node2group:
        plot_lineage_graph(G, node2group, save_path=plot_path)

    return G, tbl, node2group


# -----------------------------------------------------------------------------#
# 4. Convenience wrapper (minor tweak)
# -----------------------------------------------------------------------------#
def lineage_mappings(adata, tbl_df, **kwargs):
    return build_lineage_graph(adata, tbl_df, **kwargs)






import pandas as pd
import re
from anndata import AnnData
from typing import Mapping, Sequence

# ---------------------------------------------------------------------
# configurable mapping (re-uses the one already defined in the module)
# ---------------------------------------------------------------------
BROAD_LINEAGE_PREFIXES: dict[str, list[str]] = {
    "Cxa":   ["Cxa", "Cpa", "Caa"],
    "Cxp":   ["Cxp", "Cpp", "Cap"],
    "D":     ["D"],
    "E":     ["E"],
    "MSxpa": ["MSxpa", "MSapa", "MSppa"],
    "MSxaa": ["MSxaa", "MSpaa", "MSaaa"],
    "MSxap": ["MSxap", "MSpap", "MSaap"],
    "MSxpp": ["MSxpp", "MSppp", "MSapp"],
    "ABala": ["ABala"],
    "ABalp": ["ABalp"],
    "ABara": ["ABara"],
    "ABarp": ["ABarp"],
    "ABpla": ["ABpla"],
    "ABplp": ["ABplp"],
    "ABpra": ["ABpra"],
    "ABprp": ["ABprp"],
    "Z2/Z3": ["Z2", "Z3"],
}

DROP_LINEAGES = {"unass", "nan"}
EARLY_LABELS  = {"AB early", "MS early", "Cx", "28_ce", "possi"}

# ---------------------------------------------------------------------
def add_broad_lineage(
    adata: AnnData,
    *,
    lineage_key: str = "lineage",
    new_key: str = "broad_lineage",
    prefix_map: Mapping[str, Sequence[str]] | None = None,
) -> None:
    """
    Derive a coarse 'broad_lineage' assignment in `adata.obs`.

    The procedure replicates the sequential replacements you wrote,
    but uses vectorised Pandas operations to avoid SettingWithCopy
    warnings and to stay maintainable.

    Parameters
    ----------
    adata : AnnData
        Your single-cell object (will be modified in-place).
    lineage_key : str
        Column in `adata.obs` holding the full lineage string.
    new_key : str
        Name of the created column.
    prefix_map : dict[str, list[str]] | None
        Custom prefix-to-group mapping; falls back to
        `BROAD_LINEAGE_PREFIXES` if None.
    """
    if prefix_map is None:
        prefix_map = BROAD_LINEAGE_PREFIXES

    # -----------------------------------------------------------------
    # 1. Initialise with first five characters of the full lineage
    # -----------------------------------------------------------------
    broad = adata.obs[lineage_key].astype(str).str.slice(0, 5)

    # -----------------------------------------------------------------
    # 2. Apply explicit prefix mappings
    # -----------------------------------------------------------------
    for group, prefixes in prefix_map.items():
        pattern = r"^(?:{})".format("|".join(map(re.escape, prefixes)))
        mask = broad.str.match(pattern)
        broad.loc[mask] = group

    # -----------------------------------------------------------------
    # 3. Early-stage heuristics
    # -----------------------------------------------------------------
    #   • AB* with <5 chars   → AB early
    #   • MS* with <5 chars   → MS early
    # -----------------------------------------------------------------
    mask_ab = broad.str.startswith("AB") & (broad.str.len() < 5)
    mask_ms = broad.str.startswith("MS") & (broad.str.len() < 5)
    broad.loc[mask_ab] = "AB early"
    broad.loc[mask_ms] = "MS early"

    # Consolidate early variants
    broad.replace(dict.fromkeys(EARLY_LABELS, "early cells"), inplace=True)

    # -----------------------------------------------------------------
    # 4. Hard drops / NaNs
    # -----------------------------------------------------------------
    broad = broad.where(~broad.isin(DROP_LINEAGES), pd.NA)

    # -----------------------------------------------------------------
    # 5. Attach to AnnData
    # -----------------------------------------------------------------
    adata.obs[new_key] = pd.Categorical(broad)

    # Optional: return the series if a caller wants it
    return adata.obs[new_key]




def get_representative_point(coords, method='medoid', max_n_medoid=2000,
                             k_top=3, seed=0, jitter=0, return_idx=False):
    """
    Return a single (x, y) representing these coords. 
    - If 'medoid' and len(coords)<=max_n_medoid, we pick one
      randomly from the top k_top best medoid candidates.
    - Otherwise, nearest to centroid.
    """
    import random
    from scipy.spatial.distance import cdist
    import numpy as np
    random.seed(seed)  # for reproducibility
    n = len(coords)
    if n == 0:
        return np.array([np.nan, np.nan])
    if method == 'medoid' and n <= max_n_medoid:
        dist_mat = cdist(coords, coords)  # shape (n,n)
        sum_dists = dist_mat.sum(axis=1)
        sorted_indices = np.argsort(sum_dists)
        if k_top>n: 
            k_top=n
        best_indices = sorted_indices[:k_top]
        chosen_idx = random.choice(best_indices)
        coord = coords[chosen_idx]
    else:
        # fallback
        centroid = coords.mean(axis=0)
        dists = np.linalg.norm(coords - centroid, axis=1)
        chosen_idx = np.argmin(dists)
        coord = coords[chosen_idx]
    
    # Add jitter for visualization
# lineage_helpers.py  inside get_representative_point
    if jitter > 0:
        jittered_coord = coord + jitter * np.random.randn(coord.shape[-1])
    else:
        jittered_coord = coord

    if return_idx:
        return jittered_coord, chosen_idx
    else:
        return jittered_coord
    

# --------------------------------------------------------------------------
# 2.  assign every node a list of cell indices
# --------------------------------------------------------------------------
import pandas as pd
import networkx as nx
from anndata import AnnData
from typing import Dict, Sequence, Optional

# lineage_helpers.py  (drop this near the top of the file)
import math
import pandas as pd
from typing import Sequence, Any, List

def _as_clean_list(x: Any) -> List[str]:
    """
    Return a list of string annotations.

    • None / NaN / empty → []
    • single string      → [string]
    • list/tuple/set     → list(with NaNs removed)
    • anything else      → []
    """
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return []

    if isinstance(x, str):
        return [] if x.lower() == "nan" else [x]

    if isinstance(x, (list, tuple, set, pd.Series)):
        return [str(v) for v in x if not (isinstance(v, float) and math.isnan(v))]

    return []            # fallback for strange types




def assign_cells_to_tree(
    G: nx.DiGraph,
    adata: AnnData,
    *,
    lineage_key: str = "lineage",
    celltype_key: str = "plot.cell.type",
    prefer: str = "lineage",        # "lineage" | "celltype" | "auto"
) -> Dict[str, Sequence[int]]:
    """
    Attach `cells` (list[int]) to each node of *G* using annotations stored
    on the node:

        • first try node['lineage_annot']  (list of lineage strings)
        • otherwise node['celltype_annot'] (list of cell-type strings)

    Parameters
    ----------
    prefer : "lineage"|"celltype"|"auto"
        * "lineage"  – always try lineage first, then fall back to cell-type.
        * "celltype" – the opposite.
        * "auto"     – whichever annotation list on the node is non-empty.

    Returns
    -------
    node_to_cells : dict[node, list[int]]
        Also written in-place into `G.nodes[n]["cells"]`.
    """
    # pre-compute index dictionaries for vector-fast lookup
    lineage_to_cells = adata.obs.groupby(lineage_key).indices
    celltype_to_cells = adata.obs.groupby(celltype_key).indices

    node_to_cells: Dict[str, Sequence[int]] = {}

    for n, data in G.nodes(data=True):
        lin_annot = _as_clean_list(data.get("lineage_annot"))
        ct_annot  = _as_clean_list(data.get("celltype_annot"))
        chosen: Optional[Sequence[int]] = []

        # decide which list to try first
        first, second = ("lineage", "celltype") if prefer == "lineage" else ("celltype", "lineage")
        if prefer == "auto":
            first, second = ("lineage", "celltype") if lin_annot else ("celltype", "lineage")

        for mode in (first, second):
            annots = lin_annot if mode == "lineage" else ct_annot
            for a in annots:
                # extend keeps duplicates out by casting to set later
                if mode == "lineage":
                    idxs = list(lineage_to_cells.get(a, []))
                else:
                    idxs = list(celltype_to_cells.get(a, []))

                chosen.extend(idxs)    
            if chosen:      # stop if we already found cells
                break

        chosen = sorted(set(chosen))            # uniqueness + reproducible order
        if chosen:
            G.nodes[n]["cells"] = chosen
            node_to_cells[n] = chosen

    return node_to_cells

from scipy.spatial.distance import cdist
import numpy as np

def compute_node_representatives(
    G,
    adata,
    *,
    emb_key: str,
    method: str = "mean",            # "mean"|"median"|"medoid"|…
    rep_attr: str | None = None,
    store_index_for_medoid: bool = False,   # <─ new toggle
    G_lineage_key: str = "lineage_annot",
    G_celltype_key: str = "celltype_annot",
    adata_lineage_key: str = "lineage_complete",
    adata_celltype_key: str = "cell_type",
    seed: int | None = None,
    max_n_medoid: int = 2_000,
    k_top: int = 3,
    jitter: float = 0.0,
):
    """
    Attach a representative **vector** (default) to every lineage node.

    • For "medoid" you now get the coordinate *unless*
      ``store_index_for_medoid=True`` (back-compat mode).

    Returns
    -------
    int
        Number of nodes that received a representative.
    """
    if rep_attr is None:
        rep_attr = f"{emb_key}_{method}"

    n_done = 0
    X = adata.obsm[emb_key]          # cached pointer

    for node, attrs in G.nodes(data=True):
        # --- choose cells for this lineage node -----------------------------
        lin = parse_annotation(attrs.get(G_lineage_key))
        ct  = parse_annotation(attrs.get(G_celltype_key))
        if lin:
            mask = adata.obs[adata_lineage_key].isin(lin)
        elif ct:
            mask = adata.obs[adata_celltype_key].isin(ct)
        else:
            G.nodes[node][rep_attr] = np.nan
            continue

        idx = np.where(mask)[0]
        if idx.size == 0:
            G.nodes[node][rep_attr] = np.nan
            continue

        coords = X[idx]

        # --- compute representative -----------------------------------------
        m = method.lower()
        if m in {"mean", "centroid"}:
            rp = get_representative_point(
                coords,
                method="mean",
                max_n_medoid=max_n_medoid,
                k_top=k_top,
                jitter=jitter,
                seed=seed,
            )
        elif m == "medoid":
            rp_coord = get_representative_point(
                coords, method="medoid",
                max_n_medoid=max_n_medoid, k_top=k_top,
                jitter=jitter, seed=seed,
            )
            if store_index_for_medoid:
                # save the *row index* (old behaviour)
                medoid_local = np.argmin(((coords - rp_coord) ** 2).sum(axis=1))
                rp = int(idx[medoid_local])
            else:
                rp = rp_coord                  # save vector (new default)
        else:                                  # any exotic method delegated
            rp = get_representative_point(
                coords, method=method,
                max_n_medoid=max_n_medoid, k_top=k_top,
                jitter=jitter, seed=seed,
            )

        G.nodes[node][rep_attr] = rp
        n_done += 1

    return n_done


# --------------------------------------------------------------------------- #
#  geodesic_distance_matrix  – k-NN graph distances between lineage nodes
# --------------------------------------------------------------------------- #
def geodesic_distance_matrix(
    G,
    adata,
    *,
    conn_key: str | None = None,
    rep_key: str | None = None,      # passed to sc.pp.neighbors if graph missing
    rep_attr: str = "medoid",        # node attribute holding *cell index*
    n_neighbors: int = 30,
    directed: bool = False,
    hop_weight: bool = False,
    return_labels: bool = True,
):
    """
    Geodesic (or hop-count) distances between lineage nodes on a k-NN graph.

    The attribute ``G.nodes[n][rep_attr]`` **must be an integer cell-index**
    (as produced by ``compute_node_representatives(..., method="medoid",
    store_index_for_medoid=True)``).

    Parameters
    ----------
    conn_key
        Name of the connectivity matrix already in ``adata.obsp``.  If ``None``,
        the function will (re-)run :pyfunc:`scanpy.pp.neighbors` with
        ``use_rep=rep_key`` and ``n_neighbors``.
    hop_weight
        If *True* every edge weight is set to 1.0 so the result is an
        unweighted hop count.

    Returns
    -------
    D : ndarray (L × L)
    lin_nodes : list[str]   (only when *return_labels* is True)
    """
    import scanpy as sc
    import numpy as np
    import scipy.sparse.csgraph as csgraph
    
    # ------------------------------------------------------------------ 1 · graph
    if conn_key is None:
        tmp_key = f"_nn_{n_neighbors}"
        if tmp_key not in adata.uns:
            sc.pp.neighbors(
                adata,
                n_neighbors=n_neighbors,
                use_rep=rep_key,      # None → falls back to adata.X
                key_added=tmp_key,
            )
        conn_key = adata.uns[tmp_key]["connectivities_key"]

    G_knn = adata.obsp[conn_key].tocsr()

    if hop_weight:
        G_knn.data[:] = 1.0          # all edges length 1
    if not directed:
        G_knn = G_knn.maximum(G_knn.T)   # force symmetry

    # ------------------------------------------------------------------ 2 · lineage nodes
    lin_nodes, cell_idx = [], []
    for n, data in G.nodes(data=True):
        val = data.get(rep_attr, None)
        if val is None or np.isnan(val):
            continue
        if not (np.isscalar(val) and np.equal(np.mod(val, 1), 0)):
            raise ValueError(
                f"Node {n!r} has non-integer '{rep_attr}' – "
                "run compute_node_representatives(..., store_index_for_medoid=True)"
            )
        lin_nodes.append(n)
        cell_idx.append(int(val))

    if not lin_nodes:
        raise ValueError(
            f"No nodes carry an integer '{rep_attr}' attribute. "
            "Did you compute medoid indices?"
        )

    # ------------------------------------------------------------------ 3 · Dijkstra
    dist_src_all = csgraph.dijkstra(
        G_knn,
        directed=directed,
        indices=np.asarray(cell_idx, dtype=int),
    )

    D = dist_src_all[:, cell_idx]
    D[np.isinf(D)] = np.nan          # unreachable → NaN

    return (D, lin_nodes) if return_labels else D



def embedding_distance_matrix(
    G,
    adata,
    *,
    emb_key: str,
    metric: str = "euclidean",            # "euclidean" | "cosine" | …
    rep_attr: str = "medoid",             # which node attribute to use
    return_labels: bool = True,
):
    """
    Pair-wise distances **between lineage nodes** based on the data stored in
    ``G.nodes[n][rep_attr]``.

    • If that attribute is an **int / np.integer** ⇒ treated as a cell index  
      → coordinates are taken from ``adata.obsm[emb_key]``.

    • Otherwise it is expected to be a length-d vector already in the same
      embedding space.

    Returns
    -------
    D : ndarray (L × L)
    lin_nodes : list[str]    (only when *return_labels* is True)
    """
    lin_nodes, vectors, idx_int = [], [], []

    for n, data in G.nodes(data=True):
        val = data.get(rep_attr, None)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        if np.isscalar(val) and np.equal(np.mod(val, 1), 0):  # integer index
            idx_int.append(int(val))
            lin_nodes.append(n)
        else:
            vectors.append(np.asarray(val, dtype=float))
            lin_nodes.append(n)

    if vectors and idx_int:
        raise ValueError(
            f"Attribute '{rep_attr}' is a mixture of indices and coordinates."
        )

    if idx_int:
        Z = adata.obsm[emb_key][np.asarray(idx_int, dtype=int)]
    else:
        Z = np.vstack(vectors)

    D = cdist(Z, Z, metric=metric)
    return (D, lin_nodes) if return_labels else D


# --------------------------------------------------------------------------- #
#  embedded_tree_distance_matrix                                              #
# --------------------------------------------------------------------------- #
def embedded_tree_distance_matrix(
    G: nx.Graph,
    adata,
    *,
    emb_key: str,
    rep_attr: str = "medoid",
    metric: str = "euclidean",
    return_labels: bool = True,
):
    """
    Distance between lineage nodes measured **along the lineage tree** but
    ignoring (“skipping over”) any internal node that has *no* coordinate.

    The path A–B–C–D therefore contributes

        ‖x_A – x_C‖ + ‖x_C – x_D‖

    when B is unmapped.

    Only endpoints that *do* have a coordinate are included in the final
    matrix.
    """
    # ---- 1 · collect coordinates -----------------------------------------
    coords = {}                # node → vector
    for n, data in G.nodes(data=True):
        val = data.get(rep_attr, None)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        if np.isscalar(val) and np.equal(np.mod(val, 1), 0):
            coords[n] = adata.obsm[emb_key][int(val)]
        else:
            coords[n] = np.asarray(val, float)

    mapped_nodes = sorted(coords.keys())
    if not mapped_nodes:
        raise ValueError(f"No node carries coordinates via '{rep_attr}'")

    idx = {n: i for i, n in enumerate(mapped_nodes)}
    L   = len(mapped_nodes)
    D   = np.full((L, L), np.nan)

    # distance helper
    _dist = lambda a, b: cdist([a], [b], metric=metric)[0, 0]

    # ---- 2 · pairwise path compression ------------------------------------
    for i, src in enumerate(mapped_nodes):
        for j, tgt in enumerate(mapped_nodes[i + 1 :], start=i + 1):
            G_undirected = G.to_undirected(as_view=True)
            path = nx.shortest_path(G_undirected, src, tgt)      # unique in a tree
            acc, prev = 0.0, src

            # skip endpoints (start from 1, stop at -1)
            for node in path[1:]:
                if node not in coords:                # unmapped → ignore
                    continue
                acc += _dist(coords[prev], coords[node])
                prev = node
            D[i, j] = D[j, i] = acc

    np.fill_diagonal(D, 0.0)
    return (D, mapped_nodes) if return_labels else D



def pairwise_lineage_distances(
    G,
    adata,
    *,
    mode: str = "euclidean",       # "euclidean"|"cosine"|"geodesic"|"hop"
    emb_key: str | None = None,    # needed for euclid/cosine
    rep_attr: str = "medoid",
    conn_key: str | None = None,   # needed for geodesic/hop
    rep_key: str | None = None,
    n_neighbors: int = 30,
    directed: bool = False,
    return_labels: bool = True,
):
    """
    One call → four distance flavours:

    mode="euclidean" | "cosine"
        • Uses vectors / indices in ``G.nodes[n][rep_attr]``  
        • You must set `emb_key`.

    mode="geodesic"
        • Dijkstra on a k-NN graph (edge weight = Euclidean distance).  
        • Provide `conn_key` OR let the helper create it from `rep_key`.

    mode="hop"
        • Same as geodesic but every edge has length 1.0.

    Returns
    -------
    D : ndarray  (L×L)
    lin_nodes : list[str]   (when *return_labels* is True)
    """
    m = mode.lower()
    if m in {"euclidean", "cosine"}:
        if emb_key is None:
            raise ValueError("emb_key required for Euclidean / cosine distance")
        D, nodes = embedding_distance_matrix(
            G, adata,
            emb_key=emb_key,
            rep_attr=rep_attr,
            metric=m,
        )

    elif m in {"geodesic", "hop"}:
        D, nodes = geodesic_distance_matrix(
            G, adata,
            conn_key=conn_key,
            rep_attr=rep_attr,
            rep_key=rep_key or emb_key,   # same notion as Scanpy
            n_neighbors=n_neighbors,
            directed=directed,
            hop_weight=(m == "hop"),
        )
    elif m == "tree_geodesic":
        if emb_key is None:
            raise ValueError("emb_key required for tree-embedded distance")
        D, nodes = embedded_tree_distance_matrix(
            G, adata,
            emb_key=emb_key,
            rep_attr=rep_attr,
            metric="euclidean",      # or expose via a kwarg if you like
        )
    else:
        raise ValueError("mode must be euclidean, cosine, geodesic or hop")

    return (D, nodes) if return_labels else D

# ──────────────────────────────────────────────────────────────────────────────
# 3.  tiny convenience → DataFrame of centres (for heat-maps, etc.)
# ──────────────────────────────────────────────────────────────────────────────
def node_centres_dataframe(G, rep_attr: str):
    """
    Collect every non-NaN centre vector stored in ``G.nodes[n][rep_attr]`` into
    a tidy ``pd.DataFrame`` (index=node name, columns=embedding dims).
    """
    import pandas as pd

    rows, names = [], []
    for n, data in G.nodes(data=True):
        v = data.get(rep_attr, None)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if np.isscalar(v):
            continue          # skip pure indices – this helper is for coords
        rows.append(np.asarray(v, dtype=float))
        names.append(n)
    return pd.DataFrame(rows, index=names)




# --------------------------------------------------------------------------
# 0.  tiny import block (bottom of lineage_helpers.py)
# --------------------------------------------------------------------------
import numpy as np
import scipy.sparse.csgraph as csgraph
from scipy.spatial.distance import cdist








import numpy as np
import networkx as nx


def _clean_distance_matrix(D, nan_action="drop", fill_value=None):
    """
    Make matrix acceptable to scikit-bio DistanceMatrix.

    nan_action : "drop" | "fill"
        • "drop" → remove any row/col that contains NaN in *either* matrix
        • "fill" → replace NaN with `fill_value`
    """
    D = np.asarray(D, float)

    # 1) enforce symmetry numerically
    D = 0.5 * (D + D.T)

    # 2) set diagonal exactly zero
    np.fill_diagonal(D, 0.0)

    # 3) handle NaNs
    if np.isnan(D).any():
        if nan_action == "fill":
            if fill_value is None:
                # default to the max finite distance times 1.1
                fill_value = np.nanmax(D) * 1.1
            D = np.nan_to_num(D, nan=fill_value)
        elif nan_action == "drop":
            keep = ~np.isnan(D).any(axis=1)
            D = D[keep][:, keep]
        else:
            raise ValueError("nan_action must be 'drop' or 'fill'")

    return D



from skbio.stats.distance import DistanceMatrix, mantel

def mantel_correlation(D1: np.ndarray, D2: np.ndarray,
                       method="spearman", permutations=999) -> tuple:
    """
    Returns (r, p_value, z) from the Mantel test.
    """
    D1 = _clean_distance_matrix(D1, nan_action="drop")
    D2 = _clean_distance_matrix(D2, nan_action="drop")
    print("Mantel test: D1 shape =", D1.shape, "D2 shape =", D2.shape)
    assert D1.shape == D2.shape, "Distance matrices must have the same shape"
    dm1 = DistanceMatrix(D1)
    dm2 = DistanceMatrix(D2)
    r, p, z = mantel(dm1, dm2,
                     method=method,
                     permutations=permutations,
                     alternative="two-sided",
                     strict=False)
    return r, p, z


def parse_annotation(annot_value):
    """
    Convert annotation (which could be None, np.nan, a string, 
    a list of strings, etc.) into a clean list of valid strings.
    """
    if annot_value is None:
        return []
    if not isinstance(annot_value, list):
        annot_list = [annot_value]
    else:
        annot_list = annot_value

    valid_strings = []
    for val in annot_list:
        if pd.isna(val):  # real NaN
            continue
        if val == 'nan':  # string 'nan'
            continue
        valid_strings.append(str(val))
    return valid_strings



# -----------------------------------------------------------------------------#
#  PATH & PLOTTING UTILITIES
# -----------------------------------------------------------------------------#
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ---------------------------- 1.1  path building -----------------------------#
def get_root_leaf_paths(G: nx.DiGraph, root: str | None = None) -> list[list[str]]:
    """
    Return every root→leaf path in *G*.

    Parameters
    ----------
    G     : directed acyclic lineage graph
    root  : optional root node.  If ``None`` the first node with in-degree 0
            is used.

    Returns
    -------
    paths : list of node-lists
    """
    if root is None:
        roots = [n for n in G.nodes if G.in_degree(n) == 0]
        if not roots:
            raise ValueError("No root with in_degree==0 found")
        root = roots[0]

    leaves = [n for n in G.nodes if G.out_degree(n) == 0]
    return [nx.shortest_path(G, root, leaf) for leaf in leaves]


# ---------------------------- 1.2  leaf → broad group ------------------------#
def map_leaf_to_broad_group(
    leaf: str,
    broad_map: dict[str, list[str]] | None = None,
) -> str | None:
    """
    Map *leaf* to the first key in *broad_map* whose prefix list matches.
    Falls back to ``BROAD_LINEAGE_PREFIXES`` when *broad_map* is None.
    """
    broad_map = broad_map or BROAD_LINEAGE_PREFIXES
    for grp, prefixes in broad_map.items():
        if any(leaf.startswith(p) for p in prefixes):
            return grp
    return None


def label_paths(
    paths: list[list[str]],
    broad_map: dict[str, list[str]] | None = None,
) -> list[str]:
    """
    Label each root→leaf *path* by its leaf’s broad group (or the leaf name).
    """
    return [map_leaf_to_broad_group(p[-1], broad_map) or p[-1] for p in paths]



# -----------------------------------------------------------------------------#
#  plot_lineage_paths
# -----------------------------------------------------------------------------#
def plot_lineage_paths(
    adata,
    G: nx.DiGraph,
    paths: list[list[str]],
    path_labels: list[str],
    *,
    # ─────────  embedding / key names  ───────── #
    basis: str = "Concord-decoder_UMAP",
    G_lineage_key: str = "linannot_mapped",
    G_celltype_key: str = "celltype_mapped",
    adata_lineage_key: str = "lineage_complete",
    adata_celltype_key: str = "cell_type",
    # ─────────  appearance  ───────── #
    palette: str | Sequence[str] | dict[str, str] = "tab20",
    marker_size: int = 2,
    marker_edgewidth: float = 0.1,
    marker_alpha: float = 0.8,
    line_width: float = 0.4,
    dpi: int = 600,
    figsize: tuple[int, int] = (4, 4),
    plot_label: bool = False,
    # ─────────  multi-panel options  ───────── #
    multi_panel: bool = False,
    n_cols: int = 8,
    exclude_groups: list[str] | None = None,
    # ─────────  misc.  ───────── #
    seed: int | None = None,
    save_path: str | Path | None = None,
):
    """
    Draw lineage paths on a 2-D embedding.

    *palette* accepts
        • a Matplotlib colormap name (str) — e.g. "tab20", "viridis" …  
        • a sequence of colours — list/tuple of hex codes / colour names  
        • a dict {group: colour} for explicit mapping
    """

    exclude_groups = set(exclude_groups or [])
    uniq_groups = sorted({g for g in path_labels if g not in exclude_groups})

    # ── resolve palette → {group: colour} -----------------------------------
    if isinstance(palette, dict):
        group2color = palette.copy()
        missing = [g for g in uniq_groups if g not in group2color]
        if missing:
            cmap = cm.get_cmap("tab20", len(missing))
            group2color.update({g: cmap(i) for i, g in enumerate(missing)})
    else:
        if isinstance(palette, str):
            cmap = cm.get_cmap(palette, len(uniq_groups))
            colours = [cmap(i) for i in range(len(uniq_groups))]
        else:  # sequence
            colours = list(palette)
            if len(colours) < len(uniq_groups):
                colours *= int(np.ceil(len(uniq_groups) / len(colours)))
        group2color = dict(zip(uniq_groups, colours))

    # background points cached once
    bg_x, bg_y = adata.obsm[basis][:, 0], adata.obsm[basis][:, 1]

    # ── helper to draw a single path ----------------------------------------
    def _draw_one_path(ax, path, *, color):
        rep_pts, node_labels = [], []
        for node in path:
            attrs = G.nodes[node]
            lin = parse_annotation(attrs.get(G_lineage_key))
            ct  = parse_annotation(attrs.get(G_celltype_key))

            if lin:
                mask = adata.obs[adata_lineage_key].isin(lin)
                sel = lin
            elif ct:
                mask = adata.obs[adata_celltype_key].isin(ct)
                sel = ct
            else:
                rep_pts.append([np.nan, np.nan]); node_labels.append((node, [])); continue

            idx = np.where(mask)[0]
            if idx.size == 0:
                rep_pts.append([np.nan, np.nan]); node_labels.append((node, []))
            else:
                coords = adata.obsm[basis][idx]
                rp = get_representative_point(coords, method="medoid",
                                              max_n_medoid=2000, k_top=3,
                                              jitter=0, seed=seed)
                rep_pts.append(rp); node_labels.append((node, sel))

        rep_pts = np.asarray(rep_pts)
        valid = ~np.isnan(rep_pts[:, 0])
        if valid.any():
            ax.plot(rep_pts[valid, 0], rep_pts[valid, 1],
                    color=color, marker="o", markersize=marker_size,
                    markeredgecolor="black", markeredgewidth=marker_edgewidth,
                    linewidth=line_width, alpha=marker_alpha, zorder=1)
            if plot_label:
                for (x, y), (name, ann) in zip(rep_pts[valid],
                                               [node_labels[i] for i in np.where(valid)[0]]):
                    ax.text(x, y, f"{name}\n{ann}", fontsize=2, color="black", alpha=0.5)

    # ── layout --------------------------------------------------------------
    if multi_panel:
        if not uniq_groups:
            raise ValueError("No groups to plot (all excluded?)")

        n_rows = int(np.ceil(len(uniq_groups) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(2 * n_cols, 2 * n_rows), dpi=dpi)
        axes = axes.flatten()

        for i, grp in enumerate(uniq_groups):
            ax = axes[i]
            ax.scatter(bg_x, bg_y, s=0.1, color="lightgray", alpha=0.5,
                       edgecolors="none", rasterized=True, zorder=0)
            for p, lbl in zip(paths, path_labels):
                if lbl == grp:
                    _draw_one_path(ax, p, color=group2color[grp])
            ax.set_title(grp, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
        for j in range(len(uniq_groups), len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
    else:
        plt.figure(figsize=figsize, dpi=dpi)
        plt.scatter(bg_x, bg_y, s=0.1, color="lightgray", alpha=0.4,
                    edgecolors="none", rasterized=True, zorder=0)
        for p, lbl in zip(paths, path_labels):
            _draw_one_path(plt.gca(), p, color=group2color[lbl])
        plt.title(f"Lineage paths on {basis}", fontsize=12)
        plt.xticks([]); plt.yticks([]); plt.xlabel(""); plt.ylabel("")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()




# ──────────────────────────────────────────────────────────────────────────────
#  lineage distance helpers
# ──────────────────────────────────────────────────────────────────────────────
import itertools
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# ------------------------------------------------------------------ 1.1 ------
# --------------------------------------------------------------------------- #
#  Accurate AB-like lineage relationship
# --------------------------------------------------------------------------- #
# 3. Define relationships
def classify_lineage_relationship(node1, node2):
    """ Define relationship between two nodes based on naming structure. """
    gen1, gen2 = len(node1), len(node2)

    if gen1 != gen2:
        return None  # Only compare nodes of the same generation

    # Check if they share the same parent
    if node1[:-1] == node2[:-1]:  
        return "Sisters",1
    
    # Check if they share the same grandparent
    if node1[:-2] == node2[:-2]:  
        return "Cousins",2
    
    # Check if they share the same great-grandparent
    if node1[:-3] == node2[:-3]:  
        return "2nd Cousins",3
    
    if node1[:-4] == node2[:-4]:  
        return "3rd Cousins",4
    
    if node1[:-5] == node2[:-5]:  
        return "4th Cousins",5
    
    if node1[:-6] == node2[:-6]:
        return "5th Cousins",6
    
    return "6th+ Cousins",np.nan  # More distant relatives



# ------------------------------------------------------------------ 1.2 ------




def lineage_pair_table(
    G: nx.DiGraph,
    D: np.ndarray,
    nodes: list[str],
    *,
    root="AB",
    relationship_key="relationship",
    append_nan=False,
):
    """
    Convert a (L×L) distance matrix + node list into the tidy pair table:

        node1 · node2 · relationship · relationship_distance ·
        generation · latent_distance
    """
    root_nodes = [n for n in G.nodes if str(n).startswith(root)]
    #root_nodes = [n for n in nodes if n.startswith(root)]
    gen = {n: len(n) - len(root) for n in root_nodes}
    idx = {n: i for i, n in enumerate(nodes)}

    records = []
    for g in sorted(set(gen.values())):
        same_gen = [n for n in root_nodes if gen[n] == g]
        for n1, n2 in itertools.combinations(same_gen, 2):
            rel, rel_dist = classify_lineage_relationship(n1, n2)
            if rel is None:
                continue
            d = D[idx[n1], idx[n2]] if (n1 in idx and n2 in idx) else np.nan
            if np.isnan(d) and not append_nan:
                continue
            records.append([n1, n2, rel, rel_dist, g, d])

    cols = ["node1", "node2", relationship_key,
            "relationship_distance", "generation", "latent_distance"]
    return pd.DataFrame.from_records(records, columns=cols)



# ------------------------------------------------------------------ 1.3a -----
def generation_spearman_corr(
    df_pairs: pd.DataFrame,
    *,
    generation_range: tuple[int, int] = (5, 9),
    relationship_key: str = "relationship",
) -> pd.DataFrame:
    """
    Return a tidy DataFrame with one Spearman correlation per lineage × generation,
    restricted to `generation_range` and excluding “6th+ Cousins”.
    """
    df = (
        df_pairs[
            df_pairs["generation"].between(*generation_range)
            & (df_pairs[relationship_key] != "6th+ Cousins")
        ]
        .copy()
    )

    corr = (
        df.groupby(["generation", "node1"])
        .apply(
            lambda d: spearmanr(
                d["relationship_distance"], d["latent_distance"]
            )[0]
            if len(d) > 1
            else np.nan
        )
        .rename("spearman_corr")
        .reset_index()
    )
    return corr


# ------------------------------------------------------------------ 1.3b -----
def plot_lineage_distance_boxplot(
    df_pairs: pd.DataFrame,
    *,
    root: str,
    basis_name: str,
    generation_range: tuple[int, int] = (5, 9),
    relationship_key: str = "relationship",
    figsize: tuple[int, int] = (3, 1.5),
    save_path: str | Path | None = None,
    corr_df: pd.DataFrame | None = None,
):
    """
    Box-and-whisker plot of latent distances per generation, overlaid with the
    median lineage–wise Spearman ρ (distance vs genealogical distance).

    If *corr_df* is **None**, the correlations are computed on-the-fly with
    `generation_spearman_corr(...)`; otherwise the supplied table is used.
    """
    # ── 1 · subset to wanted rows ────────────────────────────────────────────
    df_plot = df_pairs[
        df_pairs["generation"].between(*generation_range)
        & (df_pairs[relationship_key] != "6th+ Cousins")
    ].copy()
    df_plot["generation_label"] = df_plot["generation"].apply(lambda g: f"{root}{g}")

    # ── 2 · correlations ─────────────────────────────────────────────────────
    if corr_df is None:
        corr_df = generation_spearman_corr(
            df_pairs,
            generation_range=generation_range,
            relationship_key=relationship_key,
        )

    med_corr = corr_df.groupby("generation")["spearman_corr"].median()

    # ── 3 · palette & figure ────────────────────────────────────────────────
    palette = sns.color_palette("husl", n_colors=df_plot[relationship_key].nunique())
    fig, ax = plt.subplots(figsize=figsize, dpi=600)
    sns.boxplot(
        x="generation_label",
        y="latent_distance",
        hue=relationship_key,
        data=df_plot,
        palette=palette,
        width=0.95,
        linewidth=0.5,
        fliersize=0,
        ax=ax,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    y_max = df_plot["latent_distance"].max()
    for i, (g, rho) in enumerate(med_corr.items()):
        ax.text(i, y_max * 1.05, f"{rho:.2f}", ha="center", va="bottom", fontsize=6)

    ax.set(
        xlabel="Cell Generation",
        ylabel="Latent-space distance",
        title=f"{basis_name} distance distribution in {root} lineage",
    )
    ax.tick_params(axis="x", labelsize=6)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(
        title="Lineage relationship",
        fontsize=6,
        title_fontsize=6,
        loc="upper left",
        bbox_to_anchor=(1, 1),
    )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()

    return ax

# --------------------------------------------------------------------------- #
#  lineage_correlation_boxplot  (stratified-by-generation option)
# --------------------------------------------------------------------------- #
def lineage_correlation_boxplot(
    correlation_dict: dict[str, pd.DataFrame],
    *,
    # ───────── which data to show ───────── #
    show_keys: list[str] | None = None,
    sort: bool = True,                  # sort by median correlation
    generation_range: tuple[int, int] | None = None,   # (min, max) → per-gen grid
    reference_method: str | None = None,               # MW-U stars vs this
    # ───────── appearance ───────── #
    palette: str = "RdBu",
    figsize_single: tuple[float, float] = (2.2, 1.2),
    n_cols: int = 4,                                   # grid for per-gen mode
    flier_marker_size: float = 0.5,
    title: str = "Spearman correlation\n(lineage ↔ latent distance)",
    custom_rc: dict | None = None,
    # ───────── misc ───────── #
    save_path: str | Path | None = None,
    return_data: bool = False,
):
    """
    Compare lineage–wise Spearman correlations across embeddings.

    • Default: single horizontal box-plot (all generations pooled).  
    • Pass *generation_range=(g_min, g_max)* → grid of subplots
      (one box-plot per generation).

    Significance (Mann-Whitney-U) stars are always drawn **within each
    generation** (or once, if pooled) against *reference_method*.

    Stars: **** ≤1e-4, *** ≤1e-3, ** ≤1e-2, * ≤0.05
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.stats as stats
    import contextlib
    import numpy as np
    import pandas as pd

    # ── 0 · input -----------------------------------------------------------
    if show_keys is None:
        show_keys = list(correlation_dict.keys())

    df_all = pd.concat(
        [correlation_dict[m].assign(method=m) for m in show_keys],
        ignore_index=True
    ).dropna(subset=["spearman_corr"])

    if generation_range:
        g_min, g_max = generation_range
        df_all = df_all[df_all["generation"].between(g_min, g_max)]

    generations = sorted(df_all["generation"].unique())  # after filtering

    # ── 1 · colour palette --------------------------------------------------
    med_global = df_all.groupby("method")["spearman_corr"].median()
    if sort:
        sorted_methods = med_global.sort_values(ascending=False).index.tolist()
    else:
        sorted_methods = list(med_global.index)

    colors = sns.color_palette(palette, len(sorted_methods))
    method_color = {m: colors[i] for i, m in enumerate(sorted_methods)}

    flierprops = dict(marker="o", markersize=flier_marker_size,
                      markerfacecolor="black", linestyle="none")

    # ── 2 · helper to draw ONE box-plot axis --------------------------------
    def _draw_one(ax, df_gen: pd.DataFrame, title_suffix: str = "", 
                override_order=None,
                override_palette=None) -> None:
        med = df_gen.groupby("method")["spearman_corr"].median()
        order = override_order or sorted_methods
        pal   = override_palette or method_color
        sns.boxplot(
            y="method", x="spearman_corr",
            data=df_gen,
            palette=pal,
            width=0.7, linewidth=0.5,
            order=sorted_methods,
            flierprops=flierprops,
            ax=ax,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        # median labels
        for i, m in enumerate(sorted_methods):
            if m in med:
                ax.text(med[m] + 0.02, i, f"{med[m]:.2f}",
                        ha="left", va="center", fontsize=6)

        # MW-U stars vs reference
        if reference_method and reference_method in sorted_methods:
            ref_vals = df_gen[df_gen["method"] == reference_method]["spearman_corr"]
            ref_idx = sorted_methods.index(reference_method)
            x_off = df_gen["spearman_corr"].max() + 0.05

            def stars(p):
                return "****" if p <= 1e-4 else \
                       "***"  if p <= 1e-3 else \
                       "**"   if p <= 1e-2 else \
                       "*"    if p <= 0.05 else ""

            for m in sorted_methods:
                if m == reference_method:
                    continue
                test_vals = df_gen[df_gen["method"] == m]["spearman_corr"]
                if test_vals.empty:
                    continue
                _, p = stats.mannwhitneyu(ref_vals, test_vals,
                                          alternative="two-sided")
                sig = stars(p)
                if not sig:
                    continue
                test_idx = sorted_methods.index(m)
                ax.plot([x_off, x_off], [ref_idx, test_idx],
                        color="black", linewidth=0.5)
                ax.plot([x_off, x_off + 0.05], [test_idx, test_idx],
                        color="black", linewidth=0.5)
                ax.text(x_off + 0.07, (ref_idx + test_idx)/2,
                        sig, fontsize=7, ha="left", va="center")

        ax.set(xlabel="", ylabel="")
        ax.tick_params(axis="x", labelsize=6)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_title(title_suffix, fontsize=8)

    # ── 3 · plotting --------------------------------------------------------
    rc_ctx = plt.rc_context(rc=custom_rc) if custom_rc else contextlib.nullcontext()
    with rc_ctx:
        if generation_range:
            # ----- grid of subplots, one per generation --------------------
            n_cols = min(n_cols, len(generations))
            n_rows = int(np.ceil(len(generations) / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(figsize_single[0]*n_cols,
                                              figsize_single[1]*n_rows),
                                     dpi=600, sharex=False, sharey=True)
            axes = np.array(axes).reshape(-1)
            for ax, gen in zip(axes, generations):
                df_g = df_all[df_all["generation"] == gen]

                # --- build a palette specific to THIS generation ----------------------
                med_g = df_g.groupby("method")["spearman_corr"].median()
                methods_g = med_g.sort_values(ascending=False).index.tolist()

                cols_g = sns.color_palette(palette, len(methods_g))
                palette_g = {m: cols_g[i] for i, m in enumerate(methods_g)}

                _draw_one(ax, df_g, f"Gen {gen}", override_order=methods_g, override_palette=palette_g)
            for ax in axes[len(generations):]:
                ax.axis("off")
            fig.suptitle(title, y=1.02, fontsize=9)
            plt.tight_layout()
        else:
            # ----- pooled single plot -------------------------------------
            fig, ax = plt.subplots(figsize=figsize_single, dpi=600)
            _draw_one(ax, df_all, "")
            ax.set_title(title, fontsize=8)
            plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        plt.show()

    if return_data:
        return (ax if generation_range is None else axes), df_all




# --------------------------------------------------------------------------- #
#  0 · utilities                                                              #
# --------------------------------------------------------------------------- #
def _iter_trim(subg: nx.DiGraph, trim_roots: list[str]) -> None:
    """Remove *each* root plus all its descendants (in-place)."""
    for r in trim_roots:
        subg.remove_nodes_from(nx.descendants(subg, r) | {r})


# --------------------------------------------------------------------------- #
#  1 · extract_subgraph                                                       #
# --------------------------------------------------------------------------- #
def extract_subgraph(
    G: nx.DiGraph,
    roots: list[str],
    trim: list[str] | None = None,
) -> nx.DiGraph:
    """
    Return the induced sub-tree that contains *roots* and **all** their
    descendants and, optionally, drop entire branches given in *trim*.
    """
    keep: set[str] = set()
    for r in roots:
        keep.update(nx.descendants(G, r))
        keep.add(r)

    subg = G.subgraph(keep).copy()
    if trim:
        _iter_trim(subg, trim)
    return subg


# --------------------------------------------------------------------------- #
#  2 · palette_from_lin_or_ct                                                 #
# --------------------------------------------------------------------------- #
def palette_from_lin_or_ct(adata, subg, pal="Set1") -> dict[str, str]:
    """
    Build a colour mapping for every ``lin_or_ct`` value that occurs in *subg*.

    Returns
    -------
    node2colour : dict[node -> colour]
    """
    import seaborn as sns

    # collect all lin_or_ct strings
    lin_ct = []
    for n in subg.nodes():
        lin_ct.extend(subg.nodes[n].get("linorct", []))
    lin_ct = sorted(set(lin_ct))

    # make palette with seaborn
    cols = sns.color_palette(pal, len(lin_ct))
    lut  = dict(zip(lin_ct, cols))

    node2colour = {}
    for n in subg.nodes():
        annot = subg.nodes[n].get("linorct", [])
        node2colour[n] = lut.get(annot[0], "lightgrey") if annot else "lightgrey"
    return node2colour


# --------------------------------------------------------------------------- #
#  3 · draw_lineage_tree                                                      #
# --------------------------------------------------------------------------- #
def draw_lineage_tree(
    subg: nx.DiGraph,
    node2colour: dict[str, str],
    *,
    show_ct: bool = True,
    figsize=(1.8, 0.7),
    dpi=600,
    save_path: str | Path | None = None,
    title: str | None = None,
):
    """Convenience wrapper around NetworkX + graphviz layout."""
    # --- labels ------------------------------------------------------------
    if show_ct:
        labels = {}
        for n in subg.nodes():
            ct = subg.nodes[n].get("celltype_annot", [])
            ct_txt = ",".join(ct) if ct else ""
            labels[n] = f"{n}/{ct_txt}" if ct_txt else n
    else:
        labels = {n: n for n in subg.nodes()}

    pos = nx.nx_agraph.graphviz_layout(subg, prog="dot", args="-Grankdir=LR")
    nlist = list(subg.nodes())
    colours = [node2colour[n] for n in nlist]

    plt.figure(figsize=figsize, dpi=dpi)
    nx.draw(
        subg, pos,
        nodelist=nlist,
        node_color=colours,
        labels=labels,
        node_size=15,
        font_size=5,
        arrowsize=4,
        width=0.5,
    )
    if title:
        plt.title(title, fontsize=7)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# --------------------------------------------------------------------------- #
#  4 · plot_paths_on_embedding                                                #
# --------------------------------------------------------------------------- #
def plot_paths_on_embedding(
    adata,
    subg: nx.DiGraph,
    paths: list[list[str]],
    *,
    basis_umap: str,
    palette_lin_or_ct: dict[str, str],
    plot_labels: bool = False,
    zoom: bool = False,
    square: bool = False,
    figsize=(2, 2),
    dpi=600,
    save_path: str | Path | None = None,
):
    """
    Draw *paths* (each is a list of node IDs) on a 2-D embedding stored in
    ``adata.obsm[basis_umap]`` and colour the vertices by *palette_lin_or_ct*.
    """
    # ---- background -------------------------------------------------------
    # ---------- background -------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.scatter(*adata.obsm[basis_umap].T, s=0.1, color="lightgray",
               alpha=0.8, edgecolors="none", rasterized=True, zorder=0)

    mask = adata.obs["lin_or_ct"].isin(palette_lin_or_ct)
    ax.scatter(*adata.obsm[basis_umap][mask].T,
            c=adata.obs.loc[mask, "lin_or_ct"]
                .map(palette_lin_or_ct)
                .fillna("lightgrey")
                .values,
            s=0.3, alpha=0.8, edgecolors="none",
            rasterized=True, zorder=1)

    # ---------- draw every path -------------------------------------------
    all_pts = []                              # ← collect every node position
    for path in paths:
        pts = []
        for n in path:
            annot = subg.nodes[n].get("linorct")
            if not annot:
                continue
            mask   = adata.obs["lin_or_ct"].isin(annot)
            coords = adata.obsm[basis_umap][mask]
            if coords.size == 0:
                continue
            pts.append(
                get_representative_point(coords, method="medoid",
                                         max_n_medoid=2000, k_top=10)
            )

        pts = np.asarray(pts)
        if pts.size == 0:
            continue
        all_pts.append(pts)                  # ← keep a copy for zooming
        ax.plot(pts[:, 0], pts[:, 1],
                color="black", linewidth=0.3,
                marker="o", markersize=3,
                markerfacecolor="white",
                markeredgewidth=0.2, zorder=2)

        if plot_labels:
            for (x, y), n in zip(pts, path):
                ax.text(x, y, n, fontsize=2, alpha=0.5)

    # ---------- optional zoom ----------------------------------------------
    if zoom and all_pts:
        stacked = np.vstack(all_pts)
        min_x, min_y = stacked.min(0)
        max_x, max_y = stacked.max(0)
        margin = 0.1 * max(max_x - min_x, max_y - min_y)
        if square:
            side  = max(max_x - min_x, max_y - min_y) + 2 * margin
            cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
            ax.set_xlim(cx - side/2, cx + side/2)
            ax.set_ylim(cy - side/2, cy + side/2)
        else:
            ax.set_xlim(min_x - margin, max_x + margin)
            ax.set_ylim(min_y - margin, max_y + margin)

    ax.set_xticks([]); ax.set_yticks([])
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()




# ──────────────────────────────────────────────────────────────────────────────
# Helpers reused from earlier messages (keep them once in your module)
# ──────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Iterable, Optional, Sequence, Tuple
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import matplotlib.pyplot as plt
import contextlib

def build_cell_to_nodes(G: nx.DiGraph) -> Dict[int, list[str]]:
    """Collect all lineage nodes each cell maps to (no tie-breaking)."""
    cell_to_nodes: Dict[int, list[str]] = {}
    for n, data in G.nodes(data=True):
        for i in data.get("cells", []) or []:
            lst = cell_to_nodes.setdefault(int(i), [])
            if n not in lst:
                lst.append(n)
    return cell_to_nodes

def lineage_hop_lookup(
    G: nx.DiGraph,
    mapped_nodes: Optional[Iterable[str]] = None,
    *,
    undirected: bool = True,
) -> Dict[str, Dict[str, int]]:
    """dist[u][v] = hop distance on the lineage tree."""
    H = G.to_undirected(as_view=True) if undirected else G
    if mapped_nodes is None:
        mapped_nodes = list(G.nodes)
    mapped_nodes = [n for n in mapped_nodes if n in G]
    dist = {}
    for u in mapped_nodes:
        d = nx.single_source_shortest_path_length(H, u)
        dist[u] = {v: d[v] for v in mapped_nodes if v in d}
    return dist

# ──────────────────────────────────────────────────────────────────────────────
# 1) Per-embedding evaluation (returns per-anchor rows)
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_embedding_vs_lineage_knn(
    adata,
    G: nx.DiGraph,
    *,
    emb_key: str,
    k_list: Sequence[int] = (30, 100, 300),
    n_anchors: Optional[int] = 1000,
    random_state: int = 0,
    metric: str = "euclidean",
    use_faiss: bool = False,               # set True if you wire in your FAISS class
    missing_policy: str = "skip",          # "skip" | "nan" | "max"
    undirected_tree: bool = True,
    cell_to_nodes: Optional[Dict[int, list[str]]] = None,
) -> pd.DataFrame:
    """
    For each anchor cell, take its kNN in adata.obsm[emb_key] and compute:
      • mean_hop  – mean of min-hop to each neighbor (lower is better)
      • purity    – fraction of neighbors with hop == 0
      • frac_mapped – neighbors with a computable hop / k

    Multiple lineage mappings per cell are supported: hop(anchor, nb) is
    min_{u in nodes(anchor), v in nodes(nb)} hop(u, v).
    """
    # ---------- 1) mappings / lineage distances ------------------------------
    if cell_to_nodes is None:
        cell_to_nodes = build_cell_to_nodes(G)
    mapped_nodes = sorted({n for nodes in cell_to_nodes.values() for n in nodes})
    dist_lookup = lineage_hop_lookup(G, mapped_nodes, undirected=undirected_tree)

    if missing_policy == "max":
        finite = [d for u in dist_lookup for d in dist_lookup[u].values()]
        max_hop = max(finite) if finite else np.nan

    # ---------- 2) kNN engine (sklearn by default) ---------------------------
    X = adata.obsm[emb_key]
    n = X.shape[0]

    get_knn = None
    if use_faiss:
        try:
            # plug in your FAISS Neighborhood here if desired
            from concord.model import Neighborhood
            eng = Neighborhood(X, k=max(k_list), use_faiss=True, metric=metric)
            def get_knn(i, k):
                return eng.get_knn(np.asarray([i]), k=k, include_self=False)[0]
        except Exception:
            get_knn = None

    if get_knn is None:
        nbrs = NearestNeighbors(n_neighbors=max(k_list)+1, metric=metric).fit(X)
        def get_knn(i, k):
            _, ind = nbrs.kneighbors(X[i:i+1], n_neighbors=k+1)
            return ind[0][1:(k+1)]

    # ---------- 3) anchors ---------------------------------------------------
    rng = np.random.default_rng(random_state)
    anchors = np.arange(n) if (n_anchors is None or n_anchors >= n) else rng.choice(n, size=n_anchors, replace=False)

    # ---------- 4) compute per-anchor stats ---------------------------------
    rows = []
    for k in k_list:
        for a in anchors:
            A = cell_to_nodes.get(int(a), [])
            if not A:  # anchor unmapped
                rows.append(dict(anchor=a, k=k, mean_hop=np.nan, purity=np.nan,
                                 n_used=0, frac_mapped=0.0))
                continue

            neigh = get_knn(int(a), k)
            hop_list = []
            mapped_ct = 0

            for b in neigh:
                B = cell_to_nodes.get(int(b), [])
                if not B:
                    if missing_policy == "skip":
                        continue
                    elif missing_policy == "nan":
                        hop_list.append(np.nan)
                    elif missing_policy == "max":
                        hop_list.append(max_hop)
                    continue

                # min hop over all mapped-node pairs
                best = np.inf
                du = dist_lookup.get  # small speed-up
                for u in A:
                    d_u = du(u, {})
                    for v in B:
                        d = d_u.get(v, np.nan)
                        if not np.isnan(d) and d < best:
                            best = d
                if np.isinf(best):
                    if missing_policy == "skip":
                        continue
                    elif missing_policy == "nan":
                        hop_list.append(np.nan)
                    elif missing_policy == "max":
                        hop_list.append(max_hop)
                else:
                    hop_list.append(best)
                    mapped_ct += 1

            hops = np.asarray(hop_list, float)
            used_mask = np.isfinite(hops)
            used = hops[used_mask]
            mean_hop = np.nan if used.size == 0 else used.mean()
            purity   = np.nan if used.size == 0 else (used == 0).mean()
            frac_mapped = mapped_ct / float(k) if k > 0 else np.nan

            rows.append(dict(anchor=a, k=k,
                             mean_hop=mean_hop, purity=purity,
                             n_used=int(used.size), frac_mapped=frac_mapped))

    return pd.DataFrame(rows)




# ──────────────────────────────────────────────────────────────────────────────
# 2) Run across multiple embeddings / methods
# ──────────────────────────────────────────────────────────────────────────────
def knn_lineage_quality_across_methods(
    adata,
    G: nx.DiGraph,
    emb_keys: Sequence[str],
    *,
    k_list: Sequence[int] = (30, 100, 300),
    n_anchors: Optional[int] = 1000,
    random_state: int = 0,
    metric: str = "euclidean",
    use_faiss: bool = False,
    missing_policy: str = "skip",
    undirected_tree: bool = True,
) -> pd.DataFrame:
    """
    Returns a tidy DataFrame with columns:
        method · k · anchor · mean_hop · purity · n_used · frac_mapped
    """
    # Build once and reuse across methods
    cell_to_nodes = build_cell_to_nodes(G)

    dfs = []
    for m in emb_keys:
        print(f"Evaluating {m} ...")
        df = evaluate_embedding_vs_lineage_knn(
            adata, G,
            emb_key=m,
            k_list=k_list,
            n_anchors=n_anchors,
            random_state=random_state,
            metric=metric,
            use_faiss=use_faiss,
            missing_policy=missing_policy,
            undirected_tree=undirected_tree,
            cell_to_nodes=cell_to_nodes,   # reuse mapping
        )
        df["method"] = m
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)



# ──────────────────────────────────────────────────────────────────────────────
# 3) Boxplots across methods & k (purity and/or mean_hop)
# ──────────────────────────────────────────────────────────────────────────────
def plot_knn_lineage_quality_single(
    df: pd.DataFrame,
    *,
    metric: str = "purity",                 # e.g. "purity" or "mean_hop"
    facet_by_k: bool = True,
    order_methods: Optional[Sequence[str]] = None,
    palette: str = "tab10",
    figsize: Tuple[float, float] = (6, 2.2),
    custom_rc: Optional[dict] = None,
    save_path: Optional[str] = None,
    # --- new options for y-limit capping ---
    y_cap_quantile: Optional[float] = None,   # e.g. 95 → cap at 95th percentile
    y_cap_expand: float = 1.03,               # add headroom above the cap
    y_cap_per_k: bool = False,                # per-facet cap when facet_by_k=True
) -> Union[plt.Axes, List[plt.Axes]]:
    """
    Boxplots of a single KNN-lineage metric across methods and k.

    If `y_cap_quantile` is provided, the y-axis top is capped at the chosen
    percentile (global or per-k when faceting), multiplied by `y_cap_expand`.

    Returns
    -------
    Axes (if facet_by_k=False) or list[Axes] (if facet_by_k=True)
    """
    if metric not in df.columns:
        raise ValueError(f"'{metric}' not found in df columns.")

    # keep only necessary cols and drop NaNs for the target metric
    need_cols = ["method", "k", metric]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")
    dfp = df[need_cols].dropna(subset=[metric]).copy()

    # decide order of methods:
    #   larger is better for 'purity', smaller is better for 'mean_hop'
    if order_methods is None:
        asc = (metric == "mean_hop")
        med = dfp.groupby("method")[metric].median().sort_values(ascending=asc)
        order_methods = med.index.tolist()

    _rc = {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "text.usetex": False,
    }
    if custom_rc:
        _rc.update(custom_rc)

    rc_ctx = plt.rc_context(rc=_rc)
    with rc_ctx:
        if facet_by_k:
            # one column per k
            g = sns.catplot(
                data=dfp,
                kind="box",
                x="method", y=metric,
                col="k",
                order=order_methods,
                palette=palette,
                sharey=True,             # same scale across k for this metric
                width=0.7, fliersize=0.5,
                height=figsize[1],
                aspect=max(
                    0.5,
                    figsize[0] / (len(np.unique(dfp["k"])) * figsize[1] + 1e-9)
                ),
            )
            # cosmetics
            for ax in g.axes.flat:
                for spine in ax.spines.values():
                    spine.set_linewidth(0.5)
                ax.tick_params(axis="x", rotation=60, labelsize=8)
                ax.tick_params(axis="y", labelsize=8)
                ax.set_xlabel("")
                ax.set_ylabel("")
            g.set_titles(col_template="k = {col_name}")

            # ---- optional y-cap ----
            if y_cap_quantile is not None:
                if y_cap_per_k:
                    # cap each facet using the subset for that k
                    for ax, kval in zip(g.axes.flat, g.col_names):
                        vals = dfp.loc[dfp["k"] == kval, metric].to_numpy()
                        if np.isfinite(vals).any():
                            p = np.nanpercentile(vals, y_cap_quantile)
                            ax.set_ylim(top=p * y_cap_expand)
                else:
                    # global cap from all values of this metric
                    vals = dfp[metric].to_numpy()
                    if np.isfinite(vals).any():
                        p = np.nanpercentile(vals, y_cap_quantile)
                        for ax in g.axes.flat:
                            ax.set_ylim(top=p * y_cap_expand)

            plt.tight_layout()
            if save_path:
                g.savefig(save_path, bbox_inches="tight", dpi=600)
            return list(g.axes.flat)

        else:
            # one axis, hue by k
            fig, ax = plt.subplots(figsize=figsize, dpi=600)
            sns.boxplot(
                data=dfp,
                x="method", y=metric, hue="k",
                order=order_methods,
                palette=palette,
                width=0.7, fliersize=0.5,
                ax=ax,
            )
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
            ax.tick_params(axis="x", rotation=60, labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
            ax.set(xlabel="", ylabel="")

            # ---- optional y-cap (single axis) ----
            if y_cap_quantile is not None:
                vals = dfp[metric].to_numpy()
                if np.isfinite(vals).any():
                    p = np.nanpercentile(vals, y_cap_quantile)
                    ax.set_ylim(top=p * y_cap_expand)

            plt.tight_layout()
            if save_path:
                fig.savefig(save_path, bbox_inches="tight", dpi=600)
            return ax