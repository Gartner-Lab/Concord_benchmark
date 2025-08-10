from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
from anndata import AnnData
from typing import Optional



def extract_timestamp(folder_name: str, method: str) -> Optional[str]:
    """
    Folder is expected to look like  <method>_<MMDD-HHMM>
    Returns the timestamp part ('MMDD-HHMM') or None if it doesn't match.
    """
    m = re.match(rf"{re.escape(method)}_(\d{{4}}-\d{{4}})", folder_name)
    return m.group(1) if m else None


def latest_run_dir(save_root: Path, method: str) -> Optional[Path]:
    """
    Locate the newest run directory for a given method.
    """
    candidates = [
        p for p in save_root.glob(f"{method}_*")
        if p.is_dir() and extract_timestamp(p.name, method)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: extract_timestamp(p.name, method))


def add_embeddings(adata: AnnData, proj_name: str, methods: list[str]) -> AnnData:
    save_root = Path(f"../save/{proj_name}")

    for method in methods:
        latest_dir = latest_run_dir(save_root, method)
        if latest_dir is None:
            print(f"[⚠️ Warning] No folder found for {method}")
            continue

        ts = extract_timestamp(latest_dir.name, method)
        embedding_file = latest_dir / f"{method}_embedding_{ts}.tsv"
        if not embedding_file.exists():
            print(f"[⚠️ Warning] Missing {embedding_file}")
            continue

        df = pd.read_csv(embedding_file, sep="\t", index_col=0)
        if set(df.index) != set(adata.obs_names):
            print(f"[❌ Error] Cell IDs mismatch for {method}")
            continue

        # 2️⃣— reorder df to match the order in adata.obs_names
        df = df.reindex(adata.obs_names)
        #  (reindex preserves the existing order in adata and raises if any ID is missing)

        # now it’s safe to attach the matrix
        adata.obsm[method] = df.values
        print(f"✅ obsm['{method}'] loaded")

    return adata



def collect_benchmark_logs(proj_name: str, methods: list[str]) -> pd.DataFrame:
    save_root = Path(f"../save/{proj_name}")
    rows = []

    for m in methods:
        run_dir = latest_run_dir(save_root, m)
        if run_dir is None:
            print(f"[⚠️ Warning] no runs found for {m}")
            continue

        ts = extract_timestamp(run_dir.name, m)
        log_tsv = run_dir / f"benchmark_log_{ts}.tsv"
        if not log_tsv.exists():
            print(f"[⚠️ Warning] missing {log_tsv}")
            continue

        df = pd.read_csv(log_tsv, sep="\t")
        df["method"] = m

        rows.append(df)

    if not rows:
        raise RuntimeError("No benchmark logs were read.")

    return pd.concat(rows, ignore_index=True)



import numpy as np
import matplotlib.pyplot as plt

def get_clean_linear_ticks(max_val, preferred_steps=[50, 100, 200, 500, 1000]):
    """
    Dynamically choose the best step size and generate clean ticks.
    """
    for step in preferred_steps:
        if max_val / step <= 10:
            tick_max = int(np.ceil(max_val / step) * step)
            return np.arange(0, tick_max + step, step)
    # fallback for huge values
    step = 2000
    tick_max = int(np.ceil(max_val / step) * step)
    return np.arange(0, tick_max + step, step)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter, LogLocator, NullFormatter
from pathlib import Path
from typing import Optional, Mapping

def plot_benchmark_performance(
    bench_df: pd.DataFrame,
    title: Optional[str] = None,
    figsize: tuple[int, int] = (10, 5),
    dpi: int = 300,
    save_path: Optional[Path] = None,
    rc: Optional[dict] = None,
    label_fontsize: int = 8,
    tick_fontsize: int = 8,
    title_fontsize: int = 14,
    unit: Literal["auto", "MiB", "GiB"] = "auto",
    metric_scale: Union[str, dict] = "auto",
):
    """
    Generates a performance benchmark plot with three horizontal bar charts:
    run-time, RAM, and VRAM usage.

    Args:
        bench_df: DataFrame with columns ["method", "time_sec", "ram_MB", "vram_MB"].
        title: Title of the whole plot.
        figsize: Size of the figure.
        dpi: Resolution of the plot.
        save_path: Optional path to save the figure.
        rc: Optional rcParams override.
        label_fontsize: Font size for axis labels.
        tick_fontsize: Font size for ticks.
        title_fontsize: Font size for the plot title.
        unit: Memory unit: "auto" (default), "MiB", or "GiB".
        metric_scale: Axis scale: "auto", "log", "linear", or dict like {"time_sec": "log", ...}.
    """
    # Determine RAM/VRAM unit
    def determine_memory_unit(column: str) -> tuple[float, str]:
        max_val = bench_df[column].max(skipna=True)
        if unit == "GiB" or (unit == "auto" and max_val > 10240):  # 10 GiB threshold
            return 1 / 1024, "GiB"
        return 1, "MiB"

    ram_factor, ram_label_unit = determine_memory_unit("ram_MB")
    vram_factor, vram_label_unit = determine_memory_unit("vram_MB")

    # Determine axis scale for each metric
    def get_scale(metric: str) -> str:
        if isinstance(metric_scale, dict):
            return metric_scale.get(metric, "linear")
        elif metric_scale in ("log", "linear"):
            return metric_scale
        return "log" if metric == "time_sec" else "linear"

    metrics = [
        ("time_sec", "Run-time (h)", 3600, get_scale("time_sec")),
        ("ram_MB", f"RAM ({ram_label_unit})", 1 / ram_factor, get_scale("ram_MB")),
        ("vram_MB", f"VRAM ({vram_label_unit})", 1 / vram_factor, get_scale("vram_MB")),
    ]

    colour = "steelblue"
    cpu_methods = bench_df.loc[bench_df["vram_MB"] == 0, "method"].tolist()
    nan_methods = bench_df[bench_df.isna().any(axis=1)]["method"].tolist()

    with plt.rc_context(rc or {}):
        fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi, constrained_layout=True)

        for ax, (key, xlabel, div, scale) in zip(axes, metrics):
            df_current = bench_df.copy()
            df_current["_sort_val"] = df_current[key].fillna(np.inf)
            df_current = df_current.sort_values("_sort_val").reset_index(drop=True)
            df_current[key] = df_current[key].fillna(0.0)
            vals = df_current[key] / div
            y_positions = np.arange(len(df_current))

            ax.barh(y_positions, vals, color=colour)
            ax.set_yticks(y_positions)
            ax.set_yticklabels(df_current["method"], fontsize=tick_fontsize)
            ax.invert_yaxis()
            ax.set_xlabel(xlabel, fontsize=label_fontsize)
            ax.tick_params(axis='x', labelsize=tick_fontsize)
            ax.grid(axis="x", ls=":", alpha=.4)

            # Add NaN text
            for y_pos, method in zip(y_positions, df_current["method"]):
                if method in nan_methods:
                    ax.text(0.02, y_pos, "NaN", va="center", fontsize=8, color="black",
                            transform=ax.get_yaxis_transform())

            # Axis scaling
            if scale == "log":
                ax.set_xscale("log")
                min_val = max(0.001, np.nanmin(vals[vals > 0]))
                max_val = np.nanmax(vals)
                tick_min = int(np.floor(np.log10(min_val)))
                tick_max = int(np.ceil(np.log10(max_val)))
                log_ticks = [10**i for i in range(tick_min, tick_max + 1)]
                ax.set_xticks(log_ticks)
                ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))

                ax.xaxis.set_minor_locator(LogLocator(subs='all'))
                ax.xaxis.set_minor_formatter(NullFormatter())
            else:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True, prune="both"))

            # CPU-only label for VRAM
            if key == "vram_MB":
                for y_pos, val, method in zip(y_positions, vals, df_current["method"]):
                    if method in cpu_methods and np.isclose(val, 0, atol=1e-9):
                        ax.text(0.02, y_pos, "CPU only", va="center", fontsize=8, color="black",
                                transform=ax.get_yaxis_transform())

        axes[0].set_ylabel("Integration Method", fontsize=label_fontsize)
        axes[0].tick_params(axis='y', labelsize=tick_fontsize)

        if title:
            fig.suptitle(title, fontsize=title_fontsize, fontweight="bold")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        else:
            plt.show()



import concord as ccd

def compute_umap_and_save(
    adata: AnnData,
    methods: list[str],
    save_dir: Path,
    file_suffix: str,
    data_dir: Path,
    file_name: str,
    seed: int = 42,
    overwrite: bool = True
):
    """
    For each method in `methods`, compute 2D and 3D UMAP from adata.obsm[method].
    If overwrite is True, existing UMAPs will be recomputed. Otherwise, they will be skipped.
    Saves both obsm (.h5) and final AnnData object (.h5ad).

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with integration embeddings.
    methods : list of str
        List of integration method names to compute UMAP from.
    save_dir : Path
        Directory to save obsm h5 file.
    file_suffix : str
        Timestamp or suffix used in saving obsm files.
    data_dir : Path
        Directory to save final .h5ad output.
    file_name : str
        Filename prefix (without .h5ad) for saving the final AnnData.
    seed : int
        Random seed for UMAP reproducibility.
    overwrite : bool
        If True, existing UMAP keys will be recomputed. If False, they will be skipped.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    for method in methods:
        if method not in adata.obsm:
            print(f"[❌ Skipping] adata.obsm['{method}'] not found — cannot compute UMAP")
            continue

        for dim in [2, 3]:
            key = f"{method}_UMAP" + ("_3D" if dim == 3 else "")

            if key in adata.obsm:
                if overwrite:
                    print(f"[⚠️ Notice] obsm['{key}'] already exists — recomputing and updating it")
                else:
                    print(f"[⏭️ Skipping] obsm['{key}'] already exists — skipping (overwrite=False)")
                    continue
            else:
                print(f"🔄 Computing {dim}D UMAP for {method}...")

            # Compute UMAP regardless (if overwrite=True or not previously computed)
            ccd.ul.run_umap(
                adata,
                source_key=method,
                result_key=key,
                n_components=dim,
                n_neighbors=30,
                min_dist=0.1,
                metric="euclidean",
                random_state=seed
            )
            print(f"✅ obsm['{key}'] updated")

    obsm_path = save_dir / f"obsm_{file_suffix}.h5"
    ccd.ul.save_obsm_to_hdf5(adata, obsm_path)
    print(f"💾 obsm saved to {obsm_path}")

    final_path = data_dir / f"{file_name}_final.h5ad"
    adata.write_h5ad(final_path)
    print(f"💾 Final AnnData saved to: {final_path}")


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_ranked_scores(
    score_dict: dict[str, pd.Series],
    figsize=(3, 0.5),
    custom_rc=None,
    method_order: list[str] | None = None,     # kept for backward-compat
    method_mapping: dict[str, str] | None = None,  # NEW: ordered mapping
    save_dir: Path | None = None,
    title: str = "Method Rankings",
    file_name: str = "ranked_scores_heatmap",
    save_format: str = "pdf",
):
    """
    Plots a heatmap of method rankings across datasets and optionally saves the plot.

    Parameters
    ----------
    score_dict : dict
        Keys are dataset names; values are Series of scores with method names as index.
    method_mapping : dict or None
        Ordered mapping {raw_method_name -> display_name}. If provided, it
        *selects*, *orders*, and *renames* methods (in insertion order).
        Takes precedence over `method_order`.
    method_order : list or None
        If provided (and `method_mapping` is None), selects and orders methods by raw names.
    """
    # ── 1) combine scores ───────────────────────────────────────────────────
    all_scores = pd.concat(score_dict.values(), axis=1)
    all_scores.columns = list(score_dict.keys())

    # ── 2) select/order/rename methods ──────────────────────────────────────
    if method_mapping is not None:
        # keep only methods present, in the mapping's order
        available = [m for m in method_mapping if m in all_scores.index]
        all_scores = all_scores.loc[available]
        # rename to display names
        rename_map = {m: method_mapping[m] for m in available}
        all_scores = all_scores.rename(index=rename_map)
    elif method_order is not None:
        valid = [m for m in method_order if m in all_scores.index]
        all_scores = all_scores.reindex(valid)

    # ── 3) compute ranks (higher score = better = rank 1) ───────────────────
    ranked = all_scores.rank(axis=0, ascending=False).astype("Int64")
    heatmap_data = ranked.astype(float)

    # ── 4) cell annotations ─────────────────────────────────────────────────
    annot_text = heatmap_data.applymap(lambda v: str(int(v)) if pd.notna(v) else "-")

    # ── 5) colormap ─────────────────────────────────────────────────────────
    cmap = sns.color_palette("viridis_r", as_cmap=True)
    cmap.set_bad(color="lightgrey")

    # ── 6) plot ─────────────────────────────────────────────────────────────
    with plt.rc_context(rc=custom_rc or {}):
        plt.figure(figsize=figsize)
        ax = sns.heatmap(
            heatmap_data.T,
            annot=annot_text.T,
            fmt="",
            cmap=cmap,
            linewidths=1,
            linecolor="white",
            cbar=False
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_title(title, fontsize=12, pad=10)
        plt.tight_layout()

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            file_path = save_dir / f"{file_name}.{save_format}"
            plt.savefig(file_path, format=save_format)

        plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib as mpl

def plot_resource_usage_heatmap(
    usage_df: pd.DataFrame,
    resource_type: str = "RAM",  # or "VRAM", "Time"
    figsize=(5, 2.5),
    custom_rc=None,
    method_order: list[str] | None = None,          # kept for back-compat
    method_mapping: dict[str, str] | None = None,   # NEW: ordered mapping
    save_dir: Path | None = None,
    title: str = "Resource Usage",
    file_name: str = "resource_usage_heatmap",
    save_format: str = "pdf",
    dataset_display_map: dict[str, str] | None = None,
    *,
    cmap: str | mpl.colors.Colormap = "Oranges",
    vmax_val: float | None = None,
):
    """
    Plots a heatmap of resource usage (RAM, VRAM, or Time) across datasets and methods.

    - If `method_mapping` is provided, it selects, orders, and renames methods
      according to insertion order of the mapping.
    - `cmap` can be a matplotlib colormap name or a Colormap object.
    - For VRAM/RAM the units are assumed MB; set `vmax_val` (in MB) to cap color scale.
    """

    def format_val(val):
        if pd.isna(val):
            return "-"                      # show "-" for NaN
        if resource_type == "VRAM" and np.isclose(val, 0, atol=1e-3):
            return ""                       # VRAM==0 → empty; "CPU" will be overlaid
        if resource_type in {"RAM", "VRAM"}:
            return f"{val:.0f}M" if val < 1024 else f"{val / 1024:.1f}G"
        if resource_type == "Time":
            return f"{int(val // 3600)}h" if val >= 3600 else (
                   f"{int(val // 60)}m" if val >= 60 else f"{int(val)}s")
        return str(val)

    annot_df = usage_df.applymap(format_val)

    # Rename datasets (rows after transpose)
    if dataset_display_map:
        heatmap_df = usage_df.T.rename(index=dataset_display_map)
        annot_df_renamed = annot_df.T.rename(index=dataset_display_map)
    else:
        heatmap_df = usage_df.T
        annot_df_renamed = annot_df.T

    # ---- NEW: select/order/rename methods (columns) ----
    if method_mapping is not None:
        available = [m for m in method_mapping if m in heatmap_df.columns]
        heatmap_df = heatmap_df[available]
        annot_df_renamed = annot_df_renamed[available]

        # rename to display names
        display_names = [method_mapping[m] for m in available]

        # ensure unique column labels (avoid ambiguity in `.loc[dataset, method]`)
        counts = {}
        unique_names = []
        for name in display_names:
            counts[name] = counts.get(name, 0) + 1
            unique_names.append(f"{name} [{counts[name]}]" if counts[name] > 1 else name)

        heatmap_df.columns = unique_names
        annot_df_renamed.columns = unique_names

    elif method_order is not None:
        valid_methods = [m for m in method_order if m in heatmap_df.columns]
        heatmap_df = heatmap_df[valid_methods]
        annot_df_renamed = annot_df_renamed[valid_methods]

    # Identify CPU-only and NaN entries
    cpu_mask = (
        heatmap_df.applymap(lambda val: np.isclose(val, 0, atol=1e-3))
        if resource_type == "VRAM" else pd.DataFrame(False, index=heatmap_df.index, columns=heatmap_df.columns)
    )
    nan_mask = heatmap_df.isna()

    # Colormap
    cmap_obj = mpl.colormaps[cmap].copy() if isinstance(cmap, str) else cmap
    if hasattr(cmap_obj, "set_bad"):
        cmap_obj.set_bad(color="lightgrey")

    # Color limits
    vmin_val = 0 if resource_type in {"RAM", "VRAM"} else None

    with plt.rc_context(rc=custom_rc or {}):
        plt.figure(figsize=figsize)
        ax = sns.heatmap(
            heatmap_df.astype(float),
            annot=annot_df_renamed,
            fmt="",
            cmap=cmap_obj,
            vmin=vmin_val,
            vmax=vmax_val,
            linewidths=1,
            linecolor="white",
            cbar=False,
            annot_kws={"fontsize": 9}
        )

        for row_idx, dataset in enumerate(heatmap_df.index):
            for col_idx, method in enumerate(heatmap_df.columns):
                if cpu_mask.loc[dataset, method]:
                    ax.text(
                        col_idx + 0.5, row_idx + 0.5,
                        "CPU", ha="center", va="center",
                        fontsize=7, color="black"
                    )
                elif nan_mask.loc[dataset, method]:
                    ax.text(
                        col_idx + 0.5, row_idx + 0.5,
                        "-", ha="center", va="center",
                        fontsize=9, color="black"
                    )

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_title(title, fontsize=12, pad=10)
        plt.tight_layout()

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{file_name}.{save_format}"
            plt.savefig(save_path, format=save_format)

        plt.show()
