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



from pathlib import Path
from typing import Optional, Mapping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator


def plot_benchmark_performance(
    bench_df: pd.DataFrame,
    title: str | None = None,
    figsize: tuple[int, int] = (10, 4.5),
    dpi: int = 300,
    save_path: Optional[Path] = None,
    rc: dict | None = None,
    metric_scale: Mapping[str, str] | None = None,  # {'ram_MB': 'log', 'vram_MB': 'auto', ...}
    log_span_thresh: float = 20.0,                  # span≥→log when scale='auto'
):
    """
    Plot run-time, RAM, and VRAM usage for each integration method.

    Parameters
    ----------
    bench_df : pd.DataFrame
        Must contain columns: 'method', 'time_sec', 'ram_MB', 'vram_MB'.
    title : str, optional
        Figure title.
    figsize, dpi : figure size & resolution.
    save_path : Path, optional
        Save figure if provided.
    rc : dict, optional
        Matplotlib rc overrides (passed to rc_context).
    metric_scale : mapping, optional
        Override per-metric axis scale. Valid: {'linear','log','auto'}.
        Missing keys fall back to defaults: time=log, RAM=linear, VRAM=linear.
    log_span_thresh : float
        When scale='auto', switch to log if (max/min) span ≥ this.
    """

    # ------------------------------------------------------------------ #
    # defaults + user overrides
    # ------------------------------------------------------------------ #
    scale_map = {
        "time_sec": "log",
        "ram_MB":   "linear",
        "vram_MB":  "linear",
    }
    if metric_scale:
        scale_map.update(metric_scale)

    # autoscale units (MiB→GiB) so tick labels stay short ----------------
    def _auto_unit(vals_mb: np.ndarray, label_mib: str):
        # Use GiB if any value ≥ 2048 MiB
        if np.nanmax(vals_mb) >= 2048:
            return vals_mb / 1024.0, label_mib.replace("MiB", "GiB")
        return vals_mb, label_mib

    # metric spec: (column, label, divider)  (divider will be replaced if GiB)
    metrics = [
        ("time_sec", "Run-time (h)", 3600.0),
        ("ram_MB",   "RAM (MiB)",       1.0),
        ("vram_MB",  "VRAM (MiB)",      1.0),
    ]

    cpu_methods = bench_df.loc[bench_df["vram_MB"] == 0, "method"].tolist()
    colour = "steelblue"

    # ------------------------------------------------------------------ #
    # plotting
    # ------------------------------------------------------------------ #
    with plt.rc_context(rc or {}):
        fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi, constrained_layout=True)
        fig, axes = plt.subplots(
            1, 3, figsize=figsize, dpi=dpi, constrained_layout=True
        )

        for ax, (key, xlabel, div, scale) in zip(axes, metrics):
            df = bench_df.copy()
            df["_sort_val"] = df[key].fillna(np.inf)  # Use np.inf so NaN sorts last
            df = df.sort_values("_sort_val", ascending=True)
            vals = df[key] / div
        for ax, (key, base_label, divisor) in zip(axes, metrics):
            df = bench_df.sort_values(key, ascending=True).copy()

            # convert units if needed
            if key.endswith("_MB"):
                vals_raw = df[key].to_numpy(dtype=float)
                vals_conv, label = _auto_unit(vals_raw, base_label)
                divisor = 1.0  # already converted
            else:
                vals_raw = df[key].to_numpy(dtype=float)
                vals_conv = vals_raw / divisor
                label = base_label

            y = np.arange(len(df))

            # decide scale -----------------------------------------------------
            scale = scale_map.get(key, "linear")
            if scale == "auto":
                pos = vals_conv[vals_conv > 0]
                span = (pos.max() / pos.min()) if pos.size else 1.0
                scale = "log" if span >= log_span_thresh else "linear"

            # handle zeros for log plotting ------------------------------------
            if scale == "log":
                pos = vals_conv[vals_conv > 0]
                if pos.size == 0:
                    # all zeros? fall back to linear to avoid log underflow
                    scale = "linear"
                    vals_plot = vals_conv
                else:
                    min_pos = pos.min()
                    floor = min_pos / 10.0  # show tiny stub for zeros
                    vals_plot = np.where(vals_conv > 0, vals_conv, floor)
            else:
                vals_plot = vals_conv

            # draw bars --------------------------------------------------------
            ax.barh(y, vals_plot, color=colour)
            ax.set_yticks(y)
            ax.set_yticklabels(df["method"], fontsize=tick_fontsize)
            ax.invert_yaxis()
            ax.set_xlabel(xlabel, fontsize=label_fontsize)
            ax.tick_params(axis='x', labelsize=tick_fontsize)
            ax.set_xlabel(label)
            ax.grid(axis="x", ls=":", alpha=.4)

            df = df.drop(columns="_sort_val")

            for ypos, val, method, raw_val in zip(y, vals, df["method"], df[key]):
                if pd.isna(raw_val):
                    ax.text(0.02, ypos, "NaN", va="center", fontsize=8, color="black",
                            transform=ax.get_yaxis_transform())

            # scale-specific ticks --------------------------------------------
            if scale == "log":
                ax.set_xscale("log")
                min_val = max(0.001, np.nanmin(vals[vals > 0]))
                max_val = np.nanmax(vals)
                tick_min = 10 ** int(np.floor(np.log10(min_val)))
                tick_max = 10 ** int(np.ceil(np.log10(max_val)))
                log_ticks = [x for x in [0.001, 0.01, 0.1, 1, 10, 100] if tick_min <= x <= tick_max]
                ax.set_xticks(log_ticks)
                ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
                # nice decade ticks between data min/max
                pos = vals_plot[vals_plot > 0]
                tick_min = 10 ** np.floor(np.log10(pos.min()))
                tick_max = 10 ** np.ceil(np.log10(pos.max()))
                decades = 10 ** np.arange(np.log10(tick_min), np.log10(tick_max) + 1)
                ax.set_xticks(decades)
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            else:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=False, prune="both"))

            # annotate CPU-only on VRAM panel ---------------------------------
            if key == "vram_MB":
                # use original (converted) value for offset
                vmax = np.nanmax(vals_plot)
                text_offset = vmax * 0.02 if vmax > 0 else 1
                for ypos, val, meth in zip(y, vals_plot, df["method"]):
                    if meth in cpu_methods:
                        # place text just to the right of the bar
                        ax.text(val + text_offset, ypos, "CPU only",
                                va="center", fontsize=8, color="black")

        axes[0].set_ylabel("Integration Method", fontsize=label_fontsize)
        axes[0].tick_params(axis='y', labelsize=tick_fontsize)

        if title:
            fig.suptitle(title, fontsize=14, fontweight="bold")

            fig.suptitle(title, fontsize=16, fontweight="bold")

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
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
