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

def plot_benchmark_performance(
        bench_df: pd.DataFrame,
        title: str = None,
        figsize: tuple[int, int] = (10, 5),
        dpi: int = 300,
        save_path: Optional[Path] = None,
        rc: dict | None = None,
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, FuncFormatter, MaxNLocator

    metrics = [
        ("time_sec", "Run-time (h)", 3600, "log"),
        ("ram_MB", "RAM (MiB)", 1, "linear"),
        ("vram_MB", "VRAM (MiB)", 1, "linear"),
    ]

    cpu_methods = bench_df.loc[bench_df["vram_MB"] == 0, "method"].tolist()
    colour = "steelblue"

    with plt.rc_context(rc or {}):
        fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)

        for ax, (key, xlabel, div, scale) in zip(axes, metrics):
            df = bench_df.sort_values(key, ascending=True).copy()
            vals = df[key] / div
            y = np.arange(len(df))

            ax.barh(y, vals, color=colour)
            ax.set_yticks(y)
            ax.set_yticklabels(df["method"])
            ax.invert_yaxis()
            ax.set_xlabel(xlabel)
            ax.grid(axis="x", ls=":", alpha=.4)

            if scale == "log":
                ax.set_xscale("log")
                min_val = max(0.001, np.nanmin(vals[vals > 0]))
                max_val = np.nanmax(vals)
                tick_min = 10**int(np.floor(np.log10(min_val)))
                tick_max = 10**int(np.ceil(np.log10(max_val)))
                log_ticks = [x for x in [0.001, 0.01, 0.1, 1, 10, 100] if tick_min <= x <= tick_max]
                ax.set_xticks(log_ticks)
                ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
            else:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True, prune="both"))


            if key == "vram_MB":
                text_offset = vals.max() * 0.02 if vals.max() > 0 else 1
                for ypos, val, meth in zip(y, vals, df["method"]):
                    if meth in cpu_methods:
                        ax.text(val + text_offset, ypos, "CPU only",
                                va="center", fontsize=8, color="black")

        axes[0].set_ylabel("Integration Method")
        if title:
            fig.suptitle(title, fontsize=16, fontweight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
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
