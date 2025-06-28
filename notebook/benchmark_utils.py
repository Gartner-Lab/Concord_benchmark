from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
from anndata import AnnData
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
import concord as ccd
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
        if not df.index.equals(adata.obs_names):
            print(f"[❌ Error] obs_names mismatch for {method}")
            continue

        adata.obsm[f"{method}"] = df.values
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

        rows.append(pd.read_csv(log_tsv, sep="\t"))

    if not rows:
        raise RuntimeError("No benchmark logs were read.")

    return pd.concat(rows, ignore_index=True)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

def plot_benchmark_performance(
        bench_df: pd.DataFrame,
        title: str = None,
        figsize: tuple[int, int] = (12, 4),
        dpi: int = 300,
        save_path: Optional[Path] = None,
        rc: dict | None = None,
):
    """
    Three horizontal bar-plots (time, RAM, VRAM) – each metric sorted
    best-to-worst.  Bars are a single colour; methods that ran without a
    GPU show “CPU only” next to their 0-MiB VRAM bar.
    """
    metrics = [
        ("time_sec", "Run-time (h)", 3600,
         lambda v: np.arange(0, np.ceil(v.max()) + 1, 1)),
        ("ram_MB",   "RAM (MiB)",    1,
         lambda v: np.linspace(0, v.max(), 5)),
        ("vram_MB",  "VRAM (MiB)",   1,
         lambda v: np.linspace(0, v.max(), 5)),
    ]

    cpu_methods = bench_df.loc[bench_df["vram_MB"] == 0, "method"].tolist()
    colour      = "steelblue"

    with plt.rc_context(rc or {}):
        fig, axes = plt.subplots(1, 3, sharey=False,
                                 figsize=figsize, dpi=dpi)

        for ax, (key, xlabel, div, tick_fun) in zip(axes, metrics):
            df = bench_df.sort_values(key, ascending=True).copy()
            vals = df[key] / div
            y    = np.arange(len(df))

            ax.barh(y, vals, color=colour)
            ax.set_yticks(y)
            ax.set_yticklabels(df["method"])
            ax.invert_yaxis()
            ax.set_xlabel(xlabel)
            ax.set_xticks(tick_fun(vals))
            ax.grid(axis="x", ls=":", alpha=.4)

            # ── annotate CPU-only bars just for the VRAM panel ───────────────
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





def run_scib_benchmark(
    adata: AnnData,
    embedding_keys: list,
    batch_key: str = "batch",
    label_key: str = "cell_type",
    n_jobs: int = 4,
) -> Benchmarker:
    """
    Run scib-metrics benchmark on given embeddings.

    Parameters:
    - adata: AnnData object with embeddings in .obsm
    - embedding_keys: list of .obsm keys to evaluate (e.g. ['Harmony', 'scVI', 'Concord'])
    - batch_key: obs column for batch
    - label_key: obs column for cell type labels
    - n_jobs: number of CPU cores to use (-1 = all cores, default = 4)

    Returns:
    - Benchmarker object (use .get_results() or .plot_results_table())
    """
    bm = Benchmarker(
        adata=adata,
        batch_key=batch_key,
        label_key=label_key,
        bio_conservation_metrics=BioConservation(),
        batch_correction_metrics=BatchCorrection(),
        embedding_obsm_keys=embedding_keys,
        n_jobs=n_jobs,
    )
    bm.benchmark()
    return bm
