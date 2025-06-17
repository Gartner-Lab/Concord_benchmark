import re
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

import concord as ccd


def add_embeddings(adata: AnnData, proj_name: str, methods: list[str]) -> AnnData:
    """
    Add the latest integration embeddings and UMAPs to adata.obsm.
    
    Parameters:
        adata: AnnData object (already loaded)
        proj_name: name of the project (e.g. 'pbmc_DeSimone_10K')
        methods: list of method names (e.g. ['scvi', 'harmony'])
        
    Returns:
        Modified AnnData object with obsm['X_{method}'] and obsm['{method}_UMAP']
    """
    save_root = Path(f"../save/{proj_name}")

    def extract_timestamp(folder_name, method):
        match = re.match(rf"{method}_(\d{{4}}-\d{{4}})", folder_name)
        return match.group(1) if match else None

    for method in methods:
        candidates = [
            p for p in save_root.glob(f"{method}_*") if p.is_dir() and extract_timestamp(p.name, method)
        ]
        if not candidates:
            print(f"[⚠️ Warning] No folder found for method: {method}")
            continue

        latest_dir = sorted(candidates, key=lambda p: extract_timestamp(p.name, method), reverse=True)[0]
        timestamp = extract_timestamp(latest_dir.name, method)
        embedding_file = latest_dir / f"{method}_embedding_{timestamp}.tsv"

        if not embedding_file.exists():
            print(f"[⚠️ Warning] Missing embedding file: {embedding_file}")
            continue

        df = pd.read_csv(embedding_file, sep="\t", index_col=0)

        if not df.index.equals(adata.obs_names):
            print(f"[❌ Error] obs_names mismatch for: {method}")
            continue

        adata.obsm[f"X_{method}"] = df.values
        print(f"✅ obsm['X_{method}'] loaded")

        ccd.ul.run_umap(adata, source_key=f"X_{method}", result_key=f"{method}_UMAP")
        print(f"✅ obsm['{method}_UMAP'] computed")

    return adata

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
