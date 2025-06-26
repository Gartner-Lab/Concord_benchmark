

import numpy as np
import pandas as pd
from scipy.stats import entropy
from concord.utils import Neighborhood

def batch_mixing_analysis(adata, latent_key="latent", batch_key="batch", k=10, n_core=500, core_cells=None):
    """
    Perform batch mixing analysis on the latent space in adata.obsm.

    Parameters:
    -----------
    adata : anndata.AnnData
        AnnData object containing latent embeddings in `adata.obsm[latent_key]` 
        and batch labels in `adata.obs[batch_key]`.
    latent_key : str, optional
        Key in `adata.obsm` containing the latent embeddings. Default is "latent".
    batch_key : str, optional
        Key in `adata.obs` containing batch labels. Default is "batch".
    k : int, optional
        Number of nearest neighbors to consider. Default is 10.
    n_core : int, optional
        Number of core cells to sample uniformly. Default is 500.
    core_cells : list, optional
        List of core cells to use. If not provided, core cells are sampled uniformly. Default is None.
    Returns:
    --------
    pd.DataFrame
        DataFrame containing batch composition in kNN neighborhoods, indexed by core sample indices.
    """

    # Extract latent embeddings and batch labels
    latent = adata.obsm[latent_key]
    batch_labels = adata.obs[batch_key].values

    # Initialize the neighborhood class
    nn = Neighborhood(latent, k=k, use_faiss=True)

    # Sample core cells uniformly
    if core_cells is not None:
        core_samples = core_cells
    else:
        np.random.seed(42)  # for reproducibility
        core_samples = np.random.choice(latent.shape[0], size=min(n_core, latent.shape[0]), replace=False)

    # Get kNN indices
    knn_indices = nn.get_knn(core_samples, k=k, include_self=False)

    # Get all unique batch labels
    unique_batches = np.unique(batch_labels)

    # Collect results in a list (to avoid slow `pd.concat` in a loop)
    results = []

    for i, neighbors in enumerate(knn_indices):
        neighbor_batches = batch_labels[neighbors]

        # Count occurrences of each batch
        batch_counts = pd.Series(neighbor_batches).value_counts(normalize=True).to_dict()
    
        # Ensure all batch categories exist in every row (fill missing ones with 0)
        batch_counts_full = {batch: batch_counts.get(batch, 0) for batch in unique_batches}

        # Compute entropy of batch distribution
        batch_entropy = entropy(list(batch_counts.values()))

        # Store results
        results.append({"core_cell": core_samples[i], "entropy": batch_entropy, **batch_counts_full})

    # Convert to DataFrame
    batch_df = pd.DataFrame(results)


    return batch_df

