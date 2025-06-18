



import numpy as np
import pandas as pd
import anndata as ad

def simulate_expression_data_groups(num_genes=6, num_cells=12, num_groups=2, mean_expression=10, noise_std=1.0, 
                                    p_gene_nonspecific=0.1, group_key='group', permute=False, seed=42):
    """
    Simulates a dataset with gene expression where each cell group expresses a specific set of group-specific genes,
    and a certain fraction of non-specific genes shared across groups. Returns an AnnData object containing the expression
    matrix and group information.
    
    Parameters:
        num_genes (int): Total number of genes in the dataset.
        num_cells (int): Total number of cells in the dataset.
        num_groups (int): Number of groups to split both the genes and cells into.
        mean_expression (float): Mean expression level for expressed genes in each group.
        noise_std (float): Standard deviation of the Gaussian noise added to each gene.
        p_gene_nonspecific (float): Fraction of genes to be non-specific and shared across all cell groups.
        seed (int): Random seed for reproducibility.
    
    Returns:
        adata (anndata.AnnData): AnnData object containing the expression matrix and group information.
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Calculate the base number of genes and cells per group
    genes_per_group = num_genes // num_groups
    cells_per_group = num_cells // num_groups
    
    # Calculate the number of non-specific genes
    num_nonspecific_genes = int(num_genes * p_gene_nonspecific)
    
    # Initialize the expression matrix with zeros
    expression_matrix = np.zeros((num_cells, num_genes))
    cell_groups = []  # To store the cell group information
    
    # Loop over each group to assign mean expression and add Gaussian noise
    for group in range(num_groups):
        # Determine the ranges for cells and genes for the current group
        cell_start = group * cells_per_group
        cell_end = (cell_start + cells_per_group) if group < num_groups - 1 else num_cells
        
        gene_start = group * genes_per_group
        gene_end = (gene_start + genes_per_group) if group < num_groups - 1 else num_genes
        
        # Assign expression levels for group-specific genes
        expression_matrix[cell_start:cell_end, gene_start:gene_end] = (
            mean_expression + np.random.normal(0, noise_std, (cell_end - cell_start, gene_end - gene_start))
        )
        
        # Randomly select non-specific genes
        other_genes = np.setdiff1d(np.arange(num_genes), np.arange(gene_start, gene_end))
        nonspecific_gene_indices = np.random.choice(other_genes, num_nonspecific_genes, replace=False)
    
        # Assign expression levels for non-specific genes shared across all groups
        expression_matrix[cell_start:cell_end, nonspecific_gene_indices] = (
            mean_expression + np.random.normal(0, noise_std, (cell_end - cell_start, num_nonspecific_genes))
        )
        
        # Record the group information for these cells
        cell_groups.extend([f"{group_key}_{group+1}"] * (cell_end - cell_start))
    
    # Create an AnnData object
    obs = pd.DataFrame({f"{group_key}": cell_groups}, index=[f"Cell_{i+1}" for i in range(num_cells)])
    var = pd.DataFrame(index=[f"Gene_{i+1}" for i in range(num_genes)])
    adata = ad.AnnData(X=expression_matrix, obs=obs, var=var)
    
    if permute:
        # Permute the cell order
        adata = adata[np.random.permutation(adata.obs_names), :]
    return adata




def simulate_gradual_change_expression(num_genes=10, num_cells=10, direction='both', time_key = 'time', max_expression=1, noise_level=0.1):
    """Simulates gradual increase or decrease in gene expression."""
    # Create an empty matrix for expression
    expression_matrix = np.zeros((num_cells, num_genes))

    for i in range(num_genes):
        # Gradual increase or decrease based on direction
        noise = np.random.normal(0, noise_level, num_cells)
        if direction == 'increase':
            expression_matrix[:, i] = np.linspace(0, max_expression, num_cells) + noise
        elif direction == 'decrease':
            expression_matrix[:, i] = np.linspace(max_expression, 0, num_cells) + noise
        elif direction == 'both':
            if i >= num_genes//2:
                expression_matrix[:, i] = np.linspace(0, max_expression, num_cells) + noise
            else:
                expression_matrix[:, i] = np.linspace(max_expression, 0, num_cells) + noise
                
    # Make adata out of the expression matrix
    obs = pd.DataFrame(index=[f"Cell_{i+1}" for i in range(num_cells)])
    obs[time_key] = np.linspace(0, 1, num_cells)   
    var = pd.DataFrame(index=[f"Gene_{i+1}" for i in range(num_genes)])
    adata = ad.AnnData(X=expression_matrix, obs=obs, var=var)

    return adata




def simulate_gradual_gene_shift(num_genes=10, num_cells=100, direction='increase', group_size = 3, gap_size=5,
                                       mean_expression=10, noise_std=1.0, seed=42):
    """
    Simulates gene expression where genes are gradually turned on or off sequentially over pseudotime.

    Parameters:
        num_genes (int): Total number of genes.
        num_cells (int): Total number of cells (or steps) across pseudotime.
        direction (str): Direction of change: 'increase', 'decrease', or 'both'.
        steps_per_gene (int): Number of cells (or steps) for each gene to fully turn on or off.
        mean_expression (float): Mean expression level for genes when fully on.
        noise_std (float): Standard deviation of Gaussian noise added to gene expression.
        seed (int): Random seed for reproducibility.
    
    Returns:
        adata (anndata.AnnData): AnnData object with the simulated expression matrix and pseudotime information.
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    assert(group_size <= num_genes)
    assert(gap_size <= num_cells)

    step_size = num_cells // gap_size

    # Initialize the expression matrix with zeros
    expression_matrix = np.zeros((num_cells, num_genes))
    
    # Pseudotime or cell indices
    pseudotime = np.arange(num_cells)
    
    ngroups = num_genes // group_size + 1
    mid = num_genes // 2 // group_size

    # Loop over each gene and apply gradual increase/decrease
    for i in range(ngroups):
        gene_idx = np.arange(min(i*group_size, num_genes), min((i+1)*group_size, num_genes))
        start = min(i * gap_size, num_cells)  # When the gene starts turning on/off
        end = min(start + step_size, num_cells)  # When the gene reaches full expression
        
        if direction == 'increase':
            # Gradually increase gene expression
            expression_matrix[start:end, gene_idx] = np.linspace(0, mean_expression, end - start)[:, None]
            expression_matrix[end:, gene_idx] = mean_expression

        elif direction == 'decrease':
            # Gradually decrease gene expression
            expression_matrix[:start, gene_idx] = mean_expression
            expression_matrix[start:end, gene_idx] = np.linspace(mean_expression, 0, end - start)[:, None]
        
        elif direction == 'both':
            # Half the genes increase, half decrease
            if i >= mid:
                expression_matrix[start:end, gene_idx] = np.linspace(0, mean_expression, end - start)[:, None]
                expression_matrix[end:, gene_idx] = mean_expression
            else:
                expression_matrix[:start, gene_idx] = mean_expression
                expression_matrix[start:end, gene_idx] = np.linspace(mean_expression, 0, end - start)[:, None]
        
        # Add noise to the expression levels
        expression_matrix[:, gene_idx] += np.random.normal(0, noise_std, (num_cells, len(gene_idx)))
    
      # Create an AnnData object
    obs = pd.DataFrame({'time': pseudotime}, index=[f"Cell_{i+1}" for i in range(num_cells)])
    var = pd.DataFrame(index=[f"Gene_{i+1}" for i in range(num_genes)])
    adata = ad.AnnData(X=expression_matrix, obs=obs, var=var)
    
    return adata




def combine_adatas_by_gene_dimension(adata1, adata2, reindex_cells=True):
    if reindex_cells:
        # Reindex the cells to match cell names
        adata1.obs_names = [f"cell_{i}" for i in range(adata1.n_obs)]
        adata2.obs_names = [f"cell_{i}" for i in range(adata2.n_obs)]
    else:
        assert np.all(adata1.obs_names == adata2.obs_names), "Cell names must be the same in both AnnData objects."
    
    # Rename genes to indicate origin
    adata1.var.index = [f"{gene}_1" for gene in adata1.var.index]
    adata2.var.index = [f"{gene}_2" for gene in adata2.var.index]
    
    # Concatenate gene dimensions (X matrices)
    combined_X = np.hstack([adata1.X, adata2.X])
    
    # Combine the obs dataframes to include both cell type and batch information
    combined_obs = pd.concat([adata1.obs, adata2.obs], axis=1)
    
    # Combine the var dataframes to include all genes
    combined_var = pd.concat([adata1.var, adata2.var])
    
    # Create the combined AnnData object
    combined_adata = ad.AnnData(X=combined_X, obs=combined_obs, var=combined_var)
    
    return combined_adata



def permute_and_concatenate_adata_independent(adata, x=10, seed=42):
    """
    Create a new AnnData by permuting cells independently for each repetition 
    and concatenating the permuted adata `x` times.

    Parameters:
        adata (AnnData): The original AnnData object.
        x (int): Number of times to independently permute and concatenate the data.
        seed (int): Random seed for reproducibility.

    Returns:
        AnnData: New AnnData object with independently permuted cells concatenated `x` times.
    """
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Store all independently permuted adata objects
    permuted_adatas = []
    
    for i in range(x):
        # Generate a random permutation of cell indices for each repetition
        permuted_indices = np.random.permutation(adata.shape[0])
        adata_permuted = adata[permuted_indices]
        # Apply the random permutation to the adata and store the result
        permuted_adatas.append(adata_permuted)
    # Concatenate all the independently permuted adata objects
    concatenated_adata = permuted_adatas[0].concatenate(permuted_adatas[1:], batch_key='datacopy')
    print(concatenated_adata.obs['batch'].unique())
    return concatenated_adata



