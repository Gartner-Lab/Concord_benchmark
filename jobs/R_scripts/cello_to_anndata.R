

library(VisCello)
file_path = "./data/CE_CB/C.elegans_C.briggsae_Embryo_Single_Cell/Cello/eset.rds"
eset <- readRDS(file_path)
ce_genes = fData(eset)[fData(eset)$species == "C.elegans",]
cb_genes = fData(eset)[fData(eset)$species == "C.briggsae",]
shared_genes = intersect(cb_genes$gene_short_name, ce_genes$gene_short_name)

exprs_data <- exprs(eset)
fdata <- fData(eset)
pdata <- pData(eset)

# Extract cell indices for each species
c_briggsae_cells <- pdata$species == "C.briggsae"
c_elegans_cells <- pdata$species == "C.elegans"

# Subset gene data for each species
ce_genes_shared <- ce_genes[match(shared_genes, ce_genes$gene_short_name), ]
cb_genes_shared <- cb_genes[match(shared_genes, cb_genes$gene_short_name), ]

# Subset the expression matrix for each species and shared genes
exprs_ce_shared <- exprs_data[rownames(ce_genes_shared), c_elegans_cells]
exprs_cb_shared <- exprs_data[rownames(cb_genes_shared), c_briggsae_cells]

# Verify the order of genes is consistent across species
if (!all(ce_genes_shared$gene_short_name == cb_genes_shared$gene_short_name)) {
    stop("The gene order for shared genes is not consistent between species.")
}

# Combine the expression matrices by columns (cells from both species)
exprs_shared <- cbind(exprs_cb_shared, exprs_ce_shared)
rownames(exprs_shared) = shared_genes
# Subset the feature data for the shared genes (could use either species' metadata)
fdata_shared <- ce_genes_shared
fdata_shared$ce_rowname = rownames(ce_genes_shared)
fdata_shared$cb_rowname = rownames(cb_genes_shared)
rownames(fdata_shared) = shared_genes
# Combine phenotype data (keeping it as is since it represents both species)
pdata_shared <- rbind(pdata[c_briggsae_cells, ], pdata[c_elegans_cells, ])

# Create a new ExpressionSet with the shared orthologs expression data
eset_shared <- new("ExpressionSet", assayData = assayDataNew("environment", exprs = exprs_shared), 
                             phenoData = AnnotatedDataFrame(pdata_shared),
                             featureData = AnnotatedDataFrame(fdata_shared))

saveRDS(eset_shared, "./data/CE_CB/eset_sharedg.rds")

library(anndata)
library(reticulate)
eset_shared <- readRDS("./data/CE_CB/eset_sharedg.rds")
use_condaenv("py39", required = TRUE)

# Also save the dimension reduction
global_umap <- clist$`Joint global`@proj$`global UMAP [2D]`
global_umap_ordered <- global_umap[colnames(eset_shared),]

adata <- AnnData(
    X = t(exprs(eset_shared)),
    obs = pData(eset_shared),
    var = fData(eset_shared),
)

adata$obsm[["Original_umap"]] <- global_umap_ordered

adata$write_h5ad(paste0("./data/CE_CB/cbce_sharedg.h5ad"))





# After processing with Concord, some updates on clist
eset = readRDS("./data/CE_CB/cello_CBCE_processed_Sep22-1526/eset.rds")
eset$cell_type_or_lineage <- as.character(eset$cell_type)
eset$cell_type_or_lineage[eset$cell_type_or_lineage == "unannotated"] = as.character(eset$lineage[eset$cell_type_or_lineage == "unannotated"])

saveRDS(eset, paste0("./data/CE_CB/cello_CBCE_processed_Sep22-1526/", "eset.rds"))

# Attach old result
clist = readRDS("./data/CE_CB/cello_CBCE_processed_Sep22-1526/clist.rds")
clist$`All cells`@proj[["Original_UMAP"]] = global_umap_ordered
saveRDS(clist, "./data/CE_CB/cello_CBCE_processed_Sep22-1526/clist.rds")




