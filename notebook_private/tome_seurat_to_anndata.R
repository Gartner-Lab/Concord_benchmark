
library(Seurat)

seu_obj <- readRDS("data/mouse_tome/seurat_object_E7.5.rds")
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
