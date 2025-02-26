
library(VisCello)

mtx <- readRDS("./data/mouse_organogenesis/gene_count_cleaned_sampled_100k.RDS")
fmeta <- read.csv("./data/mouse_organogenesis/gene_annotate.csv")
pmeta <- read.csv("./data/mouse_organogenesis/cell_annotate.csv")
rownames(fmeta) = fmeta$gene_id
rownames(pmeta) <- pmeta$sample

pmeta <- pmeta[colnames(mtx), ]
all(colnames(mtx) %in% rownames(pmeta))
identical(rownames(fmeta), rownames(mtx))

eset <- new("ExpressionSet", assayData = assayDataNew("environment", exprs = mtx), 
                   phenoData = AnnotatedDataFrame(pmeta),
                   featureData = AnnotatedDataFrame(fmeta))

saveRDS(eset, "./data/mouse_organogenesis/eset_100k.rds")


library(anndata)
library(reticulate)
eset <- readRDS("./data/mouse_organogenesis/eset_100k.rds")
use_condaenv("py39", required = TRUE)

# Also save the dimension reduction
main_umap <- pData(eset)[c('Main_trajectory_umap_1', 'Main_trajectory_umap_2', 'Main_trajectory_umap_3')]
main_umap_refined = pData(eset)[c('Main_trajectory_refined_umap_1', 'Main_trajectory_refined_umap_2', 'Main_trajectory_refined_umap_3')]

adata <- AnnData(
    X = t(exprs(eset)),
    obs = pData(eset),
    var = fData(eset),
)

adata$obsm[["Original_main_umap"]] <- main_umap
adata$obsm[["Original_main_umap_refined"]] <- main_umap_refined

adata$write_h5ad(paste0("./data/mouse_organogenesis/adata_sampled_100k.h5ad"))

