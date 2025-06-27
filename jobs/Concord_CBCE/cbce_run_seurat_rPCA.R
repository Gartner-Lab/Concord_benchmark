
library(Seurat)
library(monocle3)
pyenv = 'cellpath'

library(anndata)
library(reticulate)
use_condaenv(pyenv, required = TRUE)

cds = readRDS('../../data/CE_CB/for_qin_briggsae_elegans/cds_bg_20240903.rds')
seurat_obj = readRDS("../../data/CE_CB/for_qin_briggsae_elegans/seurat_rpca_integration_joined_20240823.rds")

print("Dataset loaded")

cds_match = cds[,colnames(seurat_obj)]
pmeta = as.data.frame(pData(cds_match))
seurat_obj@meta.data = pmeta


start_time <- Sys.time()

seu.list <- SplitObject(seurat_obj, split.by = "dataset3")
length(seu.list)
##-------------------------------
## 3) rPCA Integration
##-------------------------------
# 3a) Find integration anchors using reciprocal PCA
anchors <- FindIntegrationAnchors(
  object.list = seu.list,
  anchor.features = 10000,
  reduction = "rpca"
)

# 3b) Integrate data
integrated <- IntegrateData(anchorset = anchors)

##-------------------------------
## 4) Record end time and compute elapsed
##-------------------------------
end_time <- Sys.time()
elapsed_time <- end_time - start_time
cat("Integration completed in:", elapsed_time, "\n")

##-------------------------------
## 5) Save results
##-------------------------------
# Save the integrated Seurat object for later use
saveRDS(integrated, file = "../../data/CE_CB/seu_int_QZ/seurat_integrated_rpca.rds")

# Save the runtime to a simple RDS or CSV
saveRDS(elapsed_time, file = "../../data/CE_CB/seu_int_QZ/time_log_seurat_rpca.rds")

print("Integration saved")
