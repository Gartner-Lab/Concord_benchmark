#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(Seurat)
  library(Biobase)     # for ExpressionSet
  library(optparse)
  library(peakRAM)
  library(readr)
})

## ---- CLI ----
opt <- optparse::parse_args(
  optparse::OptionParser(
    option_list = list(
      optparse::make_option("--timestamp", type="character",
                            help="timestamp string")
    )
  )
)
FILE_SUFFIX <- opt$timestamp

## ---- Config (filled by generator) ----
PROJ_NAME <- "CE_CB"
ESET_DIR  <- "../data/cel_packerN2/viscello_ds_CE_CB"           # directory produced by converter
METHOD    <- "seurat_cca"             # rpca / cca …

DATA_DIR <- ESET_DIR                # eset.rds lives right here
SAVE_DIR <- file.path("..","..","save", PROJ_NAME,
                      paste0(METHOD, "_", FILE_SUFFIX))
dir.create(SAVE_DIR, recursive = TRUE, showWarnings = FALSE)

cat("Reading ExpressionSet:", file.path(DATA_DIR, "eset.rds"), "\n")
eset <- readRDS(file.path(DATA_DIR, "eset.rds"))

counts  <- Biobase::exprs(eset)
meta_df <- Biobase::pData(eset)
seu <- CreateSeuratObject(counts = counts, meta.data = meta_df)
cat("ExpressionSet → Seurat:", ncol(seu), "cells\n")

## ---- split & integrate ----
seu.list <- SplitObject(seu, split.by = "batch")
for (i in seq_along(seu.list)) {
  seu.list[[i]] <- NormalizeData(seu.list[[i]], verbose = FALSE)
  seu.list[[i]] <- FindVariableFeatures(seu.list[[i]], selection.method = "vst",
                                        nfeatures = 2000, verbose = FALSE)
}
anchor_feats <- min(10000, nrow(seu))

ram_res <- peakRAM({
  if (METHOD == "rpca") {
    anchors <- FindIntegrationAnchors(
      object.list     = seu.list,
      anchor.features = anchor_feats,
      reduction       = "rpca", dims = 1:50
    )
  } else if (METHOD == "cca") {
    anchors <- FindIntegrationAnchors(
      object.list     = seu.list,
      anchor.features = anchor_feats,
      reduction       = "cca", dims = 1:50
    )
  } else {
    stop(paste("Unsupported METHOD:", METHOD))
  }
  integrated <<- IntegrateData(anchorset = anchors, dims = 1:50)
})

elapsed_sec <- ram_res$Elapsed_Time_sec
ram_mb      <- ram_res$Peak_RAM_Used_MiB
vram_mb     <- 0
cat(sprintf("Done: %.1f s | peak RAM %.0f MiB\n", elapsed_sec, ram_mb))

## ---- Embedding & export ----
DefaultAssay(integrated) <- "integrated"
integrated <- ScaleData(integrated, verbose = FALSE)
integrated <- RunPCA(integrated, npcs = 50, verbose = FALSE)

embed_mat <- Embeddings(integrated, "pca")
embed_path <- file.path(SAVE_DIR,
                        paste0(METHOD, "_embedding_", FILE_SUFFIX, ".tsv"))
write.table(embed_mat, embed_path, sep = "\t", quote = FALSE, col.names = NA)
cat("Saved embedding →", embed_path, "\n")

## ---- Save object & benchmark ----
saveRDS(integrated,
        file = file.path(SAVE_DIR,
               paste0("seurat_integrated_", METHOD, "_", FILE_SUFFIX, ".rds")))

log_df <- data.frame(method   = METHOD,
                     time_sec = elapsed_sec,
                     ram_MB   = ram_mb,
                     vram_MB  = vram_mb)
write_tsv(log_df,
          file = file.path(SAVE_DIR,
                 paste0("benchmark_log_", FILE_SUFFIX, ".tsv")))
cat("Outputs written to", SAVE_DIR, "\n")
