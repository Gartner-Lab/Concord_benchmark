#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(Seurat)
  library(Biobase)
  library(optparse)
  library(peakRAM)
  library(readr)        # write_tsv()
})

## ───────────── CLI ─────────────
opt <- parse_args(OptionParser(option_list = list(
  make_option("--timestamp", type = "character",
              default = format(Sys.time(), "%m%d-%H%M"),
              help    = "timestamp string  [auto if omitted]")
)))
FILE_SUFFIX <- opt$timestamp

## ───────────── Config (from generator) ─────────────
PROJ_NAME   <- "cel_packerN2"
ESET_DIR    <- "../../data/cel_packerN2/viscello_ds_cel_packerN2"          # contains eset.rds
METHOD      <- "seurat_cca"            # e.g.  seurat_rpca / seurat_cca
LATENT_DIM  <- 300
BATCH_KEY   <- "batch"

SAVE_DIR <- file.path("..","..","save", PROJ_NAME, paste0(METHOD, "_", FILE_SUFFIX))
dir.create(SAVE_DIR, recursive = TRUE, showWarnings = FALSE)

cat("▶ Loading ExpressionSet …\n")
eset <- readRDS(file.path(ESET_DIR, "eset.rds"))
seu  <- CreateSeuratObject(
          counts    = Biobase::exprs(eset),
          meta.data = Biobase::pData(eset))

## ───────────── Split & normalise ─────────────
seu.list <- SplitObject(seu, split.by = BATCH_KEY)
seu.list <- lapply(seu.list, NormalizeData, verbose = FALSE)

anchor_feats <- rownames(seu)        # use all genes

## ───────────── Integration with RAM profiling ─────────────
ram_res <- peakRAM({
  if (grepl("rpca",  METHOD)) {
    seu.list <- lapply(seu.list, function(x) {
      x <- ScaleData(x, features = anchor_feats, verbose = FALSE)
      x <- RunPCA(x,  features = anchor_feats,
                  npcs = LATENT_DIM, verbose = FALSE)
      x                                    # return
    })
    anchors <- FindIntegrationAnchors(
        seu.list, anchor.features = anchor_feats,
        reduction = "rpca", dims = 1:LATENT_DIM)
  } else if (grepl("cca", METHOD)) {
    anchors <- FindIntegrationAnchors(
        seu.list, anchor.features = anchor_feats,
        reduction = "cca",  dims = 1:LATENT_DIM)
  } else {
    stop(paste("Unsupported METHOD:", METHOD))
  }
  integrated <<- IntegrateData(anchors, dims = 1:LATENT_DIM)
})

elapsed_sec <- ram_res$Elapsed_Time_sec
# peakRAM can overflow on some kernels → fall back to Total_RAM if necessary
peak_mb     <- ram_res$Peak_RAM_Used_MiB
if (is.na(peak_mb) || peak_mb > 1e6) peak_mb <- ram_res$Total_RAM_Used_MiB
vram_mb     <- 0

cat(sprintf("✔  Integration %.1f s  |  RAM %.0f MiB\n",
            elapsed_sec, peak_mb))

## ───────────── Embedding export ─────────────
DefaultAssay(integrated) <- "integrated"
integrated <- ScaleData(integrated, verbose = FALSE)
integrated <- RunPCA(integrated, npcs = LATENT_DIM, verbose = FALSE)

embed_mat <- Embeddings(integrated, "pca")
write.table(embed_mat,
            file = file.path(
              SAVE_DIR,
              paste0(METHOD, "_embedding_", FILE_SUFFIX, ".tsv")),
            sep = "\t", quote = FALSE, col.names = NA)

## ───────────── Save outputs ─────────────
saveRDS(integrated,
        file = file.path(
          SAVE_DIR,
          paste0("seurat_integrated_", METHOD, "_", FILE_SUFFIX, ".rds")))

# 1) concise benchmark (matches Python side)
bench_df <- data.frame(method   = METHOD,
                       time_sec = elapsed_sec,
                       ram_MB   = peak_mb,
                       vram_MB  = vram_mb)
write_tsv(bench_df,
          file = file.path(
            SAVE_DIR,
            paste0("benchmark_log_", METHOD, "_", FILE_SUFFIX, ".tsv")))

# 2) full peakRAM profile (peak + total + all metadata)
write_tsv(ram_res,
          file = file.path(
            SAVE_DIR,
            paste0("ram_profile_", METHOD, "_", FILE_SUFFIX, ".tsv")))

cat("🎉  All outputs written to", SAVE_DIR, "\n")
