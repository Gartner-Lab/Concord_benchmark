


#!/usr/bin/env python
# coding: utf-8
"""
Generate job-submission (.sh) and R-runner (.R) files that:
  • read an .h5ad,
  • convert it to a Seurat object via reticulate,
  • perform Seurat integration (rpca | cca | …),
  • time the run & write *.rds logs.

Usage (identical to your Python generator):
  python generate_seurat_scripts.py \
      --proj_name CE_CB \
      --eset_dir ./generated_data/viscello_ds_CE_CB/ \
      --methods rpca cca \
      --batch_key dataset3 \
      --state_key cell_type \
      --latent_dim 50 \
      --device auto \
      --mem 16G --scratch 100G --runtime 04:00:00 \
      --conda_env cellpath \
      --output_dir ./generated_scripts
"""
import argparse, json, textwrap
from pathlib import Path


R_TEMPLATE = r'''#!/usr/bin/env Rscript
suppressPackageStartupMessages({{
  library(Seurat)
  library(Biobase)
  library(optparse)
  library(peakRAM)
  library(readr)        # write_tsv()
}})

## ───────────── CLI ─────────────
opt <- parse_args(OptionParser(option_list = list(
  make_option("--timestamp", type = "character",
              default = format(Sys.time(), "%m%d-%H%M"),
              help    = "timestamp string  [auto if omitted]")
)))
FILE_SUFFIX <- opt$timestamp

## ───────────── Config (from generator) ─────────────
PROJ_NAME   <- "{proj_name}"
ESET_DIR    <- "{eset_dir}"          # contains eset.rds
METHOD      <- "{method}"            # e.g.  seurat_rpca / seurat_cca
LATENT_DIM  <- {latent_dim}
BATCH_KEY   <- "{batch_key}"

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
ram_res <- peakRAM({{
  if (grepl("rpca",  METHOD)) {{
    seu.list <- lapply(seu.list, function(x) {{
      x <- ScaleData(x, features = anchor_feats, verbose = FALSE)
      x <- RunPCA(x,  features = anchor_feats,
                  npcs = LATENT_DIM, verbose = FALSE)
      x                                    # return
    }})
    anchors <- FindIntegrationAnchors(
        seu.list, anchor.features = anchor_feats,
        reduction = "rpca", dims = 1:LATENT_DIM)
  }} else if (grepl("cca", METHOD)) {{
    anchors <- FindIntegrationAnchors(
        seu.list, anchor.features = anchor_feats,
        reduction = "cca",  dims = 1:LATENT_DIM)
  }} else {{
    stop(paste("Unsupported METHOD:", METHOD))
  }}
  integrated <<- IntegrateData(anchors, dims = 1:LATENT_DIM)
}})

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
'''




# ──────────────────────────────────────────────────
SH_TEMPLATE = """#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -r y
#$ -pe smp 1
#$ -l mem_free={mem}
#$ -l scratch={scratch}
#$ -l h_rt={runtime}

echo "Running on: $(hostname)"

module load CBI miniforge3/24.3.0-0
conda activate cellpath
export LD_LIBRARY_PATH=/wynton/home/gartner/zhuqin/.conda/envs/cellpath/lib:$LD_LIBRARY_PATH

Rscript cbce_run_seurat_rPCA.R

TIMESTAMP=$(date +'%m%d-%H%M')
Rscript {script_name}.R --timestamp $TIMESTAMP
"""

# ──────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--proj_name',      required=True)
    p.add_argument('--eset_dir', required=True,
               help="directory that contains eset.rds produced by the converter")
    p.add_argument('--methods', nargs='+', required=True,
                   help="Seurat integration strategy names (e.g. rpca cca)")
    p.add_argument('--batch_key',      required=True)
    p.add_argument('--state_key',      required=True)   # kept for interface parity
    p.add_argument('--latent_dim',     type=int, default=50)  # unused but accepted
    p.add_argument('--device',         default='auto', choices=['auto','cpu','cuda'])
    p.add_argument('--mem',            default='8G')
    p.add_argument('--scratch',        default='50G')
    p.add_argument('--runtime',        default='01:00:00')
    p.add_argument('--conda_env',      default='cellpath')
    p.add_argument('--output_dir',     default='./generated_scripts')
    args = p.parse_args()

    out_root = Path(args.output_dir) / f"benchmark_{args.proj_name}"
    out_root.mkdir(parents=True, exist_ok=True)

    for method in args.methods:
        base = f"benchmark_{args.proj_name}_{method}"
        # --- R script ---
        r_txt = R_TEMPLATE.format(
            proj_name     = args.proj_name,
            eset_dir      = args.eset_dir,
            batch_key     = args.batch_key,
            method        = method,
            latent_dim   = args.latent_dim,
        )
        r_path = out_root / f"{base}.R"
        r_path.write_text(r_txt)

        # --- shell wrapper ---
        sh_txt = SH_TEMPLATE.format(
            mem        = args.mem,
            scratch    = args.scratch,
            runtime    = args.runtime,
            conda_env  = args.conda_env,
            script_name= base
        )
        sh_path = out_root / f"{base}.sh"
        sh_path.write_text(sh_txt)

        rel_r  = r_path.relative_to(Path(args.output_dir))
        rel_sh = sh_path.relative_to(Path(args.output_dir))
        print(f"✅ Generated: {rel_r}\n✅ Generated: {rel_sh}\n")

if __name__ == "__main__":
    main()

