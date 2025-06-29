


#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(optparse)
  library(Matrix)
  library(Seurat)          # for Seurat output
  # library(SingleCellExperiment)   # uncomment if you want SCE instead
})

## ---------- CLI ----------
opt <- optparse::parse_args(
  optparse::OptionParser(
    option_list = list(
      optparse::make_option("--h5ad",        type="character"),
      optparse::make_option("--out_prefix",  type="character",
                            help="prefix for .mtx/.tsv intermediates"),
      optparse::make_option("--rds_out",     type="character",
                            help="final .rds path"),
      optparse::make_option("--python_env",  type="character",
                            default = "concord",
                            help="conda env with anndata")
    )
  )
)

if (is.null(opt$h5ad) || is.null(opt$out_prefix) || is.null(opt$rds_out))
  stop("Must supply --h5ad, --out_prefix, and --rds_out")

## ---------- 1. call python to dump mtx + metadata ----------
dump_py <- sprintf("
import anndata, scipy.io, pandas as pd, numpy as np, sys, os, scipy.sparse
ad = anndata.read_h5ad('%s')
X = ad.layers.get('counts', ad.X)
if not scipy.sparse.issparse(X):
    X = scipy.sparse.csr_matrix(X)
scipy.io.mmwrite('%s.mtx', X)
pd.DataFrame(ad.var).to_csv('%s_genes.tsv', sep='\\t', index=True)
pd.DataFrame(ad.obs).to_csv('%s_cells.tsv', sep='\\t', index=True)
",
  opt$h5ad, opt$out_prefix, opt$out_prefix, opt$out_prefix
)

cmd <- sprintf("conda activate %s && python - <<PY\n%s\nPY",
               opt$python_env, dump_py)
if (system(cmd) != 0) stop("Python export failed")

## ---------- 2. read intermediates into R ----------
mat  <- readMM(paste0(opt$out_prefix, ".mtx"))
genes <- read.delim(paste0(opt$out_prefix, "_genes.tsv"), row.names = 1,
                    check.names = FALSE)
cells <- read.delim(paste0(opt$out_prefix, "_cells.tsv"), row.names = 1,
                    check.names = FALSE)

rownames(mat) <- rownames(genes)
colnames(mat) <- rownames(cells)

## ---------- 3. build Seurat object ----------
seu <- CreateSeuratObject(counts = mat, meta.data = cells)
# If you want SCE instead:
# sce <- SingleCellExperiment(list(counts = mat), colData = cells, rowData = genes)

## ---------- 4. save ----------
saveRDS(seu, file = opt$rds_out)
cat("✅  Saved Seurat object to", opt$rds_out, "\n")
