#!/usr/bin/env python3
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import scanpy as sc

example = """example:

python src/preprocessing/preprocess.py data/raw/PBMC/ data/processed/PBMC/PBMC68k.h5ad --annotations data/raw/PBMC/barcodes_annotations.tsv
 
 """

parser = argparse.ArgumentParser(
    "preprocess raw 10x gene expression data", epilog=example
)
parser.add_argument(
    "raw",
    metavar="r",
    help="directory containing observations (barcodes.tsv), features(genes.tsv), and the expression matrix (matrix.tsv)",
)
parser.add_argument(
    "output",
    metavar="o",
    help="path to store the preprocessed data in .h5ad format",
)
parser.add_argument(
    "--annotations",
    metavar="a",
    help="path to annotations (barcodes, celltype .tsv)",
    required=False,
    default=None,
)
parser.add_argument(
    "--min_cells",
    required=False,
    default=3,
    help="genes expressed in less than min_cells cells are discarded",
    type=int,
)
parser.add_argument(
    "--min_genes",
    required=False,
    default=10,
    help="cells with less than min_genes expressed are discarded",
    type=int,
)
parser.add_argument("--library_size", required=False, default=20000, type=int)
parser.add_argument("--louvain_res", required=False, default=0.15, type=float)
parser.add_argument(
    "--highly_var_no",
    required=False,
    default=1000,
    help="number of highly variable genes to identify",
    type=int,
)

args = parser.parse_args()

anndata = sc.read_10x_mtx(args.raw, make_unique=True, gex_only=True)

# clustering
ann_clustered = anndata.copy()
sc.pp.recipe_zheng17(ann_clustered)
sc.tl.pca(ann_clustered, n_comps=50)
sc.pp.neighbors(ann_clustered, n_pcs=50)
sc.tl.louvain(ann_clustered, resolution=args.louvain_res)
anndata.obs["cluster"] = ann_clustered.obs["louvain"]

# get cluster ratios
cells_per_cluster = Counter(anndata.obs["cluster"])
cluster_ratios = dict()
for key, value in cells_per_cluster.items():
    cluster_ratios[key] = value / anndata.shape[0]
anndata.uns["cluster_ratios"] = cluster_ratios
anndata.uns["clusters_no"] = len(cluster_ratios)

# filtering
sc.pp.filter_cells(anndata, min_genes=args.min_genes)
sc.pp.filter_genes(anndata, min_cells=args.min_cells)
anndata.uns["cells_no"] = anndata.shape[0]
anndata.uns["genes_no"] = anndata.shape[1]

# library-size normalization
sc.pp.normalize_per_cell(anndata, counts_per_cell_after=args.library_size)

if args.annotations is not None:
    annotations = pd.read_csv(args.annotations, delimiter="\t")
    annotation_dict = {
        item["barcodes"]: item["celltype"] for item in annotations.to_dict("records")
    }
    anndata.obs["barcodes"] = anndata.obs.index
    anndata.obs["celltype"] = anndata.obs["barcodes"].map(annotation_dict)


# identify highly variable genes
sc.pp.log1p(anndata)  # logarithmize the data
sc.pp.highly_variable_genes(anndata, n_top_genes=args.highly_var_no)
anndata.X = np.exp(anndata.X.toarray()) - 1  # get back original data


anndata.write_h5ad(args.output)
print("Successfully preprocessed and and saved dataset")
