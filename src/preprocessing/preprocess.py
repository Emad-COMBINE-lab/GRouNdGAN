from collections import Counter
from configparser import ConfigParser

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse


def preprocess(cfg: ConfigParser) -> None:
    """
    Apply preprocessing steps.

    Parameters
    ----------
    cfg : ConfigParser
        Parser for config file containing preprocessing params.
    """

    if cfg.get("Preprocessing", "10x") == "True":
        anndata = sc.read_10x_mtx(
            cfg.get("Preprocessing", "raw"), make_unique=True, gex_only=True
        )

    else:
        anndata = sc.read_h5ad(cfg.get("Preprocessing", "raw"))

    original_order = np.arange(anndata.n_obs)  # Store the original cell order
    np.random.shuffle(original_order)  # Shuffle the indices

    # Apply the shuffled order to the AnnData object
    anndata = anndata[original_order]

    # clustering
    ann_clustered = anndata.copy()
    sc.pp.recipe_zheng17(ann_clustered)
    sc.tl.pca(ann_clustered, n_comps=50)
    sc.pp.neighbors(ann_clustered, n_pcs=50)
    sc.tl.louvain(
        ann_clustered, resolution=float(cfg.get("Preprocessing", "louvain res"))
    )
    anndata.obs["cluster"] = ann_clustered.obs["louvain"]

    # get cluster ratios
    cells_per_cluster = Counter(anndata.obs["cluster"])
    cluster_ratios = dict()
    for key, value in cells_per_cluster.items():
        cluster_ratios[key] = value / anndata.shape[0]
    anndata.uns["cluster_ratios"] = cluster_ratios
    anndata.uns["clusters_no"] = len(cluster_ratios)

    # filtering
    sc.pp.filter_cells(anndata, min_genes=int(cfg.get("Preprocessing", "min genes")))
    sc.pp.filter_genes(anndata, min_cells=int(cfg.get("Preprocessing", "min cells")))
    anndata.uns["cells_no"] = anndata.shape[0]
    anndata.uns["genes_no"] = anndata.shape[1]

    canndata = anndata.copy()

    # library-size normalization
    sc.pp.normalize_per_cell(
        canndata, counts_per_cell_after=int(cfg.get("Preprocessing", "library size"))
    )

    if cfg.get("Preprocessing", "annotations") is not None:
        annotations = pd.read_csv(
            cfg.get("Preprocessing", "annotations"), delimiter="\t"
        )
        annotation_dict = {
            item["barcodes"]: item["celltype"]
            for item in annotations.to_dict("records")
        }
        anndata.obs["barcodes"] = anndata.obs.index
        anndata.obs["celltype"] = anndata.obs["barcodes"].map(annotation_dict)

    # identify highly variable genes
    sc.pp.log1p(canndata)  # logarithmize the data
    sc.pp.highly_variable_genes(
        canndata, n_top_genes=int(cfg.get("Preprocessing", "highly variable number"))
    )

    if issparse(canndata.X):
        canndata.X = np.exp(canndata.X.toarray()) - 1  # get back original data
    else:
        canndata.X = np.exp(canndata.X) - 1  # get back original data

    anndata = anndata[
        :, canndata.var["highly_variable"]
    ]  # only keep highly variable genes

    sc.pp.normalize_per_cell(
        anndata, counts_per_cell_after=int(cfg.get("Preprocessing", "library size"))
    )

    # sort genes by name (not needed)
    sorted_genes = np.sort(anndata.var_names)
    anndata = anndata[:, sorted_genes]

    val_size = int(cfg.get("Preprocessing", "validation set size"))
    test_size = int(cfg.get("Preprocessing", "test set size"))

    anndata[:val_size].write_h5ad(cfg.get("Data", "validation"))
    anndata[val_size : test_size + val_size].write_h5ad(cfg.get("Data", "test"))
    anndata[test_size + val_size :].write_h5ad(cfg.get("Data", "train"))

    print("Successfully preprocessed and saved dataset.")
    print("Train set:", cfg.get("Data", "train"))
    print("Validation set:", cfg.get("Data", "validation"))
    print("Test set:", cfg.get("Data", "test"))
