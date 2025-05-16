import pickle
import typing

import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

from configparser import ConfigParser


def get_imposed_grn(
    cfg: ConfigParser,
) -> typing.Tuple[
    pd.DataFrame,
    typing.Set[str],
    typing.Set[str],
    typing.List[str],
    typing.List[str],
    int,
]:
    """
    Load and process the imposed (ground-truth) GRN from a causal graph file.

    This function reads the GRouNdGAN GRN and maps numeric gene indices to gene names
    and constructs a structured representation of the GRN.

    Parameters
    ----------
    cfg : ConfigParser
        Configuration object providing the necessary file paths:
        - "Data" -> "train": Path to real cells `.h5ad` file for gene name mapping.
        - "Data" -> "causal graph": Path to the pickle file containing the ground truth GRN.
        - "GRN Benchmarking" -> "ground truth save path": Optional output path for saving the ground truth GRN in csv format.

    Returns
    -------
    imposed_grn : pd.DataFrame
        DataFrame with a single column "res" containing string-formatted edges ("TF -> target").

    imposed_TFs : set
        Set of transcription factors (TFs) involved in the imposed GRN.

    imposed_targets : set
        Set of target genes in the imposed GRN.

    imposed_edges : list of str
        List of edges as strings in the format "TF -> target".

    TFs : list of str
        List of unique transcription factors (from `imposed_TFs`), useful for downstream filtering.

    possible_edges_no : int
        Total number of possible TF-target combinations based on imposed GRN (|TFs| * |targets|).
    """
    real_cells = sc.read_h5ad(cfg.get("Data", "train"))
    gene_names = real_cells.var_names.tolist()

    imposed_TFs = set()
    imposed_targets = set()
    with open(cfg.get("Data", "causal graph"), "rb") as fp:
        imposed_grn = pickle.load(fp)

    # convert the GRN from GRouNdGAN format (based on gene number to gene name)
    imposed_edges = []
    to_save_edges = []
    for gene, tfs in imposed_grn.items():
        gene = gene_names[gene]
        imposed_targets.add(gene)
        for tf in tfs:
            tf = gene_names[tf]

            # create a third column in the df for the edge: (TF -> Gene) format
            imposed_edges.append(tf + " -> " + gene)
            to_save_edges.append({"TF (regulator)": tf, "Gene (target)": gene})
            imposed_TFs.add(tf)

    imposed_grn_save_path = cfg.get("GRN Benchmarking", "ground truth save path")
    if imposed_grn_save_path:
        save_path = imposed_grn_save_path + "ground truth GRN.csv"
        imposed_grn_df = pd.DataFrame(to_save_edges)
        imposed_grn_df.to_csv(save_path, index=False)
        print("Saved ground truth GRN to", save_path)

    imposed_grn = pd.DataFrame(imposed_edges, columns=["res"])
    TFs = list(imposed_TFs)
    possible_edges_no = len(imposed_TFs) * len(imposed_targets)

    return (
        imposed_grn,
        imposed_TFs,
        imposed_targets,
        imposed_edges,
        TFs,
        possible_edges_no,
    )


def get_fake_grn(
    cfg: ConfigParser,
    imposed_TFs: typing.Iterable[str],
    imposed_targets: typing.Iterable[str],
    TFs: typing.Iterable[str],
    possible_edges_no: int,
) -> pd.DataFrame:
    """
    Construct a benchmark gene regulatory network (GRN) DataFrame using inferred and imposed edges.

    This function loads an inferred GRN from a file, filters it based on imposed TFs
    and targets, and augments it with all possible unreported edges (assigned zero importance).
    This ensures that all theoretically possible edges are included for evaluation purposes (e.g., computing recall).

    Parameters
    ----------
    cfg : ConfigParser
        Configuration object providing the path to the inferred GRN file via the section
        "GRN Benchmarking" -> "grn to benchmark".

    imposed_TFs : Iterable[str]
        List of TFs that are considered "imposed" in the simulation and should be included as sources in the GRN.

    imposed_targets : Iterable[str]
        List of target genes that are expected to be regulated in the benchmark.

    TFs : Iterable[str]
        List of all possible TFs used to filter the GRN (must be valid source nodes).

    possible_edges_no : int
        Total number of possible edges in the benchmark GRN (used for validation/debugging).

    Returns
    -------
    pd.DataFrame
        DataFrame representing the benchmark GRN. Columns include:
        - "TF": Source transcription factor
        - "target": Target gene
        - "importance": Importance score (0 for non-inferred edges)
        - "res": String representation of the edge ("TF -> target")
    """
    # Read inferred GRN, assuming edges are already ordered by importance
    fake_grn = pd.read_csv(
        cfg.get("GRN Benchmarking", "grn to benchmark", fallback=""),
        sep="\t",
        # header=None,
    )
    fake_grn.columns = ["TF", "target", "importance"]
    fake_grn = fake_grn.reindex(
        fake_grn.importance.abs().sort_values(ascending=False).index
    )

    fake_grn["res"] = fake_grn["TF"] + " -> " + fake_grn["target"]

    unimportant_edges = {"TF": [], "target": [], "importance": [], "res": []}
    for tf in imposed_TFs:
        for gene in imposed_targets:
            unimportant_edges["TF"].append(tf)
            unimportant_edges["target"].append(gene)
            unimportant_edges["importance"].append(0)
            unimportant_edges["res"].append(tf + " -> " + gene)

    fake_grn = fake_grn[
        ~fake_grn["target"].isin(TFs)
    ]  # remove edge if a TF is also a target gene
    fake_grn = fake_grn[fake_grn["TF"].isin(TFs)]  # remove edge not coming from a TF
    fake_grn = fake_grn[fake_grn["target"].isin(imposed_targets)]
    fake_grn = fake_grn[fake_grn["TF"].isin(TFs)]

    # add all the edges that don't appear in the inference method as low importance edges
    # without this, recall might not go to 1
    fake_grn = (
        pd.concat([fake_grn, pd.DataFrame(unimportant_edges)])
        .drop_duplicates(subset=["res"])
        .reset_index(drop=True)
    )
    assert fake_grn.shape[0] == possible_edges_no
    return fake_grn


def compute_precision_at_k(
    fake_grn: pd.DataFrame,
    imposed_grn: pd.DataFrame,
    TFs: typing.Union[typing.List[str], typing.Set[str]],
    numTFs_pergene: int,
    save_path: str,
) -> None:
    """
    Compute precision at k for inferred GRN against the imposed ground truth.
    It plots the precision and baseline curves and optionally saves the figure.

    Parameters
    ----------
    fake_grn : pd.DataFrame
        Inferred GRN with columns "TF", "target", and "res" (formatted as "TF -> target").

    imposed_grn : pd.DataFrame
        Ground truth GRN with a column "res" representing edges in "TF -> target" format.

    TFs : Union[List[str], Set[str]]
        List or set of valid transcription factors. Used to exclude predictions where a TF is treated as a target.

    numTFs_pergene : int
        Maximum number of top TFs to evaluate per gene.
    save_path : str
        Directory path where the precision-at-k plot should be saved. If empty, the plot is not saved.

    Returns
    -------
    None
    """
    range_TFs_pergene = range(1, numTFs_pergene + 1)
    precisions = []
    baselines = []

    for n_toptfs in range_TFs_pergene:
        # the code below doesn't preserve order
        fake_grn_dict = dict(fake_grn.groupby("target")["TF"].apply(list))
        fake_grn_dict = {
            gene: set(tfs[:n_toptfs]) for (gene, tfs) in fake_grn_dict.items()
        }

        edges = []
        for gene, tfs in fake_grn_dict.items():
            if gene in TFs:
                continue
            for tf in tfs:
                edges.append(tf + " -> " + gene)

        fake_grn_dict = pd.DataFrame(edges, columns=["res"])
        common_edges = len(
            set(fake_grn_dict.iloc[:]["res"]) & set(imposed_grn.iloc[:]["res"])
        )

        consistency = common_edges / fake_grn_dict.shape[0]  # precision
        precisions.append(consistency)
        baselines.append(n_toptfs / len(TFs))

    print("Inferred GRN precision at k:", precisions)
    print("Baseline precision at k:", baselines)
    fig = plt.figure(figsize=(6, 6))
    plt.plot(
        range_TFs_pergene,
        precisions,
        label="Inferred GRN",
        marker="o",
        linewidth=3,
        linestyle="solid",
    )
    plt.plot(
        range_TFs_pergene,
        baselines,
        label="Baseline",
        marker="s",
        linewidth=3,
        linestyle="dotted",
    )
    plt.ylim([0, 1])
    plt.xlim(1, numTFs_pergene)
    plt.xticks(range_TFs_pergene)
    plt.xlabel("Number of top TFs per gene", fontsize=15)
    plt.ylabel("Precision", fontsize=15)
    plt.legend(fontsize=13, loc="upper right")
    plt.xlim([1 - 0.1, numTFs_pergene + 0.1])
    plt.grid()
    if save_path:
        plt.savefig(
            save_path + "Top_TF_precision.png",
            format="png",
            dpi=300,
            bbox_inches="tight",
        )
        print("Saved precision at k plot at", save_path + "Top_TF_precision.png")


def compute_PR(
    fake_grn: pd.DataFrame,
    imposed_grn: pd.DataFrame,
    imposed_edges: typing.List[str],
    imposed_targets: typing.Set[str],
    imposed_TFs: typing.Set[str],
    save_path: typing.Optional[str] = "",
) -> None:
    """
    Compute and plot the Precision-Recall (PR) curve for a predicted gene regulatory network (GRN)
    against a ground-truth GRN.

    Parameters
    ----------
    fake_grn : pd.DataFrame
        Inferred GRN DataFrame with a column 'res' representing edges in the format "TF -> target".

    imposed_grn : pd.DataFrame
        Ground-truth GRN DataFrame with a column 'res' representing edges in the same format.

    imposed_edges : List[str]
        List of all ground-truth edges as strings ("TF -> target").

    imposed_targets : Set[str]
        Set of ground-truth target genes.

    imposed_TFs : Set[str]
        Set of ground-truth transcription factors.

    save_path : Optional[str]
        Directory path to save the precision-recall plot as "PR_curve.png".
        If None or empty, the plot is not saved.

    Returns
    -------
    None
    """
    precisions = []
    recalls = []

    lb = np.arange(0.001, 1.001, 0.001)  # define the TOP % of edges to consider
    for _, ct in enumerate(lb):
        ct = int(fake_grn.shape[0] * ct)
        common_edges = len(
            set(fake_grn.iloc[:ct]["res"]) & set(imposed_grn.iloc[:]["res"])
        )
        recall = common_edges / imposed_grn.shape[0]
        recalls.append(recall)
        precision = common_edges / ct
        precisions.append(precision)

    fig, ax = plt.subplots(figsize=(6, 6))
    auc_score = np.around(auc(recalls, precisions), 2)
    plt.rcParams.update({"font.size": 15})

    ax.plot(
        recalls,
        precisions,
        label=f"Inferred GRN AUPRC: {auc_score}",
        linestyle="solid",
        linewidth=3,
    )
    baseline = len(imposed_edges) / (len(imposed_TFs) * len(imposed_targets))

    ax.set_xlabel("Recall", fontsize=15)
    ax.set_ylabel("Precision", fontsize=15)

    ax.axhline(y=baseline, color="dimgrey", linestyle="dashed")
    plt.text(-0.18, baseline, "random", weight="bold")

    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])

    plt.legend(fontsize=13, loc="upper right")
    plt.grid(True)
    print("Inferred GRN AURPC:", auc_score)
    print("Baseline AUPRC (random predictor):", baseline)
    if save_path:
        plt.savefig(
            save_path + "PR_curve.png", format="png", dpi=300, bbox_inches="tight"
        )
        print("Saved PR curve at", save_path + "PR_curve.png")


def evaluate(cfg: ConfigParser) -> None:
    """
    Evaluate the reconstructed GRN against the ground truth GRN

    Parameters
    ----------
    cfg : ConfigParser
        Parser for config file containing program params.

    Returns
    -------
    None
    """
    (
        imposed_grn,
        imposed_TFs,
        imposed_targets,
        imposed_edges,
        TFs,
        possible_edges_no,
    ) = get_imposed_grn(cfg)
    fake_grn = get_fake_grn(cfg, imposed_TFs, imposed_targets, TFs, possible_edges_no)

    if cfg.getboolean("GRN Benchmarking", "compute precision at k"):
        print()
        compute_precision_at_k(
            fake_grn,
            imposed_grn,
            TFs,
            cfg.getint("GRN Benchmarking", "k"),
            cfg.get("GRN Benchmarking", "plots save path"),
        )

    if cfg.getboolean("GRN Benchmarking", "compute pr"):
        print()
        compute_PR(
            fake_grn,
            imposed_grn,
            imposed_edges,
            imposed_targets,
            imposed_TFs,
            cfg.get("GRN Benchmarking", "plots save path"),
        )
