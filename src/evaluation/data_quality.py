import typing

import tensorflow as tf
import numpy as np
import scanpy as sc
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as font_manager
from matplotlib import rcParams
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd

from configparser import ConfigParser
import evaluation.MMD as MMD
from evaluation.lisi import compute_lisi


font_dir = ["Atkinson_Hyperlegible/Web Fonts/TTF/"]
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)

# Set font family globally
rcParams["font.family"] = "Atkinson Hyperlegible"
rcParams.update({"font.size": 13})


def read_datasets(cfg: ConfigParser) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Load real and simulated (fake) gene expression datasets.

    Parameters
    ----------
    cfg : ConfigParser
        Parser for config file containing program params.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing:
        - real_cells.X : NumPy array of real gene expression data
        - fake_cells.X : NumPy array of simulated gene expression data, truncated to match real data row count
    """
    test_cells_path = cfg.get("Data", "test")

    fake_cells_path = cfg.get("Evaluation", "simulated data path", fallback="")
    if not fake_cells_path:  # Fall back to generation path
        fake_cells_path = cfg.get("Generation", "generation path", fallback="")
    if (
        not fake_cells_path
    ):  # Fall back on default save dir if generation path is also empty
        fake_cells_path = cfg.get("EXPERIMENT", "output directory") + "simulated.h5ad"

    real_cells = sc.read_h5ad(test_cells_path)
    fake_cells = sc.read_h5ad(fake_cells_path)[: real_cells.shape[0]]

    return real_cells.X, fake_cells.X


def plot_tSNE(
    real: np.ndarray, fake: np.ndarray, output_dir: str
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Perform t-SNE embedding on real and fake cell data and save a scatter plot.

    Parameters
    ----------
    real : np.ndarray
        A NumPy array of real cell data with shape (n_real_cells, n_features).
    fake : np.ndarray
        A NumPy array of fake/generated cell data with shape (n_fake_cells, n_features).
    output_dir : str
        Path to the directory where the t-SNE plot image will be saved.
        If empty, the plot is not saved.

    Returns
    -------
    typing.Tuple[np.ndarray, np.ndarray]
        A tuple containing the 2D t-SNE embeddings of the real and fake data,
        in the form (real_embedding, fake_embedding).
    """
    matplotlib.rcParams.update({"font.size": 15})

    embedded_cells = TSNE().fit_transform(np.concatenate((real, fake), axis=0))

    real_embedding = embedded_cells[0 : real.shape[0], :]
    fake_embedding = embedded_cells[fake.shape[0] :, :]

    plt.clf()
    plt.figure(figsize=(8, 8))

    plt.scatter(
        real_embedding[:, 0],
        real_embedding[:, 1],
        c="blue",
        label="real",
        alpha=0.5,
    )

    plt.scatter(
        fake_embedding[:, 0],
        fake_embedding[:, 1],
        c="red",
        label="fake",
        alpha=0.5,
    )

    plt.grid(True)
    plt.legend(
        loc="lower left", numpoints=1, ncol=2, fontsize=15, bbox_to_anchor=(0, 0)
    )
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    if output_dir:
        plt.savefig(output_dir + "tSNE.png", format="png", dpi=300, bbox_inches="tight")
        print("t-SNE plot saved to", output_dir + "tSNE.png")

    return real_embedding, fake_embedding


from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from scipy.spatial import distance
import numpy as np


def compute_distances(
    real_cells: np.ndarray, fake_cells: np.ndarray, axis: int = 0
) -> typing.Tuple[float, float]:
    """
    Compute Euclidean and Cosine distances between the mean expression profiles of real and fake cells.


    Parameters
    ----------
    real_cells : np.ndarray
        A NumPy array representing real cell data (cells x features).
    fake_cells : np.ndarray
        A NumPy array representing fake cell data (cells × features).
    axis : int, optional
        Axis along which to compute the mean expression (default is 0, meaning across cells), by default 0

    Returns
    -------
    typing.Tuple[float, float]
        A tuple containing:
        - Euclidean distance between the mean expression profiles.
        - Cosine distance between the mean expression profiles.
    """

    # calculate mean expression across cells
    fake_mean_expression = np.mean(fake_cells, axis=axis).reshape(1, -1)
    real_mean_expression = np.mean(real_cells, axis=axis).reshape(1, -1)
    return (
        euclidean_distances(fake_mean_expression, real_mean_expression).item(),
        cosine_distances(fake_mean_expression, real_mean_expression).item(),
    )


def compute_RF_AUROC(
    real_cells: np.ndarray,
    fake_cells: np.ndarray,
    output_dir: str,
    n_components: int = 50,
) -> float:
    """
    Compute the AUROC and plot the ROC curve of a Random Forest classifier distinguishing real from fake cells.

    Parameters
    ----------
    real_cells : np.ndarray
        A NumPy array representing real cell data (cells x features).
    fake_cells : np.ndarray
        A NumPy array representing fake/generated cell data (cells x features).
    output_dir : str
        Path to the directory where the ROC curve plot will be saved. The plot is saved as 'RF.png'.
    n_components : int, optional
        Number of principal components to retain during PCA, by default 50

    Returns
    -------
    float
        The area under the ROC curve (AUROC) for the classifier.
    """
    # create labels for real and fake data (real = 1, fake = 0)
    y_real = np.ones(real_cells.shape[0])
    y_fake = np.zeros(fake_cells.shape[0])

    # perform PCA
    pca = PCA(n_components=n_components)

    # split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        np.concatenate((real_cells, fake_cells), axis=0),
        np.hstack((y_real, y_fake)),
        test_size=0.3,
        shuffle=True,
    )

    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # train and test RF
    rf = RandomForestClassifier(n_estimators=1000)
    rf.fit(X_train, y_train)
    preds = rf.predict_proba(X_test)
    fpr, tpr, _ = metrics.roc_curve(y_test, preds[:, 1])
    roc_auc = metrics.roc_auc_score(y_test, preds[:, 1])

    # plot ROC
    plt.figure(figsize=(5, 5))
    plt.title("Receiver Operating Characteristic")
    plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.savefig(output_dir + "RF.png", format="png", bbox_inches="tight")
    print("RF ROC plot saved to", output_dir + "RF.png")

    return roc_auc


def evaluate(cfg: ConfigParser) -> None:
    """
    Assess the data quality of the simulated dataset.

    - T-SNE plots of real vs simulated cells (jointly embedded and plotted).
    - Euclidean and Cosine distances between the centroids of real cells and simulated cells.
        The centroid cell was obtained by calculating the mean along the gene axis (across all simulated or real cells).
    - Area under the receiver operating characteristic curve of a random forest in distinguishing real cells from simulated ones.
        We perform dimensionality reduction using PCA to extract the top 50 PCs of each cell as the input features to the RF model.
        The RF model is composed of 1000 trees and the Gini impurity was used to measure the quality of a split.
    - Maximum mean discrepancy (MMD) to estimate the proximity of high-dimensional distributions of real and simulated cells without
      creating centroids.
    - Mean integration local inverse Simpson's Index (miLISI).
        iLISI captures the effective number of datatypes (real or simulated) to which datapoints of its local neighborhood belong.

    As a “control” (and to enable calibration of these scores), we also calculated these metrics using two halves of the reference test set.

    Parameters
    ----------
    cfg : ConfigParser
        Parser for config file containing program params.
    """
    real_cells, fake_cells = read_datasets(cfg)

    # Split the test set into 2 to compute the control metrics
    num_rows = real_cells.shape[0]
    half = num_rows // 2
    real_cells_ctr1 = real_cells[:half, :]
    real_cells_ctr2 = real_cells[half:, :]

    if cfg.getboolean("Evaluation", "plot tsne"):
        tsne_real, tsne_generated = plot_tSNE(
            real_cells, fake_cells, cfg.get("EXPERIMENT", "output directory")
        )

    if cfg.getboolean("Evaluation", "compute euclidean distance") or cfg.getboolean(
        "Evaluation", "compute cosine distance"
    ):
        euclidean, cosine = compute_distances(real_cells, fake_cells)
        euclidean_ctr, cosine_ctr = compute_distances(real_cells_ctr1, real_cells_ctr2)

    if cfg.getboolean("Evaluation", "compute euclidean distance"):
        print()
        print("Euclidean distance (real vs fake):", euclidean)
        print("Euclidean distance (control):", euclidean_ctr)

    if cfg.getboolean("Evaluation", "compute cosine distance"):
        print()
        print("Cosine distance (real vs fake):", cosine)
        print("Cosine distance (control):", cosine_ctr)

    if cfg.getboolean("Evaluation", "compute rf auroc"):
        print()
        rf_auroc = compute_RF_AUROC(
            real_cells, fake_cells, cfg.get("EXPERIMENT", "output directory")
        )
        print("RF AUROC:", rf_auroc)

    if cfg.getboolean("Evaluation", "compute MMD"):
        print()
        with tf.device("cpu:0"):
            print(
                "MMD (real vs fake):",
                MMD.MMD(real_cells).compute(real_cells, fake_cells),
            )
            print(
                "MMD (control):",
                MMD.MMD(real_cells).compute(real_cells_ctr1, real_cells_ctr2),
            )

    if cfg.getboolean("Evaluation", "compute miLISI"):
        print()
        tsne_real_ctr1, tsne_real_ctr2 = plot_tSNE(real_cells_ctr1, real_cells_ctr2, "")

        tsne_coords = np.vstack((tsne_real, tsne_generated))
        tsne_coords_ctr = np.vstack((tsne_real_ctr1, tsne_real_ctr2))

        metadata = pd.DataFrame(
            ["real"] * real_cells.shape[0] + ["generated"] * fake_cells.shape[0],
            columns=["type"],
        )
        metadata_ctr = pd.DataFrame(
            ["ctr1"] * tsne_real_ctr1.shape[0] + ["ctr2"] * tsne_real_ctr2.shape[0],
            columns=["type"],
        )

        lisis = compute_lisi(tsne_coords, metadata, ["type"])
        lisis_ctr = compute_lisi(tsne_coords_ctr, metadata_ctr, ["type"])
        print("miLISI (real vs fake):", np.mean(lisis))
        print("miLISI (control):", np.mean(lisis_ctr))
