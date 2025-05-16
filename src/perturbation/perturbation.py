import typing

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib import rcParams
import numpy as np
import scanpy as sc
import umap.umap_ as umap
import seaborn as sns
from matplotlib import cm
from matplotlib.lines import Line2D
import torch

from configparser import ConfigParser
from factory import get_factory, parse_list
from sc_dataset import get_loader

font_dir = ["Atkinson_Hyperlegible/Web Fonts/TTF/"]
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)

# Set font family globally
rcParams["font.family"] = "Atkinson Hyperlegible"
rcParams.update({"font.size": 13})
UMAP = umap.UMAP(random_state=60, n_neighbors=15)


def plot_UMAP(
    real: np.ndarray,
    fake: np.ndarray,
    real_labels: typing.Optional[typing.Union[list[str], np.ndarray]] = None,
    fit: bool = False,
    fake_title: str = "Fake",
    case_ctr: str = "ctr",
    save_path: typing.Optional[str] = None,
) -> None:
    """
    Plot UMAP projections and density plots for real and generated (fake) cell data.

    Parameters
    ----------
    real : np.ndarray
        Real cell expression data (cells x genes).
    fake : np.ndarray
        Fake/generated cell expression data (cells x genes).
    real_labels : Optional[Union[list[str], np.ndarray]], optional
        List or array of cell type labels for real cells (used for color-coded scatter plots), by default None
    fit : bool, optional
        Whether to fit a new UMAP model on the concatenated data (`True`),
        or transform using an existing fitted UMAP model (`False`), by default False
    fake_title : str, optional
        Title used for fake cells in the plots (e.g., "Generated", "Simulated"), by default "Fake"
    case_ctr : str, optional
        Identifier used in the saved filenames (before of after pert), by default "ctr"
    save_path : Optional[str], optional
        If provided, saves the scatter and density plots as PNGs, by default None
    """

    if real_labels is not None:
        real_labels = np.array(real_labels)
        celltypes = set(real_labels)
        n_classes = len(celltypes)

    if fit:
        embedded_cells = UMAP.fit_transform(
            np.concatenate((real, fake), axis=0),
        )
    else:
        embedded_cells = UMAP.transform(np.concatenate((real, fake), axis=0))

    real_embedding = embedded_cells[0 : real.shape[0], :]
    fake_embedding = embedded_cells[real.shape[0] :, :]

    # Figure 1: Scatter Plots
    plt.clf()
    fig1 = plt.figure(figsize=(20, 6))

    ax1 = fig1.add_subplot(1, 3, 1)
    if real_labels is not None:
        # Get the original tab20 colormap
        colormap = cm.get_cmap("tab20")

        # Get the colors from tab20 excluding the red color
        colors = [
            colormap(i) for i in range(colormap.N) if i != 6
        ]  # Remove red color at index 2

        # Create a new colormap without the red color
        colormap = cm.colors.ListedColormap(colors)

        colormap = [colormap(i) for i in np.linspace(0, 1, n_classes)]

        colors = {celltype: colormap[i] for i, celltype in enumerate(set(real_labels))}

        for i in set(real_labels):
            mask = real_labels[:] == i
            ax1.scatter(
                real_embedding[mask, 0],
                real_embedding[mask, 1],
                c=np.array([colors[i]]),
                label=str(i) + " (real)",
                alpha=0.7,
            )

    else:
        ax1.scatter(
            real_embedding[:, 0],
            real_embedding[:, 1],
            c="blue",
            label="real",
            alpha=0.7,
        )

    ax1.set_title("Real Cells")
    ax1.grid(True)
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")

    # Subplot 2: Fake Cells Scatter Plot
    ax2 = fig1.add_subplot(1, 3, 2)
    ax2.scatter(
        fake_embedding[:, 0],
        fake_embedding[:, 1],
        c="red",
        label="fake",
        alpha=0.7,
    )

    ax2.set_title(fake_title)
    ax2.grid(True)
    ax2.set_xlabel("UMAP 1")
    ax2.set_ylabel("UMAP 2")

    # Subplot 3: Real and Fake Cells Combined Scatter Plot
    ax3 = fig1.add_subplot(1, 3, 3)
    if real_labels is not None:
        for i in set(real_labels):
            mask = real_labels[:] == i
            ax3.scatter(
                real_embedding[mask, 0],
                real_embedding[mask, 1],
                c=np.array([colors[i]]),
                label=str(i) + " (real)",
                alpha=0.7,
            )
    else:
        ax3.scatter(
            real_embedding[:, 0],
            real_embedding[:, 1],
            c="blue",
            label="real",
            alpha=0.7,
        )

    ax3.scatter(
        fake_embedding[:, 0],
        fake_embedding[:, 1],
        c="red",
        label="fake",
        alpha=0.7,
    )

    ax3.set_xlabel("UMAP 1")
    ax3.set_ylabel("UMAP 2")

    ax3.set_title(f"Real and {fake_title} Cells")
    ax3.grid(True)

    # Get handles and labels from ax1
    handles, labels = ax1.get_legend_handles_labels()

    if real_labels is not None:
        # Place shared legend to the far left
        fig1.legend(
            handles,
            labels,
            title="Cell Types",
            loc="center left",
            bbox_to_anchor=(-0.17, 0.5),
            borderaxespad=0.0,
        )
        plt.tight_layout(rect=[0.01, 0, 1, 1])  # Leave space for legend on the left

    # Figure 2: Density Plots
    fig2 = plt.figure(figsize=(20, 6))

    # Subplot 1: Real Cells Density Plot
    ax4 = fig2.add_subplot(1, 3, 1)
    sns.kdeplot(
        real_embedding[:, 0],
        real_embedding[:, 1],
        cmap="Blues",
        shade=True,
        shade_lowest=False,
        ax=ax4,
        cbar=True,
    )
    ax4.set_title("Real Cells Density")
    ax4.grid(True)

    # Subplot 2: Fake Cells Density Plot
    ax5 = fig2.add_subplot(1, 3, 2)
    sns.kdeplot(
        fake_embedding[:, 0],
        fake_embedding[:, 1],
        cmap="Reds",
        shade=True,
        shade_lowest=False,
        ax=ax5,
        cbar=True,
        cbar_kws={"format": "%.3g"},  # Set format to limit significant figures
    )
    ax5.set_title(f"{fake_title} Cells Density")
    ax5.grid(True)

    # Subplot 3: Real and Fake Cells Combined Density Plot
    ax6 = fig2.add_subplot(1, 3, 3)
    sns.kdeplot(
        np.hstack((real_embedding[:, 0], fake_embedding[:, 0])),
        np.hstack((real_embedding[:, 1], fake_embedding[:, 1])),
        cmap="Greys",
        shade=True,
        shade_lowest=False,
        ax=ax6,
        cbar=True,
        cbar_kws={"format": "%.3g"},  # Set format to limit significant figures
    )
    ax6.set_title(f"Real and {fake_title} Cells Density")
    ax6.grid(True)

    plt.tight_layout()

    if save_path is not None:
        fig1.savefig(
            save_path + f"scatter_plot_{case_ctr}.png", dpi=300, bbox_inches="tight"
        )

        fig2.savefig(
            save_path + f"density_plot_{case_ctr}.png", dpi=300, bbox_inches="tight"
        )


def perturb(cfg: ConfigParser) -> None:
    """
    Performs perturbation experiment defined in the configuration file.
    Saves cells and UMAP plots before and after perturbation.

    Parameters
    ----------
    cfg : ConfigParser
        Parser for config file containing program params.
    """
    # use the same number of cels as the test set
    cells_no = cfg.getint("Preprocessing", "test set size")

    test_set = sc.read_h5ad(cfg.get("Data", "test"))

    # Read the GAN
    gan = get_factory(cfg).get_gan()
    print("Loaded GAN")
    checkpoint = cfg.get("EXPERIMENT", "checkpoint")
    print("Using checkpoint at", checkpoint)

    # get real cells
    loader = get_loader(cfg.get("Data", "test"), cells_no)
    real_cells, real_labels = next(iter(loader))
    real_cells = real_cells.cpu().numpy()
    real_cells[:cells_no], real_labels[:cells_no]

    # get fake cells without perturbation
    fake_cells = gan.generate_cells(cells_no, checkpoint)

    #### LET USER DEFINE PATH IN CFG
    if "celltype" in test_set.obs:
        plot_UMAP(
            test_set.X,
            fake_cells,
            real_labels=test_set.obs.celltype.to_numpy(),
            fit=True,
            case_ctr="before_perturbation",
            save_path=cfg.get("Perturbation", "save dir"),
        )
    else:
        plot_UMAP(
            test_set.X,
            fake_cells,
            fit=True,
            case_ctr="before_perturbation",
            save_path=cfg.get("Perturbation", "save dir"),
        )

    gan.gen.tf_expressions = None
    gan.gen.pert_mode = True
    fake_cells = gan.generate_cells(cells_no, checkpoint)
    fake_cells_new = gan.generate_cells(cells_no, checkpoint)
    assert (
        fake_cells == fake_cells_new
    ).all(), "perturbation mode should be deterministic"

    tfs_to_perturb = parse_list(cfg.get("Perturbation", "tfs to perturb"), str)
    pert_values = parse_list(cfg.get("Perturbation", "perturbation values"), float)

    gene_names = list(test_set.var_names)
    tfs_idx = [gene_names.index(tf) for tf in tfs_to_perturb]
    tf_idx = [gan.gen.tfs.index(tf_idx) for tf_idx in tfs_idx]
    print(tf_idx)

    unperturbed_tfs = gan.gen.tf_expressions.clone()
    pert_tensor = torch.tensor(
        pert_values,
        device=gan.gen.tf_expressions.device,
        dtype=gan.gen.tf_expressions.dtype,
    )
    gan.gen.tf_expressions[:, tf_idx] = pert_tensor.unsqueeze(0)
    fake_cells_perturbed = gan.generate_cells(cells_no, checkpoint)
    gan.gen.tf_expressions = unperturbed_tfs

    if "celltype" in test_set.obs:
        plot_UMAP(
            test_set.X,
            fake_cells_perturbed,
            real_labels=test_set.obs.celltype.to_numpy(),
            fit=False,
            case_ctr="after_perturbation",
            save_path=cfg.get("Perturbation", "save dir"),
        )
    else:
        plot_UMAP(
            test_set.X,
            fake_cells_perturbed,
            fit=False,
            case_ctr="after_perturbation",
            save_path=cfg.get("Perturbation", "save dir"),
        )

    fake_cells = sc.AnnData(fake_cells)
    fake_cells.obs_names = np.repeat("ctr_fake", fake_cells.shape[0])
    fake_cells.obs_names_make_unique()
    fake_cells.write(cfg.get("Perturbation", "save dir") + "before_perturbation.h5ad")

    fake_cells_perturbed = sc.AnnData(fake_cells_perturbed)
    fake_cells_perturbed.obs_names = np.repeat(
        "pert_fake", fake_cells_perturbed.shape[0]
    )
    fake_cells_perturbed.obs_names_make_unique()
    fake_cells_perturbed.write(
        cfg.get("Perturbation", "save dir") + "after_perturbation.h5ad"
    )
    print(
        "Saved cells before and after perturbation to",
        cfg.get("Perturbation", "save dir"),
    )
