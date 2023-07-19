import os
import typing
from abc import ABC

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from sklearn.manifold import TSNE
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from gans.gan import GAN


class ConditionalGAN(GAN, ABC):
    @staticmethod
    def _sample_pseudo_labels(
        batch_size: int, cluster_ratios: torch.Tensor
    ) -> torch.Tensor:
        """
        Randomly samples cluster labels following a multinomial distribution.

        Parameters
        ----------
        batch_size : int
            The number of samples to generate (normally equal to training batch size).
        cluster_ratios : torch.Tensor
            Tensor containing the parameters of the multinomial distribution
            (ex: torch.Tensor([0.5, 0.3, 0.2]) for 3 clusters with occurence
            probabilities of  0.5, 0.3, and 0.2 for clusters 0, 1, and 2, respectively).

        Returns
        -------
        torch.Tensor
            Tensor containing a batch of samples cluster labels.
        """
        cluster_ratios = 1 - cluster_ratios
        mn_logits = torch.tile(-torch.log(cluster_ratios), (batch_size, 1))
        labels = torch.multinomial(mn_logits, 1)

        return labels.flatten()

    def _generate_tsne_plot(
        self,
        valid_loader: DataLoader,
        output_dir: typing.Union[str, bytes, os.PathLike],
    ) -> None:
        """
        Generate t-SNE plot during training.

        Parameters
        ----------
        valid_loader : DataLoader
            Validation set DataLoader.
        output_dir : typing.Union[str, bytes, os.PathLike]
            Directory to save the t-SNE plots.
        """
        tsne_path = output_dir + "/TSNE"
        if not os.path.isdir(tsne_path):
            os.makedirs(tsne_path)

        fake_cells, fake_labels = self.generate_cells(len(valid_loader.dataset))
        valid_cells, valid_labels = next(iter(valid_loader))
        valid_labels = valid_labels.flatten()

        embedded_cells = TSNE().fit_transform(
            np.concatenate((valid_cells, fake_cells), axis=0)
        )

        real_embedding = embedded_cells[0 : valid_cells.shape[0], :]
        fake_embedding = embedded_cells[valid_cells.shape[0] :, :]

        colormap = cm.nipy_spectral
        colors = [colormap(i) for i in np.linspace(0, 1, self.num_classes)]

        plt.clf()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(32, 12))

        for i in range(self.num_classes):
            mask = valid_labels[:] == i

            ax1.scatter(
                real_embedding[mask, 0],
                real_embedding[mask, 1],
                c=colors[i],
                marker="o",
                label="real_" + str(i),
            )

        ax1.legend(
            loc="lower left", numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0)
        )

        for i in range(self.num_classes):
            mask = fake_labels[:] == i
            ax2.scatter(
                fake_embedding[mask, 0],
                fake_embedding[mask, 1],
                c=colors[i],
                marker="o",
                label="fake_" + str(i),
            )

        ax2.legend(
            loc="lower left", numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0)
        )

        plt.savefig(tsne_path + "/step_" + str(self.step) + ".jpg")

        with SummaryWriter(f"{output_dir}/TensorBoard/TSNE") as w:
            w.add_figure("t-SNE plot", fig, self.step)

        plt.close()
