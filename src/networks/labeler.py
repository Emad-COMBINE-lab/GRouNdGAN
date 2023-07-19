import typing

import torch
from torch import nn


class Labeler(nn.Module):
    def __init__(
        self, num_genes: int, num_tfs: int, labeler_layers: typing.List[int]
    ) -> None:
        """
        Labeler network's constructor.

        Parameters
        ----------
        num_genes : int
            Number of target genes (all genes excluding TFs) in the dataset.
        num_tfs : int
            Number of transcription factors in the dataset.
        labeler_layers : typing.List[int]
            List of integers corresponding to the number of neurons
            at each deep layer of the labeler.
        """
        super(Labeler, self).__init__()

        self.num_genes = num_genes
        self.num_tfs = num_tfs
        self.labeler_layers = labeler_layers

        self._create_labeler()

    def forward(self, target_genes: torch.Tensor) -> torch.Tensor:
        """
        Function for completing a forward pass of the labeler.
        This network performs a regression by predicting TF expression
        from target gene expression.

        Parameters
        ----------
        target_genes : torch.Tensor
            Tensor containing target gene expression of (fake/real) cells.

        Returns
        -------
        torch.Tensor
            Tensor containing regulatory TFs.
        """
        return self._labeler(target_genes)

    def _create_labeler(self) -> None:
        """Method for creating a labeler network."""
        layers = []
        input_dim = self.num_genes
        for output_dim in self.labeler_layers:
            layers.append(
                nn.Sequential(
                    nn.Linear(input_dim, output_dim),
                    nn.BatchNorm1d(output_dim),
                    nn.ReLU(inplace=True),
                )
            )
            input_dim = output_dim
        layers.append(nn.Sequential(nn.Linear(input_dim, self.num_tfs)))

        self._labeler = nn.Sequential(*layers)
