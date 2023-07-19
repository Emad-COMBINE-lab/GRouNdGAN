import typing

import torch
from torch import nn


class LSN(nn.Module):
    def __init__(
        self,
        library_size: int,
        device: typing.Optional[str] = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """
        Library size normalization (LSN) layer.

        Parameters
        ----------
        library_size : int
            Total number of counts per generated cell.
        device : typing.Optional[str], optional
            Specifies to train on 'cpu' or 'cuda'. Only 'cuda' is supported for training the
            GAN but 'cpu' can be used for inference, by default "cuda" if torch.cuda.is_available() else"cpu".

        """
        super().__init__()
        self.library_size = library_size
        self.device = device
        self.scale = None

    def forward(
        self, in_: torch.Tensor, reuse_scale: typing.Optional[bool] = False
    ) -> torch.Tensor:
        """
        Function for completing a forward pass of the LSN layer.

        Parameters
        ----------
        in_ : torch.Tensor
            Tensor containing gene expression of cells.
        reuse_scale : typing.Optional[bool], optional
            If set to true, the LSN layer will scale the cells by
            the same scale as the previous batch. Useful for performing
            perturbation studies. By default False

        Returns
        -------
        torch.Tensor
            Gene expression of cells after library size normalization.
        """
        gammas = torch.ones(in_.shape[0]).to(self.device) * self.library_size
        sigmas = torch.sum(in_, 1)
        scale = torch.div(gammas, sigmas)

        if reuse_scale:
            if self.scale is not None:
                scale = self.scale # use previously set scale if not first pass through the frozen LSN layer
            else:
                self.scale = scale # if first pass through the frozen LSN layer
        else:
            self.scale = None  # unfreeze LSN scale if set

        return torch.nan_to_num(
            torch.transpose(torch.transpose(in_, 0, 1) * scale, 0, 1), nan=0.0
        ) # possible NaN if all genes are zero-expressed - NaNs are thus replaced with zeros
