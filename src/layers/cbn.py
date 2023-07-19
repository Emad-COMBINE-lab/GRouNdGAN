import torch
from torch import nn


class ConditionalBatchNorm(nn.Module):
    def __init__(self, num_features: int, num_classes: int) -> None:
        """
        1D Conditional batch normalization (CBN) layer (Dumoulin et al.,
        2016; De Vries et al.,2017).

        Parameters
        ----------
        num_features : int
            Number of input features.
        num_classes : int
            Number of classes (i.e., distinct labels).
        """
        super().__init__()
        self.num_features = num_features

        # regular batch norm without learnable parameters
        self._batch_norm = nn.BatchNorm1d(num_features, affine=False)

        self._embed = nn.Embedding(num_classes, num_features * 2)
        # Initialise scale at N(1, 0.02)
        self._embed.weight.data[:, :num_features].normal_(1, 0.02)
        # Initialise shift at 0
        self._embed.weight.data[:, num_features:].zero_()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Perform CBN given a batch of labels.

        Parameters
        ----------
        x : torch.Tensor
            Tensor on which to perform CBN.
        y : torch.Tensor
            A batch of labels.

        Returns
        -------
        torch.Tensor
            Conditionally batch normalized input.
        """
        out = self._batch_norm(x)

        # separate weight and bias from the embedding
        scale, shift = self._embed(y).chunk(2, 1)

        # shift and scale activations based on labels y provided
        return scale * out + shift
