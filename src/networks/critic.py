import typing

import torch
from torch import nn


class Critic(nn.Module):
    def __init__(self, x_input: int, critic_layers: typing.List[int]) -> None:
        """
        Non-conditional Critic's constructor.

        Parameters
        ----------
        x_input : int
            The dimension of the input tensor.
        critic_layers : typing.List[int]
            List of integers corresponding to the number of neurons
            at each hidden layer of the critic.
        """
        super(Critic, self).__init__()

        self.x_input = x_input
        self.critic_layers = critic_layers

        self._create_critic()

    def forward(self, data: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Function for completing a forward pass of the critic.

        Parameters
        ----------
        data : torch.Tensor
            Tensor containing gene expression of (fake/real) cells.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        torch.Tensor
            1-dimensional tensor representing fake/real cells.
        """
        return self._critic(data)

    def _create_critic(self) -> None:
        """Method for creating the Critic's network."""
        layers = []
        input_size = self.x_input
        for output_size in self.critic_layers:
            layers.append(self._create_critic_block(input_size, output_size))
            input_size = output_size  # update input size for the next layer

        # outermost layer
        layers.append(self._create_critic_block(input_size, 1, final_layer=True))
        self._critic = nn.Sequential(*layers)

    @staticmethod
    def _create_critic_block(
        input_dim: int, output_dim: int, final_layer: typing.Optional[bool] = False
    ) -> nn.Sequential:
        """
        Function for creating a sequence of operations corresponding to
        a Critic block; a linear layer, and ReLU (except in the final block).

        Parameters
        ----------
        input_dim : int
            The block's input dimensions.
        output_dim : int
            The block's output dimensions.
        final_layer : typing.Optional[bool], optional
            Indicates if the block contains the final layer, by default False.

        Returns
        -------
        nn.Sequential
            Sequential container containing the modules.
        """
        linear_layer = nn.Linear(input_dim, output_dim)
        torch.nn.init.zeros_(linear_layer.bias)
        if not final_layer:
            torch.nn.init.kaiming_normal_(
                linear_layer.weight, mode="fan_in", nonlinearity="relu"
            )
            return nn.Sequential(linear_layer, nn.ReLU(inplace=True))
        # don't use an activation function at the
        # outermost layer of the critic's network
        else:
            torch.nn.init.xavier_uniform_(linear_layer.weight)
            return nn.Sequential(linear_layer)


class ConditionalCritic(Critic):
    def __init__(
        self, x_input: int, critic_layers: typing.List[int], num_classes: int
    ) -> None:
        """
        Conditional Critic's constructor - Projection Discriminator (Miyato et al.,2018).

        Parameters
        ----------
        x_input : int
            The dimension of the input tensor.
        critic_layers : typing.List[int]
            List of integers corresponding to the number of neurons
            at each hidden layer of the critic.
        num_classes : int
            Number of clusters.
        """
        self.num_classes = num_classes

        super(ConditionalCritic, self).__init__(x_input, critic_layers)

    def forward(
        self, data: torch.Tensor, labels: torch.Tensor = None, *args, **kwargs
    ) -> torch.Tensor:
        """
        Function for completing a forward pass of the conditional critic.

        Parameters
        ----------
        data : torch.Tensor
            Tensor containing gene expression of (fake/real) cells.
        labels : torch.Tensor
            Tensor containing labels corresponding to cells (data parameter).
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        torch.Tensor
            1-dimensional tensor representing fake/real cells.
        """
        y = data
        for layer in self._critic[:-2]:
            y = layer(y)

        output = self._critic[-2](y)
        proj = self._critic[-1](labels)
        output += torch.sum(proj * y, dim=1, keepdim=True)
        return output

    def _create_critic(self) -> None:
        """Method for creating the Conditional Critic's network."""
        self._critic = nn.ModuleList()
        input_size = self.x_input
        for output_size in self.critic_layers:
            self._critic.append(self._create_critic_block(input_size, output_size))
            input_size = output_size  # update input size for the next layer

        # outermost layer
        self._critic.append(self._create_critic_block(input_size, 1, final_layer=True))

        # projection layer
        proj_layer = nn.Embedding(self.num_classes, input_size)
        nn.init.xavier_uniform_(proj_layer.weight)
        self._critic.append(proj_layer)


class ConditionalCriticProj(Critic):
    def __init__(
        self, x_input: int, critic_layers: typing.List[int], num_classes: int
    ) -> None:
        """
        Conditional Critic's constructor using a modified implementation of
        Projection Discriminator (Marouf et al, 2020).

        Parameters
        ----------
        x_input : int
            The dimension of the input tensor.
        critic_layers : typing.List[int]
            List of integers corresponding to the number of neurons
            at each hidden layer of the critic.
        num_classes : int
            Number of clusters.
        """
        self.num_classes = num_classes

        super(ConditionalCriticProj, self).__init__(x_input, critic_layers)

    def forward(
        self, data: torch.Tensor, labels: torch.Tensor = None, *args, **kwargs
    ) -> torch.Tensor:
        """
        Function for completing a forward pass of the conditional critic.

        Parameters
        ----------
        data : torch.Tensor
            Tensor containing gene expression of (fake/real) cells.
        labels : torch.Tensor
            Tensor containing labels corresponding to cells (data parameter).
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        torch.Tensor
            1-dimensional tensor representing fake/real cells.
        """
        y = data
        for layer in self._critic[:-2]:
            y = layer(y)

        output = self._critic[-2](labels)
        proj = self._critic[-1](labels)
        output += torch.sum(proj * y, dim=1, keepdim=True)
        return output

    def _create_critic(self) -> None:
        """Method for creating the Conditional Critic's network."""
        self._critic = nn.ModuleList()
        input_size = self.x_input
        for output_size in self.critic_layers:
            self._critic.append(self._create_critic_block(input_size, output_size))
            input_size = output_size  # update input size for the next layer

        # bias layer (replaces the output linear layer)
        proj_bias = nn.Embedding(self.num_classes, 1)
        torch.nn.init.zeros_(proj_bias.weight)
        self._critic.append(proj_bias)

        # projection layer
        proj_layer = nn.Embedding(self.num_classes, input_size)
        nn.init.xavier_uniform_(proj_layer.weight)
        self._critic.append(proj_layer)
