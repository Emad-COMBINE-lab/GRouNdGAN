import typing

import torch
from layers.cbn import ConditionalBatchNorm
from layers.lsn import LSN
from torch import nn
from torch.nn.modules.activation import ReLU


class Generator(nn.Module):
    def __init__(
        self,
        z_input: int,
        output_cells_dim: int,
        gen_layers: typing.List[int],
        library_size: typing.Optional[typing.Union[int, None]] = None,
    ) -> None:
        """
        Non-conditional Generator's constructor.

        Parameters
        ----------
        z_input : int
            The dimension of the noise tensor.
        output_cells_dim : int
            The dimension of the output cells (number of genes).
        gen_layers : typing.List[int]
            List of integers corresponding to the number of neurons
            at each hidden layer of the generator.
        library_size : typing.Optional[typing.Union[int, None]]
            Total number of counts per generated cell.
        """
        super(Generator, self).__init__()

        self.z_input = z_input
        self.output_cells_dim = output_cells_dim
        self.gen_layers = gen_layers
        self.library_size = library_size

        self._create_generator()

    def forward(self, noise: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Function for completing a forward pass of the generator.

        Parameters
        ----------
        noise : torch.Tensor
            The noise used as input by the generator.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        torch.Tensor
            The output of the generator (genes of the generated cell).
        """
        return self._generator(noise)

    def _create_generator(self) -> None:
        """Method for creating the Generator's network."""
        layers = []
        input_size = self.z_input
        for output_size in self.gen_layers:
            layers.append(self._create_generator_block(input_size, output_size))
            input_size = output_size  # update input size for the next layer

        # outermost layer
        layers.append(
            self._create_generator_block(
                input_size, self.output_cells_dim, self.library_size, final_layer=True
            )
        )

        self._generator = nn.Sequential(*layers)

    @staticmethod
    def _create_generator_block(
        input_dim: int,
        output_dim: int,
        library_size: typing.Optional[typing.Union[int, None]] = None,
        final_layer: typing.Optional[bool] = False,
        *args,
        **kwargs
    ) -> nn.Sequential:
        """
        Function for creating a sequence of operations corresponding to
        a Generator block; a linear layer, a batchnorm (except in the final block),
        a ReLU, and LSN in the final layer.

        Parameters
        ----------
        input_dim : int
            The block's input dimensions.
        output_dim : int
            The block's output dimensions.
        library_size : typing.Optional[typing.Union[int, None]], optional
            Total number of counts per generated cell, by default None.
        final_layer : typing.Optional[bool], optional
            Indicates if the block contains the final layer, by default False.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        nn.Sequential
             Sequential container containing the modules.
        """

        linear_layer = nn.Linear(input_dim, output_dim)

        if not final_layer:
            nn.init.xavier_uniform_(linear_layer.weight)
            return nn.Sequential(
                linear_layer,
                nn.BatchNorm1d(output_dim),
                nn.ReLU(inplace=True),
            )
        else:
            # * Unable to find variance_scaling_initializer() with FAN_AVG mode
            nn.init.kaiming_normal_(
                linear_layer.weight, mode="fan_in", nonlinearity="relu"
            )
            torch.nn.init.zeros_(linear_layer.bias)

            if library_size is not None:
                return nn.Sequential(linear_layer, ReLU(), LSN(library_size))
            else:
                return nn.Sequential(linear_layer, ReLU())


class ConditionalGenerator(Generator):
    def __init__(
        self,
        z_input: int,
        output_cells_dim: int,
        num_classes: int,
        gen_layers: typing.List[int],
        library_size: typing.Optional[typing.Union[int, None]] = None,
    ) -> None:
        """
        Conditional Generator's constructor.

        Parameters
        ----------
        z_input : int
            The dimension of the noise tensor.
        output_cells_dim : int
            The dimension of the output cells (number of genes).
        num_classes : int
            Number of clusters.
        gen_layers : typing.List[int]
            List of integers corresponding to the number of neurons
            at each hidden layer of the generator.
        library_size : typing.Optional[typing.Union[int, None]], optional
            Total number of counts per generated cell, by default None.
        """
        self.num_classes = num_classes

        super(ConditionalGenerator, self).__init__(
            z_input, output_cells_dim, gen_layers, library_size
        )

    def forward(
        self, noise: torch.Tensor, labels: torch.Tensor = None, *args, **kwargs
    ) -> torch.Tensor:
        """
        Function for completing a forward pass of the generator.

        Parameters
        ----------
        noise : torch.Tensor
            The noise used as input by the generator.
        labels : torch.Tensor
            Tensor containing labels corresponding to cells to generate.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        torch.Tensor
            The output of the generator (genes of the generated cell).
        """
        y = noise
        for layer in self._generator:
            if isinstance(layer, ConditionalBatchNorm):
                y = layer(y, labels)
            else:
                y = layer(y)
        return y

    def _create_generator(self) -> None:
        """Method for creating the Generator's network."""
        self._generator = nn.ModuleList()
        input_size = self.z_input
        for output_size in self.gen_layers:
            layers = self._create_generator_block(
                input_size, output_size, num_classes=self.num_classes
            )
            for layer in layers:
                self._generator.append(layer)
            input_size = output_size  # update input size for the next layer

        # outermost layer
        self._generator.append(
            self._create_generator_block(
                input_size,
                self.output_cells_dim,
                self.library_size,
                final_layer=True,
                num_classes=self.num_classes,
            )
        )

    @staticmethod
    def _create_generator_block(
        input_dim: int,
        output_dim: int,
        library_size: typing.Optional[typing.Union[int, None]] = None,
        final_layer: typing.Optional[bool] = False,
        num_classes: int = None,
        *args,
        **kwargs
    ) -> typing.Union[nn.Sequential, tuple]:
        """
        Function for creating a sequence of operations corresponding to
        a Conditional Generator block; a linear layer, a conditional
        batchnorm (except in the final block), a ReLU, and LSN in the final layer.

        Parameters
        ----------
        input_dim : int
            The block's input dimensions.
        output_dim : int
            The block's output dimensions.
        library_size : typing.Optional[typing.Union[int, None]], optional
            Total number of counts per generated cell, by default None.
        final_layer : typing.Optional[bool], optional
            Indicates if the block contains the final layer, by default False.
        num_classes : int
            Number of clusters.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        typing.Union[nn.Sequential, tuple]
             Sequential container or tuple containing modules.
        """

        linear_layer = nn.Linear(input_dim, output_dim)

        if not final_layer:
            nn.init.xavier_uniform_(linear_layer.weight)
            return (
                linear_layer,
                ConditionalBatchNorm(output_dim, num_classes),
                nn.ReLU(inplace=True),
            )
        else:
            nn.init.kaiming_normal_(
                linear_layer.weight, mode="fan_in", nonlinearity="relu"
            )
            torch.nn.init.zeros_(linear_layer.bias)

            if library_size is not None:
                return nn.Sequential(linear_layer, ReLU(), LSN(library_size))
            else:
                return nn.Sequential(linear_layer, ReLU())
