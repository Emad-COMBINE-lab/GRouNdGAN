import itertools
import typing

import torch
from layers.lsn import LSN
from layers.masked_linear import MaskedLinear
from torch import nn
from torch.nn.modules.activation import ReLU


class CausalGenerator(nn.Module):
    def __init__(
        self,
        z_input: int,
        noise_per_gene: int,
        depth_per_gene: int,
        width_scale_per_gene: int,
        causal_controller: nn.Module,
        causal_graph: typing.Dict[int, typing.Set[int]],
        library_size: typing.Optional[typing.Union[int, None]] = None,
        device: typing.Optional[str] = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """
        Causal Generator's constructor.

        Parameters
        ----------
        z_input : int
            The dimension of the noise tensor.
        noise_per_gene : int
            Dimension of the latent space from which the noise vectors used by target generators is sampled.
        depth_per_gene : int
            Depth of the target generator networks.
        width_scale_per_gene : int
            The width scale used for the target generator networks.
            if width_scale_per_gene = 2 and a gene is regulated by 10 TFs and 1 noise vector,
            the width of the target gene generator will be 2 * (10 + 1) = 22.
            Assuming 1000 target genes, each regulated by 10 TFs and 1 noise, the total width of the
            sparse target generator will be 22000.
        causal_controller : nn.Module
            Causal controller module (retrieved from checkpoint if pretrained). It is a GAN trained on
            genes and TFs with the LSN layer removed after training. It cannot be trained on TFs only since the
            library size has to be enforced. However, during causal generator training, only TFs are used.
        causal_graph : typing.Dict[int, typing.Set[int]]
            The causal graph is a dictionary representing the TRN to impose. It has the following format:
            {target gene index: {TF1 index, TF2 index, ...}}. This causal graph has to be acyclic and bipartite.
            A TF cannot be regulated by another TF.
            Invalid: {1: {2, 3, {4, 6}}, ...} - a regulator (TF) is regulated by another regulator (TF)
            Invalid: {1: {2, 3, 4}, 2: {4, 3, 5}, ...} - a regulator (TF) is also regulated
            Invalid: {4: {2, 3}, 2: {4, 3}} - contains a cycle

            Valid causal graph example: {1: {2, 3, 4}, 6: {5, 4, 2}, ...}
        library_size : typing.Optional[typing.Union[int, None]], optional
            Total number of counts per generated cell, by default None
        device : typing.Optional[str], optional
            Specifies to train on 'cpu' or 'cuda'. Only 'cuda' is supported for training the
            GAN but 'cpu' can be used for inference, by default "cuda" if torch.cuda.is_available() else"cpu".
        """
        super().__init__()
        self.z_input = z_input
        self.noise_per_gene = noise_per_gene
        self.depth_per_gene = depth_per_gene
        self.width_scale_per_gene = width_scale_per_gene
        self.causal_graph = causal_graph
        self.library_size = library_size
        self.device = device
        self._causal_controller = causal_controller
        self._generator = None

        self.genes = list(self.causal_graph.keys())
        self.regulators = list(  # all gene regulating TFs (can contain duplicate TFs)
            itertools.chain.from_iterable(self.causal_graph.values())
        )
        self.tfs = list(set(self.regulators))

        # if a gene has X number of regulators (TFs + noises), it will have a
        # hidden layer with the width of (hidden_width * num_regulators)
        self.num_genes = len(self.genes)
        self.num_noises = self.num_genes * self.noise_per_gene  # 1 noise vector / gene
        self.num_tfs = len(self.tfs)  # number of TFs

        # For performing perturbation studies.
        # In perturbation mode, TF expressions, noise vectors, and LSN layer are frozen.
        self.pert_mode = False
        self.tf_expressions = None
        self.noise = None

        self._lsn = LSN(self.library_size)

        self._create_generator()
        self._create_labeler()

    def forward(self, noise: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Function for completing a forward pass of the generator. This includes a
        forward pass of the causal controller to generate TFs. TFs and generated
        noise are then used to complete a forward pass of the causal generator.

        Parameters
        ----------
        noise : torch.Tensor
            The noise used as input by the causal controller.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        torch.Tensor
            The output of the causal generator (gene expression matrix).
        """
        tf_expressions = self._causal_controller(noise)
        tf_expressions = tf_expressions[:, self.tfs]
        tf_expressions = tf_expressions.detach()

        # use the same tf expressions as the previous forward pass in perturbation mode
        if self.pert_mode:
            if self.tf_expressions is not None:
                tf_expressions = self.tf_expressions
            else:
                self.tf_expressions = tf_expressions

        batch_size = tf_expressions.shape[0]

        # create placeholder for cells
        cells = torch.zeros(batch_size, self.num_tfs + self.num_genes).to(self.device)
        cells = cells.index_add_(
            1, torch.tensor(self.tfs).to(self.device), tf_expressions
        )

        # lazy way of avoiding a circular dependency
        # FIXME: circular dependency
        from gans.gan import GAN

        noise = GAN._generate_noise(batch_size, self.num_noises, self.device)

        if self.pert_mode:
            if self.noise is not None:
                noise = self.noise
            else:
                self.noise = noise

        regulators = torch.cat([tf_expressions, noise], dim=1)
        gene_expression = self._generator(regulators)

        cells = cells.index_add_(
            1, torch.tensor(self.genes).to(self.device), gene_expression
        )
        if self.library_size is not None:
            # reuse previous LSN scale in perturbation mode
            cells = self._lsn(cells, reuse_scale=self.pert_mode)
        return cells

    def _create_generator(self) -> None:
        """
        Method for creating the Causal Generator's network. An independent generator can be
        created for each gene. In that case, a pass of the causal generator would require
        a pass of generator networks individually in a loop (since all gene expressions
        are needed before being passed to the LSN layer), which is very inefficient.

        Instead, we create a single large Causal Generator containing sparse connections
        to logically create independent generators for each gene. This is done by creating 3 masks:

        input mask: contains connections between genes and their regulating TFs/noise
        hidden mask: contains connections between hidden layers such that there is no connection between hidden layers of two genes' generators
        output mask: contains connections between hidden layers of each gene's generator and its expression (before LSN)

        The MaskedLinear module is used to mask weights and gradients in linear layers.
        """
        hidden_dims = (
            len(self.regulators) + self.num_noises
        ) * self.width_scale_per_gene

        # noise mask will be added to TF mask
        input_mask = torch.zeros(self.num_tfs, hidden_dims).to(self.device)
        hidden_mask = torch.zeros(hidden_dims, hidden_dims).to(self.device)
        output_mask = torch.zeros(hidden_dims, self.num_genes).to(self.device)

        prev_gene_hidden_dims = 0
        for gene, gene_regulators in self.causal_graph.items():
            gene_idx = self.genes.index(gene)
            curr_gene_hidden_dims = self.width_scale_per_gene * (
                len(gene_regulators) + self.noise_per_gene
            )
            for gene_regulator in gene_regulators:
                gene_regulator_idx = self.tfs.index(gene_regulator)

                # mask for the tfs
                input_mask[
                    gene_regulator_idx,
                    prev_gene_hidden_dims : prev_gene_hidden_dims
                    + curr_gene_hidden_dims,
                ] = 1

            # mask for the noises
            noise_mask = torch.zeros(self.noise_per_gene, hidden_dims).to(self.device)
            noise_mask[
                :, prev_gene_hidden_dims : prev_gene_hidden_dims + curr_gene_hidden_dims
            ] = 1
            input_mask = torch.cat([input_mask, noise_mask])

            # mask for hidden layer
            hidden_mask[
                prev_gene_hidden_dims : prev_gene_hidden_dims + curr_gene_hidden_dims,
                prev_gene_hidden_dims : prev_gene_hidden_dims + curr_gene_hidden_dims,
            ] = 1

            # mask for final layer
            output_mask[
                prev_gene_hidden_dims : prev_gene_hidden_dims + curr_gene_hidden_dims,
                gene_idx,
            ] = 1

            prev_gene_hidden_dims += curr_gene_hidden_dims

        generator_layers = nn.ModuleList()

        # input block
        generator_layers.append(self._create_generator_block(input_mask))

        # hidden block
        for _ in range(self.depth_per_gene):
            generator_layers.append(self._create_generator_block(hidden_mask))

        # output block
        generator_layers.append(
            self._create_generator_block(output_mask, final_layer=True)
        )

        self._generator = nn.Sequential(*generator_layers)

    def _create_labeler(self):
        self._labeler = nn.Sequential(
            nn.Linear(self.num_genes, self.num_genes * 2),
            nn.BatchNorm1d(self.num_genes * 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_genes * 2, self.num_genes * 2),
            nn.BatchNorm1d(self.num_genes * 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_genes * 2, self.num_genes * 2),
            nn.BatchNorm1d(self.num_genes * 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_genes * 2, self.num_tfs),
        )

    def _create_generator_block(
        self,
        mask: torch.Tensor,
        library_size: typing.Optional[typing.Union[int, None]] = None,
        final_layer: typing.Optional[bool] = False,
    ) -> nn.Sequential:
        """
        Method for creating a sequence of operations corresponding to
        a masked causal generator block; a masked linear layer,
        a batchnorm (except in the final block), and ReLU.

        Parameters
        ----------
        mask : torch.Tensor
            Mask Tensor with shape (n_input_feature, n_output_feature).
        library_size : typing.Optional[typing.Union[int, None]], optional
            Total number of counts per generated cell, by default None.
        final_layer : typing.Optional[bool], optional
            Indicates if the block contains the final layer, by default False.

        Returns
        -------
        nn.Sequential
             Sequential container containing the modules.
        """
        masked_linear = MaskedLinear(mask, device=self.device)

        if not final_layer:
            nn.init.xavier_uniform_(masked_linear.weight)
            masked_linear.reapply_mask()
            return nn.Sequential(
                masked_linear,
                nn.BatchNorm1d(mask.shape[1]),
                nn.ReLU(inplace=True),
            )

        else:
            nn.init.kaiming_normal_(
                masked_linear.weight, mode="fan_in", nonlinearity="relu"
            )
            masked_linear.reapply_mask()

            torch.nn.init.zeros_(masked_linear.bias)
            if library_size is not None:
                return nn.Sequential(masked_linear, ReLU(), LSN(library_size))
            else:
                return nn.Sequential(masked_linear, ReLU())

    def freeze_causal_controller(self):
        """Freezes the pretrained causal controller and disallows any further updates."""
        for param in self._causal_controller.parameters():
            param.requires_grad = False
