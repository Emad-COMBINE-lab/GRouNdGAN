import os
import typing

import numpy as np
import torch
from networks.critic import Critic
from networks.generator import Generator

from gans.conditional_gan import ConditionalGAN


class ConditionalCatGAN(ConditionalGAN):
    def __init__(
        self,
        genes_no: int,
        batch_size: int,
        latent_dim: int,
        gen_layers: typing.List[int],
        crit_layers: typing.List[int],
        num_classes: int,
        label_ratios: torch.Tensor,
        device: typing.Optional[str] = "cuda" if torch.cuda.is_available() else "cpu",
        library_size: typing.Optional[int] = 20000,
    ) -> None:
        """
        Conditional single-cell RNA-seq GAN using the conditioning method by concatenation.

        Parameters
        ----------
        genes_no : int
            Number of genes in the dataset.
        batch_size : int
            Training batch size.
        latent_dim : int
            Dimension of the latent space from which the noise vector is sampled.
        gen_layers : typing.List[int]
            List of integers corresponding to the number of neurons of each generator layer.
        crit_layers : typing.List[int]
            List of integers corresponding to the number of neurons of each critic layer.
        num_classes : int
            Number of classes in the dataset.
        label_ratios : torch.Tensor
            Tensor containing the ratio of each class in the dataset.
        device : typing.Optional[str], optional
            Specifies to train on 'cpu' or 'cuda'. Only 'cuda' is supported for training the
            GAN but 'cpu' can be used for inference, by default "cuda" if torch.cuda.is_available() else"cpu".
        library_size : typing.Optional[int], optional
            Total number of counts per generated cell, by default 20000.
        """

        self.num_classes = num_classes
        self.label_ratios = label_ratios

        super(ConditionalCatGAN, self).__init__(
            genes_no,
            batch_size,
            latent_dim,
            gen_layers,
            crit_layers,
            device,
            library_size,
        )

    def _get_gradient(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
        epsilon: torch.Tensor,
        labels: torch.Tensor = None,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute the gradient of the critic's scores with respect to interpolations
        of real and fake cells.

        Parameters
        ----------
        real : torch.Tensor
            A batch of real cells.
        fake : torch.Tensor
            A batch of fake cells.
        epsilon : torch.Tensor
            A vector of the uniformly random proportions of real/fake per interpolated cells.
        labels : torch.Tensor
            A batch of real class labels.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        torch.Tensor
            Gradient of the critic's score with respect to interpolated data.
        """

        # Mix real and fake cells together
        interpolates = real * epsilon + fake * (1 - epsilon)

        # Calculate the critic's scores on the mixed data
        critic_interpolates = self.crit(self._cat_one_hot_labels(interpolates, labels))

        # Take the gradient of the scores with respect to the data
        gradient = torch.autograd.grad(
            inputs=interpolates,
            outputs=critic_interpolates,
            grad_outputs=torch.ones_like(critic_interpolates),
            create_graph=True,
            retain_graph=True,
        )[0]
        return gradient

    def _cat_one_hot_labels(
        self, cells: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Concatenates one-hot encoded labels to a tensor.

        Parameters
        ----------
        cells : torch.Tensor
            Tensor to which to concatenate one-hot encoded class labels.
        labels : torch.Tensor
            Class labels to concatenate.

        Returns
        -------
        torch.Tensor
            Tensor with one-hot encoded labels concatenated at the tail.
        """
        one_hot = torch.nn.functional.one_hot(labels, self.num_classes)
        return torch.cat((cells.float(), one_hot.float()), 1)

    def generate_cells(
        self,
        cells_no: int,
        checkpoint: typing.Optional[typing.Union[str, bytes, os.PathLike, None]] = None,
        class_: typing.Optional[typing.Union[int, None]] = None,
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Generate cells from the Conditional GAN model.

        Parameters
        ----------
        cells_no : int
            Number of cells to generate.
        checkpoint : typing.Optional[typing.Union[str, bytes, os.PathLike, None]], optional
            Path to the saved trained model, by default None.
        class_: typing.Optional[typing.Union[int, None]] = None
            Class of the cells to generate. If None, cells with the same ratio per class
            will be generated.

        Returns
        -------
        typing.Tuple[np.ndarray, np.ndarray]
            Gene expression matrix of generated cells and their corresponding class labels.
        """
        if checkpoint is not None:
            self._load(checkpoint)

        batch_no = int(np.ceil(cells_no / self.batch_size))
        fake_cells = []
        fake_labels = []
        for _ in range(batch_no):
            noise = self._generate_noise(self.batch_size, self.latent_dim, self.device)
            if class_ is None:
                labels = self._sample_pseudo_labels(
                    self.batch_size, self.label_ratios
                ).to(self.device)
            else:
                label_ratios = torch.zeros(self.num_classes).to(self.device)
                label_ratios[class_] = 0.99
                labels = self._sample_pseudo_labels(self.batch_size, label_ratios).to(
                    self.device
                )
            fake_cells.append(
                self.gen(self._cat_one_hot_labels(noise, labels)).cpu().detach().numpy()
            )
            fake_labels.append(labels.cpu().detach().numpy())

        return (
            np.concatenate(fake_cells)[:cells_no],
            np.concatenate(fake_labels)[:cells_no],
        )

    def _build_model(self) -> None:
        """Initializes the Generator and Critic."""
        self.gen = Generator(
            self.latent_dim + self.num_classes,
            self.genes_no,
            self.gen_layers,
            self.library_size,
        ).to(self.device)
        self.crit = Critic(self.genes_no + self.num_classes, self.critic_layers).to(
            self.device
        )

    def _train_critic(
        self, real_cells, real_labels, c_lambda
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Trains the critic for one iteration.

        Parameters
        ----------
        real_cells : torch.Tensor
            Tensor containing a batch of real cells.
        real_labels : torch.Tensor
            Tensor containing a batch of real labels (corresponding to real_cells).
        c_lambda : float
            Regularization hyper-parameter for gradient penalty.

        Returns
        -------
        typing.Tuple[torch.Tensor, torch.Tensor]
            The computed critic loss and gradient penalty.
        """
        self.crit_opt.zero_grad()

        fake_noise = self._generate_noise(self.batch_size, self.latent_dim, self.device)

        fake = self.gen(self._cat_one_hot_labels(fake_noise, real_labels))

        crit_fake_pred = self.crit(self._cat_one_hot_labels(fake, real_labels).detach())
        crit_real_pred = self.crit(self._cat_one_hot_labels(real_cells, real_labels))

        epsilon = torch.rand(len(real_cells), 1, device=self.device, requires_grad=True)

        gradient = self._get_gradient(real_cells, fake.detach(), epsilon, real_labels)
        gp = self._gradient_penalty(gradient)

        crit_loss = self._critic_loss(crit_fake_pred, crit_real_pred, gp, c_lambda)

        # Update gradients
        crit_loss.backward(retain_graph=True)

        # Update optimizer
        self.crit_opt.step()

        return crit_loss, gp

    def _train_generator(self) -> torch.Tensor:
        """
        Trains the generator for one iteration.

        Returns
        -------
        torch.Tensor
            Tensor containing only 1 item, the generator loss.
        """
        self.gen_opt.zero_grad()

        fake_noise = self._generate_noise(
            self.batch_size, self.latent_dim, device=self.device
        )

        fake_labels = self._sample_pseudo_labels(self.batch_size, self.label_ratios).to(
            self.device
        )

        fake = self.gen(self._cat_one_hot_labels(fake_noise, fake_labels))
        crit_fake_pred = self.crit(self._cat_one_hot_labels(fake, fake_labels))

        gen_loss = self._generator_loss(crit_fake_pred)
        gen_loss.backward()

        # Update weights
        self.gen_opt.step()

        return gen_loss
