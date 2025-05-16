import os
import typing

import matplotlib.pyplot as plt
import numpy as np
import torch
from networks.critic import Critic
from networks.generator import Generator
from sc_dataset import get_loader
from sklearn.manifold import TSNE
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter


class GAN:
    def __init__(
        self,
        genes_no: int,
        batch_size: int,
        latent_dim: int,
        gen_layers: typing.List[int],
        crit_layers: typing.List[int],
        device: typing.Optional[str] = "cuda" if torch.cuda.is_available() else "cpu",
        library_size: typing.Optional[int] = 20000,
    ) -> None:
        """
        Non-conditional single-cell RNA-seq GAN.

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
        device : typing.Optional[str], optional
            Specifies to train on 'cpu' or 'cuda'. Only 'cuda' is supported for training the
            GAN but 'cpu' can be used for inference, by default "cuda" if torch.cuda.is_available() else"cpu".
        library_size : typing.Optional[int], optional
            Total number of counts per generated cell, by default 20000.
        """
        torch.cuda.empty_cache()

        self.genes_no = genes_no
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.gen_layers = gen_layers
        self.critic_layers = crit_layers
        self.device = device
        self.library_size = library_size

        self.gen = None
        self.crit = None
        self._build_model()

        self.step = 0
        self.gen_opt = None
        self.crit_opt = None
        self.gen_lr_scheduler = None
        self.crit_lr_scheduler = None

    @staticmethod
    def _generate_noise(batch_size: int, latent_dim: int, device: str) -> torch.Tensor:
        """
        Function for creating noise vectors: Given the dimensions (batch_size, latent_dim).

        Parameters
        ----------
        batch_size : int
            The number of samples to generate (normally equal to training batch size).
        latent_dim : int
            Dimension of the latent space to sample from.
        device : str
            The device type.

        Returns
        -------
        torch.Tensor
            A tensor filled with random numbers from the standard normal distribution.
        """
        return torch.randn(batch_size, latent_dim, device=device)

    @staticmethod
    def _set_exponential_lr(
        optimizer: torch.optim.Optimizer,
        alpha_0: float,
        alpha_final: float,
        max_steps: int,
    ) -> ExponentialLR:
        """
        Sets up exponentially decaying learning rate scheduler to be used
        with the optimizer.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer for which to create an exponential learning rate scheduler.
        alpha_0 : float
            Initial learning rate.
        alpha_final : float
            Final learning rate.
        max_steps : int
            Total number of training steps. When current_step=max_steps, alpha_final
            will be set as the learning rate.

        Returns
        -------
        ExponentialLR
            Exponential learning rate scheduler. Call the step() function on this
            scheduler in the training loop.
        """

        # Find the decay rate of the exponential learning rate
        decay_rate = (alpha_final / alpha_0) ** (1 / max_steps)
        return ExponentialLR(optimizer=optimizer, gamma=decay_rate)

    @staticmethod
    def _critic_loss(
        crit_fake_pred: torch.Tensor,
        crit_real_pred: torch.Tensor,
        gp: torch.Tensor,
        c_lambda: float,
    ) -> torch.Tensor:
        """
        Compute critic's loss given the its scores on real and fake cells,
        the gradient penalty, and gradient penalty regularization hyper-parameter.

        Parameters
        ----------
        crit_fake_pred : torch.Tensor
            Critic's score on fake cells.
        crit_real_pred : torch.Tensor
            Critic's score on real cells.
        gp : torch.Tensor
            Unweighted gradient penalty
        c_lambda : float
            Regularization hyper-parameter to be used with the gradient penalty
            in the WGAN loss.

        Returns
        -------
        torch.Tensor
            Critic's loss for the current batch.
        """
        return torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda * gp

    @staticmethod
    def _generator_loss(crit_fake_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute the generator loss from the critic's score of the generated cells.

        Parameters
        ----------
        crit_fake_pred : torch.Tensor
            The critic's score on fake generated cells.

        Returns
        -------
        torch.Tensor
            Generator's loss value for the current batch.
        """
        return -1.0 * torch.mean(crit_fake_pred)

    def _get_gradient(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
        epsilon: torch.Tensor,
        *args,
        **kwargs,
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

        Returns
        -------
        torch.Tensor
            Gradient of the critic's score with respect to interpolated data.
        """

        # Mix real and fake cells together
        interpolates = real * epsilon + fake * (1 - epsilon)

        # Calculate the critic's scores on the mixed data
        critic_interpolates = self.crit(interpolates)

        # Take the gradient of the scores with respect to the data
        gradient = torch.autograd.grad(
            inputs=interpolates,
            outputs=critic_interpolates,
            grad_outputs=torch.ones_like(critic_interpolates),
            create_graph=True,
            retain_graph=True,
        )[0]
        return gradient

    @staticmethod
    def _gradient_penalty(gradient: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient penalty given a gradient.

        Parameters
        ----------
        gradient : torch.Tensor
            The gradient of the critic's score with respect to
            the interpolated data.

        Returns
        -------
        torch.Tensor
            Gradient penalty of the given gradient.
        """
        gradient = gradient.view(len(gradient), -1)
        gradient_norm = gradient.norm(2, dim=1)

        return torch.mean((gradient_norm - 1) ** 2)

    def generate_cells(
        self,
        cells_no: int,
        checkpoint: typing.Optional[typing.Union[str, bytes, os.PathLike, None]] = None,
    ) -> np.ndarray:
        """
        Generate cells from the GAN model.

        Parameters
        ----------
        cells_no : int
            Number of cells to generate.
        checkpoint : typing.Optional[typing.Union[str, bytes, os.PathLike, None]], optional
            Path to the saved trained model, by default None.

        Returns
        -------
        np.ndarray
            Gene expression matrix of generated cells.
        """
        if checkpoint is not None:
            self._load(checkpoint)

        # find how many batches to generate
        batch_no = int(np.ceil(cells_no / self.batch_size))

        fake_cells = []
        for _ in range(batch_no):
            noise = self._generate_noise(self.batch_size, self.latent_dim, self.device)
            fake_cells.append(self.gen(noise).cpu().detach().numpy())

        return np.concatenate(fake_cells)[:cells_no]

    def _save(self, path: typing.Union[str, bytes, os.PathLike]) -> None:
        """
        Saves the model.

        Parameters
        ----------
        path : typing.Union[str, bytes, os.PathLike]
            Directory to save the model.
        """
        output_dir = path + "/checkpoints"
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        torch.save(
            {
                "step": self.step,
                "generator_state_dict": self.gen.module.state_dict(),
                "critic_state_dict": self.crit.module.state_dict(),
                "generator_optimizer_state_dict": self.gen_opt.state_dict(),
                "critic_optimizer_state_dict": self.crit_opt.state_dict(),
                "generator_lr_scheduler": self.gen_lr_scheduler.state_dict(),
                "critic_lr_scheduler": self.crit_lr_scheduler.state_dict(),
            },
            f"{path}/checkpoints/step_{self.step}.pth",
        )

    def _load(
        self,
        path: typing.Union[str, bytes, os.PathLike],
        mode: typing.Optional[str] = "inference",
    ) -> None:
        """
        Loads a saved model (.pth file).

        Parameters
        ----------
        path : typing.Union[str, bytes, os.PathLike]
            Path to the saved model.
        mode : typing.Optional[str], optional
            Specify if the loaded model is used for 'inference' or 'training', by default "inference".

        Raises
        ------
        ValueError
            If a mode other than 'inference' or 'training' is specified.
        """

        checkpoint = torch.load(path, map_location=torch.device(self.device))

        self.gen.load_state_dict(checkpoint["generator_state_dict"])
        self.crit.load_state_dict(checkpoint["critic_state_dict"])

        if mode == "inference":
            self.gen.eval()
            self.crit.eval()

        elif mode == "training":
            self.gen.train()
            self.crit.train()

            self.step = checkpoint["step"] + 1
            self.gen_opt.load_state_dict(checkpoint["generator_optimizer_state_dict"])
            self.crit_opt.load_state_dict(checkpoint["critic_optimizer_state_dict"])
            self.gen_lr_scheduler.load_state_dict(checkpoint["generator_lr_scheduler"])
            self.crit_lr_scheduler.load_state_dict(checkpoint["critic_lr_scheduler"])

        else:
            raise ValueError("mode should be 'inference' or 'training'")

    def _build_model(self) -> None:
        """Instantiates the Generator and Critic."""
        self.gen = Generator(
            self.latent_dim, self.genes_no, self.gen_layers, self.library_size
        ).to(self.device)
        self.crit = Critic(self.genes_no, self.critic_layers).to(self.device)

    def _get_loaders(
        self,
        train_file: typing.Union[str, bytes, os.PathLike],
        validation_file: typing.Union[str, bytes, os.PathLike],
    ) -> typing.Tuple[DataLoader, DataLoader]:
        """
        Gets training and validation DataLoaders for training.

        Parameters
        ----------
        train_file : typing.Union[str, bytes, os.PathLike]
            Path to training files.
        validation_file : typing.Union[str, bytes, os.PathLike]
            Path to validation files.

        Returns
        -------
        typing.Tuple[DataLoader, DataLoader]
            Train and Validation Dataloaders.
        """
        return get_loader(train_file, self.batch_size), get_loader(validation_file)

    def _add_tensorboard_graph(
        self,
        output_dir: typing.Union[str, bytes, os.PathLike],
        gen_data: typing.Union[torch.Tensor, typing.Tuple[torch.Tensor]],
        crit_data: typing.Union[torch.Tensor, typing.Tuple[torch.Tensor]],
    ) -> None:
        """
        Adds the model graph to TensorBoard.

        Parameters
        ----------
        output_dir : typing.Union[str, bytes, os.PathLike]
            Directory to save the tfevents.
        gen_data : typing.Union[torch.Tensor, typing.Tuple[torch.Tensor]]
            Input to the generator.
        crit_data : typing.Union[torch.Tensor, typing.Tuple[torch.Tensor]]
            Input to the critic.
        """
        with SummaryWriter(f"{output_dir}/TensorBoard/model/generator") as w:
            w.add_graph(self.gen.module, gen_data)
        with SummaryWriter(f"{output_dir}/TensorBoard/model/critic") as w:
            w.add_graph(self.crit.module, crit_data)

    def _update_tensorboard(
        self,
        gen_loss: float,
        crit_loss: float,
        gp: torch.Tensor,
        gen_lr: float,
        crit_lr: float,
        output_dir: typing.Union[str, bytes, os.PathLike],
    ) -> None:
        """
        Updates the TensorBoard summary logs.

        Parameters
        ----------
        gen_loss : float
            Generator loss.
        crit_loss : float
            Critic loss.
        gp : torch.Tensor
            Gradient penalty.
        gen_lr : float
            Generator's optimizer learning rate.
        crit_lr : float
            Critic's optimizer learning rate.
        output_dir : typing.Union[str, bytes, os.PathLike]
            Directory to save the tfevents.
        """

        with SummaryWriter(f"{output_dir}/TensorBoard/generator") as w:
            w.add_scalar("loss", gen_loss, self.step)

        with SummaryWriter(f"{output_dir}/TensorBoard/critic") as w:
            w.add_scalar("loss", crit_loss, self.step)

        with SummaryWriter(f"{output_dir}/TensorBoard/gp") as w:
            w.add_scalar("gradient penalty", gp, self.step)

        with SummaryWriter(f"{output_dir}/TensorBoard/generator_lr") as w:
            w.add_scalar("learning rate", gen_lr, self.step)

        with SummaryWriter(f"{output_dir}/TensorBoard/critic_lr") as w:
            w.add_scalar("learning rate", crit_lr, self.step)

    def _generate_tsne_plot(
        self,
        valid_loader: DataLoader,
        output_dir: typing.Union[str, bytes, os.PathLike],
    ) -> None:
        """
        Generates t-SNE plots during training.

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

        fake_cells = self.generate_cells(len(valid_loader.dataset))
        valid_cells, _ = next(iter(valid_loader))

        embedded_cells = TSNE().fit_transform(
            np.concatenate((valid_cells, fake_cells), axis=0)
        )

        real_embedding = embedded_cells[0 : valid_cells.shape[0], :]
        fake_embedding = embedded_cells[valid_cells.shape[0] :, :]

        plt.clf()
        fig = plt.figure()

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
            loc="lower left", numpoints=1, ncol=2, fontsize=8, bbox_to_anchor=(0, 0)
        )

        plt.savefig(tsne_path + "/step_" + str(self.step) + ".jpg")

        with SummaryWriter(f"{output_dir}/TensorBoard/TSNE") as w:
            w.add_figure("t-SNE plot", fig, self.step)

        plt.close()

    def _train_critic(
        self, real_cells: torch.Tensor, real_labels: torch.Tensor, c_lambda: float
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
        fake = self.gen(fake_noise)

        self.tb_fake = fake # for tensorboard model graph

        crit_fake_pred = self.crit(fake.detach())
        crit_real_pred = self.crit(real_cells)

        epsilon = torch.rand(len(real_cells), 1, device=self.device, requires_grad=True)

        gradient = self._get_gradient(real_cells, fake.detach(), epsilon)
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

        self.tb_fake_noise = fake_noise # for tensorboard model graph

        fake = self.gen(fake_noise)
        crit_fake_pred = self.crit(fake)

        gen_loss = self._generator_loss(crit_fake_pred)
        gen_loss.backward()

        # Update weights
        self.gen_opt.step()

        return gen_loss

    def train(
        self,
        train_files: str,
        valid_files: str,
        critic_iter: int,
        max_steps: int,
        c_lambda: float,
        beta1: float,
        beta2: float,
        gen_alpha_0: float,
        gen_alpha_final: float,
        crit_alpha_0: float,
        crit_alpha_final: float,
        checkpoint: typing.Optional[typing.Union[str, bytes, os.PathLike, None]] = None,
        output_dir: typing.Optional[str] = "output",
        summary_freq: typing.Optional[int] = 5000,
        plt_freq: typing.Optional[int] = 10000,
        save_feq: typing.Optional[int] = 10000,
    ) -> None:
        """
        Method for training the GAN.

        Parameters
        ----------
        train_files : str
            Path to training set files (TFrecords supported for now).
        valid_files : str
            Path to validation set files (TFrecords supported for now).
        critic_iter : int
            Number of training iterations of the critic for each iteration on the generator.
        max_steps : int
            Maximum number of steps to train the GAN.
        c_lambda : float
            Regularization hyper-parameter for gradient penalty.
        beta1 : float
            Coefficients used for computing running averages of gradient in the optimizer.
        beta2 : float
            Coefficient used for computing running averages of gradient squares in the optimizer.
        gen_alpha_0 : float
            Generator's initial learning rate value.
        gen_alpha_final : float
            Generator's final learning rate value.
        crit_alpha_0 : float
            Critic's initial learning rate value.
        crit_alpha_final : float
            Critic's final learning rate value.
        checkpoint : typing.Optional[typing.Union[str, bytes, os.PathLike, None]], optional
            Path to a trained model; if specified, the checkpoint is be used to resume training, by default None.
        output_dir : typing.Optional[str], optional
            Directory to which plots, tfevents, and checkpoints will be saved, by default "output".
        summary_freq : typing.Optional[int], optional
            Period between summary logs to TensorBoard, by default 5000.
        plt_freq : typing.Optional[int], optional
            Period between t-SNE plots, by default 10000.
        save_feq : typing.Optional[int], optional
            Period between saves of the model, by default 10000.
        """

        def should_run(freq):
            return freq > 0 and self.step % freq == 0 and self.step > 0

        loader, valid_loader = self._get_loaders(train_files, valid_files)
        loader_gen = iter(loader)

        # Instantiate optimizers
        self.gen_opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.gen.parameters()),
            lr=gen_alpha_0,
            betas=(beta1, beta2),
            amsgrad=True,
        )

        self.crit_opt = torch.optim.AdamW(
            self.crit.parameters(),
            lr=crit_alpha_0,
            betas=(beta1, beta2),
            amsgrad=True,
        )

        # Exponential Learning Rate
        self.gen_lr_scheduler = self._set_exponential_lr(
            self.gen_opt, gen_alpha_0, gen_alpha_final, max_steps
        )
        self.crit_lr_scheduler = self._set_exponential_lr(
            self.crit_opt, crit_alpha_0, crit_alpha_final, max_steps
        )

        if checkpoint is not None:
            self._load(checkpoint, mode="training")

        self.gen.train()
        self.crit.train()

        # We only accept training on GPU since training on CPU is impractical.
        self.device = "cuda"
        self.gen = torch.nn.DataParallel(self.gen)
        self.crit = torch.nn.DataParallel(self.crit)

        # Main training loop
        generator_losses, critic_losses = [], []
        while self.step <= max_steps:
            try:
                real_cells, real_labels = next(loader_gen)
            except StopIteration:
                loader_gen = iter(loader)
                real_cells, real_labels = next(loader_gen)

            real_cells = real_cells.to(self.device)
            real_labels = real_labels.flatten().to(self.device)

            if self.step != 0:
                mean_iter_crit_loss = 0
                for _ in range(critic_iter):
                    crit_loss, gp = self._train_critic(
                        real_cells, real_labels, c_lambda
                    )
                    mean_iter_crit_loss += crit_loss.item() / critic_iter

                critic_losses += [mean_iter_crit_loss]

                # Update learning rate
                self.crit_lr_scheduler.step()

            gen_loss = self._train_generator()
            self.gen_lr_scheduler.step()

            generator_losses += [gen_loss.item()]

            # Log and visualize progress
            if should_run(summary_freq):
                gen_mean = sum(generator_losses[-summary_freq:]) / summary_freq
                crit_mean = sum(critic_losses[-summary_freq:]) / summary_freq

                # if self.step == summary_freq:
                    # self._add_tensorboard_graph(output_dir, self.tb_fake_noise, self.tb_fake)

                self._update_tensorboard(
                    gen_mean,
                    crit_mean,
                    gp,
                    self.gen_lr_scheduler.get_last_lr()[0],
                    self.crit_lr_scheduler.get_last_lr()[0],
                    output_dir,
                )

            if should_run(plt_freq):
                self._generate_tsne_plot(valid_loader, output_dir)

            if should_run(save_feq):
                self._save(output_dir)
            print("done training step", self.step, flush=True)
            self.step += 1
