import os
import typing

import torch
from networks.critic import Critic
from networks.generator import Generator
from networks.labeler import Labeler
from networks.masked_causal_generator import CausalGenerator

from gans.gan import GAN


class CausalGAN(GAN):
    def __init__(
        self,
        genes_no: int,
        batch_size: int,
        latent_dim: int,
        noise_per_gene: int,
        depth_per_gene: int,
        width_per_gene: int,
        cc_latent_dim: int,
        cc_layers: typing.List[int],
        cc_pretrained_checkpoint: str,
        crit_layers: typing.List[int],
        causal_graph: typing.Dict[int, typing.Set[int]],
        labeler_layers: typing.List[int],
        device: typing.Optional[str] = "cuda" if torch.cuda.is_available() else "cpu",
        library_size: typing.Optional[int] = 20000,
    ) -> None:
        """
        Causal single-cell RNA-seq GAN (TODO: find a unique name).

        Parameters
        ----------
        genes_no : int
            Number of genes in the dataset.
        batch_size : int
            Training batch size.
        latent_dim : int
            Dimension of the latent space from which the noise vector used by the causal controller is sampled.
        noise_per_gene : int
            Dimension of the latent space from which the noise vectors used by target generators is sampled.
        depth_per_gene : int
            Depth of the target generator networks.
        width_per_gene : int
            The width scale used for the target generator networks.
        cc_latent_dim : int
            Dimension of the latent space from which the noise vector to the causal controller is sampled.
        cc_layers : typing.List[int]
            List of integers corresponding to the number of neurons of each causal controller layer.
        cc_pretrained_checkpoint : str
            Path to the  pretrained causal controller.
        crit_layers : typing.List[int]
            List of integers corresponding to the number of neurons of each critic layer.
        causal_graph : typing.Dict[int, typing.Set[int]]
            The causal graph is a dictionary representing the TRN to impose. It has the following format:
            {target gene index: {TF1 index, TF2 index, ...}}. This causal graph has to be acyclic and bipartite.
            A TF cannot be regulated by another TF.
            Invalid: {1: {2, 3, {4, 6}}, ...} - a regulator (TF) is regulated by another regulator (TF)
            Invalid: {1: {2, 3, 4}, 2: {4, 3, 5}, ...} - a regulator (TF) is also regulated
            Invalid: {4: {2, 3}, 2: {4, 3}} - contains a cycle

            Valid causal graph example: {1: {2, 3, 4}, 6: {5, 4, 2}, ...}
        labeler_layers : typing.List[int]
            List of integers corresponding to the width of each labeler layer.
        device : typing.Optional[str], optional
            Specifies to train on 'cpu' or 'cuda'. Only 'cuda' is supported for training the
            GAN but 'cpu' can be used for inference, by default "cuda" if torch.cuda.is_available() else"cpu".
        library_size : typing.Optional[int], optional
            Total number of counts per generated cell, by default 20000.
        """

        self.causal_controller = Generator(
            z_input=cc_latent_dim,
            output_cells_dim=genes_no,
            gen_layers=cc_layers,
            library_size=None,
        )

        checkpoint = torch.load(
            cc_pretrained_checkpoint, map_location=torch.device(device)
        )
        self.causal_controller.load_state_dict(checkpoint["generator_state_dict"])

        self.noise_per_gene = noise_per_gene
        self.depth_per_gene = depth_per_gene
        self.width_per_gene = width_per_gene
        self.causal_graph = causal_graph
        self.labeler_layers = labeler_layers
        super().__init__(
            genes_no,
            batch_size,
            latent_dim,
            None,
            crit_layers,
            device=device,
            library_size=library_size,
        )

    def _build_model(self) -> None:
        """Instantiates the Generator and Critic."""
        self.gen = CausalGenerator(
            self.latent_dim,
            self.noise_per_gene,
            self.depth_per_gene,
            self.width_per_gene,
            self.causal_controller,
            self.causal_graph,
            self.library_size,
        ).to(self.device)
        self.gen.freeze_causal_controller()

        self.crit = Critic(self.genes_no, self.critic_layers).to(self.device)

        # the number of genes and TFs are resolved by the causal generator during its instantiation
        self.labeler = Labeler(
            self.gen.num_genes, self.gen.num_tfs, self.labeler_layers
        )
        self.antilabeler = Labeler(
            self.gen.num_genes, self.gen.num_tfs, self.labeler_layers
        )

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
                "labeler_state_dict": self.labeler.module.state_dict(),
                "antilabeler_state_dict": self.antilabeler.module.state_dict(),
                "generator_optimizer_state_dict": self.gen_opt.state_dict(),
                "critic_optimizer_state_dict": self.crit_opt.state_dict(),
                "labeler_optimizer_state_dict": self.labeler_opt.state_dict(),
                "antilabeler_optimizer_state_dict": self.antilabeler_opt.state_dict(),
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
        Loads a saved causal GAN model (.pth file).

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
            # The causal GAN performs better when using batch stats (model.train() mode)

            self.gen.train()
            self.crit.train()

        elif mode == "training":
            self.gen.train()
            self.crit.train()

            self.step = checkpoint["step"] + 1
            self.gen_opt.load_state_dict(checkpoint["generator_optimizer_state_dict"])
            self.crit_opt.load_state_dict(checkpoint["critic_optimizer_state_dict"])
            self.gen_lr_scheduler.load_state_dict(checkpoint["generator_lr_scheduler"])
            self.crit_lr_scheduler.load_state_dict(checkpoint["critic_lr_scheduler"])
            self.labeler.load_state_dict(checkpoint["labeler_state_dict"])
            self.antilabeler.load_state_dict(checkpoint["antilabeler_state_dict"])
            self.labeler_opt.load_state_dict(checkpoint["labeler_optimizer_state_dict"])
            self.antilabeler_opt.load_state_dict(
                checkpoint["antilabeler_optimizer_state_dict"]
            )

        else:
            raise ValueError("mode should be 'inference' or 'training'")

    def _train_labelers(self, real_cells: torch.Tensor) -> None:
        """
        Trains the labeler (on real and fake) and anti-labeler (on fake only).

        Parameters
        ----------
        real_cells : torch.Tensor
            Tensor containing a batch of real cells.
        """
        fake_noise = self._generate_noise(self.batch_size, self.latent_dim, self.device)
        fake = self.gen(fake_noise).detach() 

        # train anti-labeler
        self.antilabeler_opt.zero_grad()
        predicted_tfs = self.antilabeler(fake[:, self.gen.module.genes])
        actual_tfs = fake[:, self.gen.module.tfs]
        antilabeler_loss = self.mse(predicted_tfs, actual_tfs)
        antilabeler_loss.backward(retain_graph=True)
        self.antilabeler_opt.step()

        # train labeler on fake data
        self.labeler_opt.zero_grad()
        predicted_tfs = self.labeler(fake[:, self.gen.module.genes])
        labeler_floss = self.mse(predicted_tfs, actual_tfs)
        labeler_floss.backward()
        self.labeler_opt.step()

        # train labeler on real data
        self.labeler_opt.zero_grad()
        predicted_tfs = self.labeler(real_cells[:, self.gen.module.genes])
        actual_tfs = real_cells[:, self.gen.module.tfs]
        labeler_rloss = self.mse(predicted_tfs, actual_tfs)
        labeler_rloss.backward()
        self.labeler_opt.step()

    def _train_generator(self) -> torch.Tensor:
        """
        Trains the causal generator for one iteration.
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

        predicted_tfs = self.labeler(fake[:, self.gen.module.genes])
        actual_tfs = fake[:, self.gen.module.tfs]
        labeler_loss = self.mse(predicted_tfs, actual_tfs)

        predicted_tfs = self.antilabeler(fake[:, self.gen.module.genes])
        antilabeler_loss = self.mse(predicted_tfs, actual_tfs)

        crit_fake_pred = self.crit(fake)
        gen_loss = self._generator_loss(crit_fake_pred)

        # comment for ablation of labeler and anti-labeler (GRouNdGAN_def_even_ablation1)
        gen_loss += labeler_loss + antilabeler_loss
        
        # uncomment for ablation of anti-labeler but keeping the labeler (GRouNdGAN_def_even_ablation2)
        # gen_loss += labeler_loss

        # uncomment for ablation of labeler but keeping the anti-labeler (GRouNdGAN_def_even_ablation3)
        # gen_loss += antilabeler_loss

        
        gen_loss.backward()

        # Update weights
        self.gen_opt.step()

        return gen_loss

    # FIXME: A lot of code duplication here with the parent train() method.
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
        labeler_alpha: float,
        antilabeler_alpha: float,
        labeler_training_interval: int,
        checkpoint: typing.Optional[typing.Union[str, bytes, os.PathLike, None]] = None,
        output_dir: typing.Optional[str] = "output",
        summary_freq: typing.Optional[int] = 5000,
        plt_freq: typing.Optional[int] = 10000,
        save_feq: typing.Optional[int] = 10000,
    ) -> None:
        """
        Method for training the causal GAN.

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
        labeler_alpha : float
            Labeler's learning rate value.
        antilabeler_alpha : float
            Anti-labeler's learning rate value.
        labeler_training_interval: int
            The number of steps after which the labeler and anti-labeler are trained.
            If 20, the labeler and anti-labeler will be trained every 20 steps.
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

        self.labeler_opt = torch.optim.AdamW(
            self.labeler.parameters(),
            lr=labeler_alpha,
            betas=(beta1, beta2),
            amsgrad=True,
        )

        self.antilabeler_opt = torch.optim.AdamW(
            self.antilabeler.parameters(),
            lr=antilabeler_alpha,
            betas=(beta1, beta2),
            amsgrad=True,
        )

        # for the labeler and anti-labeler
        self.mse = torch.nn.MSELoss()

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
        self.labeler.train()
        self.antilabeler.train()

        # We only accept training on GPU since training on CPU is impractical.
        self.device = "cuda"
        self.gen = torch.nn.DataParallel(self.gen)
        self.crit = torch.nn.DataParallel(self.crit)
        self.labeler = torch.nn.DataParallel(self.labeler)
        self.antilabeler = torch.nn.DataParallel(self.antilabeler)

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

            if should_run(labeler_training_interval):
                self._train_labelers(real_cells)

            # Log and visualize progress
            if should_run(summary_freq):
                gen_mean = sum(generator_losses[-summary_freq:]) / summary_freq
                crit_mean = sum(critic_losses[-summary_freq:]) / summary_freq

                if self.step == summary_freq:
                    self._add_tensorboard_graph(output_dir, self.tb_fake_noise, self.tb_fake)

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
