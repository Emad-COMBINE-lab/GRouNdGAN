import pickle
import typing
from abc import ABC, abstractmethod
from configparser import ConfigParser

import torch

from gans.causal_gan import CausalGAN
from gans.conditional_gan_cat import ConditionalCatGAN
from gans.conditional_gan_proj import ConditionalProjGAN
from gans.gan import GAN


def parse_list(str_list: str, type_: type) -> list:
    return list(map(type_, str.split(str_list)))


class IGANFactory(ABC):
    """
    Factory that represents a GAN.
    This factory does not keep of created references.
    """

    def __init__(self, parser: ConfigParser) -> None:
        """
        Initialize the factory.

        Parameters
        ----------
        parser : ConfigParser
            Parser for config file containing GAN model and training params.
        """
        self.parser = parser

    @abstractmethod
    def get_gan(self) -> GAN:
        """
        Returns a GAN instance

        Returns
        -------
        GAN
            GAN instance.
        """
        pass

    @abstractmethod
    def get_trainer(self) -> typing.Callable:
        """
        Returns the GAN train function.

        Returns
        -------
        typing.Callable
            GAN train() function.
        """
        pass


class GANFactory(IGANFactory):
    def get_gan(self) -> GAN:
        return GAN(
            genes_no=self.parser.getint("Data", "number of genes"),
            batch_size=self.parser.getint("Training", "batch size"),
            latent_dim=self.parser.getint("Model", "latent dim"),
            gen_layers=parse_list(self.parser["Model"]["generator layers"], int),
            crit_layers=parse_list(self.parser["Model"]["critic layers"], int),
            device=self.parser.get("EXPERIMENT", "device", fallback=None),
            library_size=self.parser.getint("Preprocessing", "library size"),
        )

    def get_trainer(self) -> typing.Callable:
        gan = self.get_gan()
        return lambda: gan.train(
            train_files=self.parser.get("Data", "train"),
            valid_files=self.parser.get("Data", "validation"),
            critic_iter=self.parser.getint("Training", "critic iterations"),
            max_steps=self.parser.getint("Training", "maximum steps"),
            c_lambda=self.parser.getfloat("Model", "lambda"),
            beta1=self.parser.getfloat("Optimizer", "beta1"),
            beta2=self.parser.getfloat("Optimizer", "beta2"),
            gen_alpha_0=self.parser.getfloat("Learning Rate", "generator initial"),
            gen_alpha_final=self.parser.getfloat("Learning Rate", "generator final"),
            crit_alpha_0=self.parser.getfloat("Learning Rate", "critic initial"),
            crit_alpha_final=self.parser.getfloat("Learning Rate", "critic final"),
            checkpoint=self.parser.get("EXPERIMENT", "checkpoint", fallback=None),
            summary_freq=self.parser.getint("Logging", "summary frequency"),
            plt_freq=self.parser.getint("Logging", "plot frequency"),
            save_feq=self.parser.getint("Logging", "save frequency"),
            output_dir=self.parser.get("EXPERIMENT", "output directory"),
        )


class ConditionalCatGANFactory(IGANFactory):
    def get_gan(self) -> ConditionalCatGAN:
        return ConditionalCatGAN(
            genes_no=self.parser.getint("Data", "number of genes"),
            batch_size=self.parser.getint("Training", "batch size"),
            latent_dim=self.parser.getint("Model", "latent dim"),
            gen_layers=parse_list(self.parser["Model"]["generator layers"], int),
            crit_layers=parse_list(self.parser["Model"]["critic layers"], int),
            num_classes=self.parser.getint("Data", "number of classes"),
            label_ratios=torch.Tensor(
                parse_list(self.parser["Data"]["label ratios"], float)
            ),
            device=self.parser.get("EXPERIMENT", "device", fallback=None),
            library_size=self.parser.getint("Preprocessing", "library size"),
        )

    def get_trainer(self) -> typing.Callable:
        gan = self.get_gan()
        return lambda: gan.train(
            train_files=self.parser.get("Data", "train"),
            valid_files=self.parser.get("Data", "validation"),
            critic_iter=self.parser.getint("Training", "critic iterations"),
            max_steps=self.parser.getint("Training", "maximum steps"),
            c_lambda=self.parser.getfloat("Model", "lambda"),
            beta1=self.parser.getfloat("Optimizer", "beta1"),
            beta2=self.parser.getfloat("Optimizer", "beta2"),
            gen_alpha_0=self.parser.getfloat("Learning Rate", "generator initial"),
            gen_alpha_final=self.parser.getfloat("Learning Rate", "generator final"),
            crit_alpha_0=self.parser.getfloat("Learning Rate", "critic initial"),
            crit_alpha_final=self.parser.getfloat("Learning Rate", "critic final"),
            checkpoint=self.parser.get("EXPERIMENT", "checkpoint", fallback=None),
            summary_freq=self.parser.getint("Logging", "summary frequency"),
            plt_freq=self.parser.getint("Logging", "plot frequency"),
            save_feq=self.parser.getint("Logging", "save frequency"),
            output_dir=self.parser.get("EXPERIMENT", "output directory"),
        )


class ConditionalProjGANFactory(IGANFactory):
    def get_gan(self) -> ConditionalProjGAN:
        return ConditionalProjGAN(
            genes_no=self.parser.getint("Data", "number of genes"),
            batch_size=self.parser.getint("Training", "batch size"),
            latent_dim=self.parser.getint("Model", "latent dim"),
            gen_layers=parse_list(self.parser["Model"]["generator layers"], int),
            crit_layers=parse_list(self.parser["Model"]["critic layers"], int),
            num_classes=self.parser.getint("Data", "number of classes"),
            label_ratios=torch.Tensor(
                parse_list(self.parser["Data"]["label ratios"], float)
            ),
            device=self.parser.get("EXPERIMENT", "device", fallback=None),
            library_size=self.parser.getint("Preprocessing", "library size"),
        )

    def get_trainer(self) -> typing.Callable:
        gan = self.get_gan()
        return lambda: gan.train(
            train_files=self.parser.get("Data", "train"),
            valid_files=self.parser.get("Data", "validation"),
            critic_iter=self.parser.getint("Training", "critic iterations"),
            max_steps=self.parser.getint("Training", "maximum steps"),
            c_lambda=self.parser.getfloat("Model", "lambda"),
            beta1=self.parser.getfloat("Optimizer", "beta1"),
            beta2=self.parser.getfloat("Optimizer", "beta2"),
            gen_alpha_0=self.parser.getfloat("Learning Rate", "generator initial"),
            gen_alpha_final=self.parser.getfloat("Learning Rate", "generator final"),
            crit_alpha_0=self.parser.getfloat("Learning Rate", "critic initial"),
            crit_alpha_final=self.parser.getfloat("Learning Rate", "critic final"),
            checkpoint=self.parser.get("EXPERIMENT", "checkpoint", fallback=None),
            summary_freq=self.parser.getint("Logging", "summary frequency"),
            plt_freq=self.parser.getint("Logging", "plot frequency"),
            save_feq=self.parser.getint("Logging", "save frequency"),
            output_dir=self.parser.get("EXPERIMENT", "output directory"),
        )


class CausalGANFactory(IGANFactory):
    def get_cc(self) -> GAN:
        return GAN(
            genes_no=self.parser.getint("Data", "number of genes"),
            batch_size=self.parser.getint("CC Training", "batch size"),
            latent_dim=self.parser.getint("CC Model", "latent dim"),
            gen_layers=parse_list(self.parser["CC Model"]["generator layers"], int),
            crit_layers=parse_list(self.parser["CC Model"]["critic layers"], int),
            device=self.parser.get("EXPERIMENT", "device", fallback=None),
            library_size=self.parser.getint("Preprocessing", "library size"),
        )

    def get_gan(self) -> CausalGAN:
        with open(self.parser.get("Data", "causal graph"), "rb") as fp:
            causal_graph = pickle.load(fp)

        return CausalGAN(
            genes_no=self.parser.getint("Data", "number of genes"),
            batch_size=self.parser.getint("Training", "batch size"),
            latent_dim=self.parser.getint("Model", "latent dim"),
            noise_per_gene=self.parser.getint("Model", "noise per gene"),
            depth_per_gene=self.parser.getint("Model", "depth per gene"),
            width_per_gene=self.parser.getint("Model", "width per gene"),
            cc_latent_dim=self.parser.getint("CC Model", "latent dim"),
            cc_layers=parse_list(self.parser["CC Model"]["generator layers"], int),
            cc_pretrained_checkpoint=self.parser.get("EXPERIMENT", "output directory")
            + f"_CC/checkpoints/step_{self.parser.getint('CC Training', 'maximum steps')}.pth",
            crit_layers=parse_list(self.parser["Model"]["critic layers"], int),
            causal_graph=causal_graph,
            labeler_layers=parse_list(self.parser["Model"]["labeler layers"], int),
            device=self.parser.get("EXPERIMENT", "device", fallback=None),
            library_size=self.parser.getint("Preprocessing", "library size"),
        )

    def get_trainer(self) -> typing.Callable:
        cc = self.get_cc()

        # the following lambda will train the causal controller for maximum steps
        # specified in the CC Training section of the config file
        # after training the causal controller, the causal GAN will be instantiated
        # with the pretrained causal controller and training will start.
        return lambda: (
            cc.train(
                train_files=self.parser.get("Data", "train"),
                valid_files=self.parser.get("Data", "validation"),
                critic_iter=self.parser.getint("CC Training", "critic iterations"),
                max_steps=self.parser.getint("CC Training", "maximum steps"),
                c_lambda=self.parser.getfloat("CC Model", "lambda"),
                beta1=self.parser.getfloat("CC Optimizer", "beta1"),
                beta2=self.parser.getfloat("CC Optimizer", "beta2"),
                gen_alpha_0=self.parser.getfloat(
                    "CC Learning Rate", "generator initial"
                ),
                gen_alpha_final=self.parser.getfloat(
                    "CC Learning Rate", "generator final"
                ),
                crit_alpha_0=self.parser.getfloat("CC Learning Rate", "critic initial"),
                crit_alpha_final=self.parser.getfloat(
                    "CC Learning Rate", "critic final"
                ),
                checkpoint=self.parser.get("EXPERIMENT", "output directory")
                + f"_CC/checkpoints/step_{self.parser.getint('CC Training', 'maximum steps')}.pth",
                summary_freq=self.parser.getint("CC Logging", "summary frequency"),
                plt_freq=self.parser.getint("CC Logging", "plot frequency"),
                save_feq=self.parser.getint("CC Logging", "save frequency"),
                output_dir=self.parser.get("EXPERIMENT", "output directory") + "_CC",
            ),
            self.get_gan().train(
                train_files=self.parser.get("Data", "train"),
                valid_files=self.parser.get("Data", "validation"),
                critic_iter=self.parser.getint("Training", "critic iterations"),
                max_steps=self.parser.getint("Training", "maximum steps"),
                c_lambda=self.parser.getfloat("Model", "lambda"),
                beta1=self.parser.getfloat("Optimizer", "beta1"),
                beta2=self.parser.getfloat("Optimizer", "beta2"),
                gen_alpha_0=self.parser.getfloat("Learning Rate", "generator initial"),
                gen_alpha_final=self.parser.getfloat(
                    "Learning Rate", "generator final"
                ),
                crit_alpha_0=self.parser.getfloat("Learning Rate", "critic initial"),
                crit_alpha_final=self.parser.getfloat("Learning Rate", "critic final"),
                labeler_alpha=self.parser.getfloat("Learning Rate", "labeler"),
                antilabeler_alpha=self.parser.getfloat("Learning Rate", "antilabeler"),
                labeler_training_interval=self.parser.getfloat(
                    "Training", "labeler and antilabeler training intervals"
                ),
                checkpoint=self.parser.get("EXPERIMENT", "checkpoint", fallback=None),
                summary_freq=self.parser.getint("Logging", "summary frequency"),
                plt_freq=self.parser.getint("Logging", "plot frequency"),
                save_feq=self.parser.getint("Logging", "save frequency"),
                output_dir=self.parser.get("EXPERIMENT", "output directory"),
            ),
        )[0]


def get_factory(cfg: ConfigParser) -> IGANFactory:
    """
    Return the factory for the GAN type based on 'model' key in the parser.

    Parameters
    ----------
    cfg : ConfigParser
        Parser for config file containing GAN model and training params.

    Returns
    -------
    IGANFactory
        Factory for the specified GAN.

    Raises
    ------
    ValueError
        If the model is unknown or not implemented.
    """
    # read the desired GAN
    model = cfg.get("Model", "type")
    factories = {
        "GAN": GANFactory(cfg),
        "proj conditional GAN": ConditionalProjGANFactory(cfg),
        "cat conditional GAN": ConditionalCatGANFactory(cfg),
        "causal GAN": CausalGANFactory(cfg),
    }

    if model in factories:
        return factories[model]
    raise ValueError(f"model '{model}' type is invalid")
