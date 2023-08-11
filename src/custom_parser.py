import argparse
from configparser import ConfigParser


def get_configparser() -> ConfigParser:
    """
    Configure and read config file .cfg .ini parser.

    Returns
    -------
    ConfigParser.
    """
    return ConfigParser(
        empty_lines_in_values=False,
        allow_no_value=True,
        inline_comment_prefixes=";",
    )


def get_argparser() -> argparse.ArgumentParser:
    """
    Initialize argument parser and add program args.

    Returns
    -------
    argparse.ArgumentParser
        Argument CLI parser.
    """
    parser = argparse.ArgumentParser(
        prog="GRouNdGAN",
        description="GRouNdGAN is a gene regulatory network (GRN)-guided causal implicit generative model for simulating single-cell RNA-seq data, in-silico perturbation experiments, and benchmarking GRN inference methods. \
            This programs also contains cWGAN and unofficial implementations of scGAN and cscGAN (with projection conditioning)",
    )

    parser._action_groups.pop()
    required = parser.add_argument_group("required arguments")
    optional = parser.add_argument_group("optional arguments")

    required.add_argument(
        "--config", required=True, help="Path to the configuration file"
    )

    optional.add_argument(
        "--preprocess",
        required=False,
        default=False,
        action="store_true",
        help="Preprocess raw data for GAN training",
    )

    optional.add_argument(
        "--create_grn",
        required=False,
        default=False,
        action="store_true",
        help="Infer a GRN from preprocessed data using GRNBoost2 and appropriately format as causal graph",
    )

    optional.add_argument(
        "--train",
        required=False,
        default=False,
        action="store_true",
        help="Start or resume model training",
    )

    optional.add_argument(
        "--generate",
        required=False,
        default=False,
        action="store_true",
        help="Simulate single-cells RNA-seq data in-silico",
    )

    return parser
