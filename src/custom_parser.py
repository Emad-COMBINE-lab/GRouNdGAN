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
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config", required=True, help="Path to the configuration file"
    )

    parser.add_argument(
        "--train",
        required=False,
        default=False,
        action="store_true",
        help="Use preprocessed data for training the model",
    )

    parser.add_argument(
        "--generate",
        required=False,
        default=False,
        action="store_true",
        help="Generate in-silico cells",
    )

    parser.add_argument(
        "--preprocess",
        required=False,
        default=False,
        action="store_true",
        help="Preprocess raw data for GAN training",
    )

    parser.add_argument(
        "--create_grn",
        required=False,
        default=False,
        action="store_true",
        help="Infer a GRN from preprocessed data using GRNBoost2 and appropriately format it to input to GRouNdGAN as causal graph",
    )
    return parser
