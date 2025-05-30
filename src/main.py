#!/usr/bin/python
from __future__ import absolute_import

import os
import shutil

import numpy as np
import scanpy as sc

from custom_parser import get_argparser, get_configparser
from factory import get_factory
from preprocessing import grn_creation, preprocess
from evaluation import data_quality, grn_inference
from perturbation import perturbation

if __name__ == "__main__":
    """
    Main script to process the data and/or start the training or
    generate cells from an existing model.
    """
    argparser = get_argparser()
    args = argparser.parse_args()

    cfg_parser = get_configparser()
    cfg_parser.read(args.config)

    # copy the config file to the output dir
    output_dir = cfg_parser.get("EXPERIMENT", "output directory")
    os.makedirs(output_dir, exist_ok=True)

    try:
        shutil.copy(args.config, output_dir)
    except shutil.SameFileError:
        pass

    # get the GAN factory
    fac = get_factory(cfg_parser)

    if args.preprocess:
        preprocess.preprocess(cfg_parser)

    if args.create_grn:
        grn_creation.create_GRN(cfg_parser)

    if args.train:
        fac.get_trainer()()
        print("Finished training")

    if args.generate:
        simulated_cells = fac.get_gan().generate_cells(
            int(cfg_parser.get("Generation", "number of cells to generate")),
            checkpoint=cfg_parser.get("EXPERIMENT", "checkpoint"),
        )

        simulated_cells = sc.AnnData(simulated_cells)
        simulated_cells.obs_names = np.repeat("fake", simulated_cells.shape[0])
        simulated_cells.obs_names_make_unique()

        # Get generation path if defined, otherwise fallback
        generation_path = cfg_parser.get("Generation", "generation path", fallback="")
        if not generation_path:
            generation_path = (
                cfg_parser.get("EXPERIMENT", "output directory") + "/simulated.h5ad"
            )

        simulated_cells.write(generation_path)
        print("Simulated cells saved to", generation_path)


    if args.evaluate:
        data_quality.evaluate(cfg_parser)

    if args.benchmark_grn:
        grn_inference.evaluate(cfg_parser)

    if args.perturb: 
        perturbation.perturb(cfg_parser)