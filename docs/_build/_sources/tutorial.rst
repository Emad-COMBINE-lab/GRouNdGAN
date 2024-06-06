Tutorial 
========

CLI
---

GRouNdGAN comes with a command-line interface. This section outlines available commands and arguments.

To use the CLI, run the ``src/main.py`` script with the desired command and any applicable options.

.. important::

    Use :code:`python3.9` instead of :code:`python` if you're running through docker or singularity.
    
.. code-block:: bash 

    $ python src/main.py --help
    usage: GRouNdGAN [-h] --config CONFIG [--preprocess] [--create_grn] [--train] [--generate]

    GRouNdGAN is a gene regulatory network (GRN)-guided causal implicit generative model for
    simulating single-cell RNA-seq data, in-silico perturbation experiments, and benchmarking GRN
    inference methods. This programs also contains cWGAN and unofficial implementations of scGAN and
    cscGAN (with projection conditioning)

    required arguments:
    --config CONFIG  Path to the configuration file

    optional arguments:
    --preprocess     Preprocess raw data for GAN training
    --create_grn     Infer a GRN from preprocessed data using GRNBoost2
                     and appropriately format as causal graph
    --train          Start or resume model training
    --generate       Simulate single-cells RNA-seq data in-silico
    
There are essentially four commands available: ``--preprocess``, ``--create_grn``, ``--train``, and ``--generate``. You must provide a config file containing inputs and hyperparameters with each command through the ``--config`` flag. 

.. note:: 

    You can run commands individually:

    .. code-block:: console

        python src/main.py --config configs/causal_gan.cfg --preprocess

    Or chain them together to run all or multiple steps in one go:

    .. code-block:: console

        python src/main.py --config configs/causal_gan.cfg --preprocess --create_grn --train --generate

Config Files
------------

GRouNdGAN uses a configuration syntax similar to INI implemented by python's `configparser <https://docs.python.org/3/library/configparser.html#module-configparser>`_ module. 

We provide three sample config files in the ``configs/`` directory: 

* ``causal_gan.cfg``: for GRouNdGAN
* ``conditional_gan.cfg``: for cscGAN with projection conditioning (Marouf et al., 2020) and cWGAN. 
* ``gan.cfg``:  for scGAN (Marouf et al., 2020) (we use this to train GRouNdGAN's causal controller)

Most of the configuration file consists of hyperparameters. You only need to modify input and output parameters which we will go through in each section. GRouNdGAN isn't very sensitive to hyperparameters. However, it is still advisable to test different choices of hyperparameters using a validation set. 

Below is the demo ``causal_gan.cfg`` config file for training GRouNdGAN using the PBMC68k dataset:

.. include:: causal_gan_cfg.rst


Project outline
---------------
GRouNdGAN is structured as follows:

.. include:: tree.rst

Demo Datasets
-------------

The provided docker image comes prepackaged with the unprocessed Mouse BoneMarrow (Paul et al., 2015) and Human PBMC68k (Zheng et al., 2017) datasets (``data/raw/PBMC`` and ``data/raw/BoneMarrow``) and human and mouse TFs, downloaded from AnimalTFDB (``data/raw/Homo_sapiens_TF.csv`` and ``data/raw/Mus_musculus_TF.csv``).  

.. note:: 

    If you have opted for a local installation, you can download these files from `here <https://nextcloud.computecanada.ca/index.php/s/pXKQ2isr47AwKEX>`_ and place them in ``data/raw/``.

    If that's too hard, this will do it in bash (you need curl and tar installed): 

    .. code-block:: bash

        curl https://nextcloud.computecanada.ca/index.php/s/WqrCqkH5zjYYMw9/download --output demo_data.tar &&
        tar -xvf demo_data.tar -C data/raw/ &&
        mv data/raw/demo/* data/raw &&
        rm demo_data.tar &&
        rm -rf data/raw/demo/

Steps 
-----

Preprocessing 
~~~~~~~~~~~~~
.. attention:: 
    Don't skip the preprocessing step, GRouNdGAN requires library-size normalized data as input. 

To run our preprocessing pipeline, your config file should contain the following arguments: 

.. code-block:: ini

    [EXPERIMENT]

        [Preprocessing]
        ; set True if data is 10x (like PBMC)
        ; set False if you're providing an .h5ad file (like BoneMarrow.h5ad)
        10x = True

        ; If 10x = True, path to the directory containing matrix.mtx, genes.tsv, and barcodes.tsv
        ; If 10x = False, path to the .h5ad file containing the expression matrix
        raw = data/raw/PBMC/ 

        validation set size = 1000 ; size of the validation set to create
        test set size = 1000 ; size of the test set to create 
        annotations = data/raw/PBMC/barcodes_annotations.tsv ; optional, leave empty if you don't have annotations
        min cells = 3 ; genes expressed in less than 3 cells are discarded
        min genes = 10 ; cells with less than 10 genes expressed are discarded
        library size = 20000 ; library size used for library-size normalization
        louvain res = 0.15 ; Louvain clustering resolution (higher resolution means finding more and smaller clusters)
        highly variable number = 1000 ; number of highly variable genes to identify
    
        [Data]
        train = data/processed/PBMC/PBMC68k_train.h5ad ; path to output the train set
        validation = data/processed/PBMC/PBMC68k_validation.h5ad ; path to output the validation set
        test = data/processed/PBMC/PBMC68k_test.h5ad ; path to output the test set

Then, run the following::

   $ python src/main.py --config configs/causal_gan.cfg --preprocess

Once completed, you will see a success message. Train, validation, and test sets should be created in the paths defined under the ``[Data]`` section of the config file.

GRN Creation 
~~~~~~~~~~~~

.. note:: 
    GRN creation isn't needed for scGAN, cscGAN, and cWGAN; you can skip the ``--create_grn`` command. 

This command uses GRNBoost2 (Moerman et al., 2018) to infer a GRN on the preprocessed train set. It then converts it into the a format that GRouNdGAN accepts.  

In addition to what was required in the previous step, you need to provide the following arguments:

.. code-block:: ini

    [GRN Preparation]
    TFs = data/raw/Homo_sapiens_TF.csv ; Path to file containing TFs (accepts AnimalTFDB csv formats)
    k = 15 ; k is the number of top most important TFs per gene to include in the GRN 
    Inferred GRN = data/processed/PBMC/inferred_grnboost2.csv ; where to write GRNBoost2's output

    [Data]
    causal graph = data/processed/PBMC/causal_graph.pkl ; where to write the created GRN
    
Run using::

   $ python src/main.py --config configs/causal_gan.cfg --create_grn

Once done, you will see the properties of the created GRN.

.. code-block:: bash

    Using 63 TFs for GRN inference.
    preparing dask client
    parsing input
    creating dask graph
    4 partitions
    computing dask graph
    shutting down client and local cluster
    finished

    Causal Graph
    -----------------  ------------
    TFs                          63
    Targets                     937
    Genes                      1000
    Possible Edges            59031
    Imposed Edges             14055
    GRN density Edges      0.238095
    -----------------  ------------

The causal graph will be written to the path specified by ``[Data]/causal graph`` in the config file.

Imposing Custom GRNs 
^^^^^^^^^^^^^^^^^^^^

It is possible to instead impose your own GRN onto GRouNdGAN. If you're opting for this option, skip the ``--create_grn`` command. Instead, create a python dictionary where keys are gene indices (``int``). For each key (gene index), the value is the set of indices ``set[int]`` coresponding to TFs that regulate the gene. 

.. image:: _static/sampleGRN.svg

The GRN in the picture above can be written in dictionary form as: 

.. code-block:: python

    causal_graph = {
        "G2": {"TF2", "TFn"}, 
        "G1": {"TF1"}, 
        "Gn": {"TF2", "TF1"},
        "G3": {"TFn", "TF1"}
    }

Converting the key/value pairs into gene/TF indices, it becomes

.. code-block:: python

    causal_graph = {
        1: {4, 5}, 
        3: {0}, 
        6: {4, 0}, 
        2: {5, 0}
    }

Then, pickle the dictionary:

.. code-block:: python

    import pickle

    with open("path/to/write/causal_graph.pkl", "wb") as fp:
        pickle.dump(causal_graph, fp, protocol=pickle.HIGHEST_PROTOCOL)

Don't forget to edit the causal graph path in the config file. 

.. code-block:: ini

    [Data]
    causal graph = path/to/write/causal_graph.pkl


* The GRN must be a directed bipartite graph
* All genes and TFs in the dataset must appear in the dictionary either as key (target gene) or value (as part of the set of TFs)  

.. Warning:: 

    Construct a biologically meaningful GRN!

    Imposing a GRN with significantly different TF-gene relationships from those observable in the reference dataset will deteriorate the quality of simulated cells as generating realistic simulated datapoints and imposing the GRN will act as contradictory tasks

Training 
~~~~~~~~

You can start training the model using the following command::

    $ python src/main.py --config configs/causal_gan.cfg --train

Upon running the command above, three folders will be created inside the path provided in the config file (``[EXPERIMENT]/output directory``) and the config file will be copied over:

* ``checkpoints/``: Containing the ``.pth`` state dictionary including model's weights, biases, etc.
* ``TensorBoard/``: Containing TensorBoard logs
* ``TSNE/``: Containing t-SNE plots of real vs simulated cells 

You can change the save, logging, and plotting frequency (default every 10000 steps) in the config file. 

Monitor training using TensorBoard::

    tensorboard --logdir="{GAN OUTPUT DIR HERE}/TensorBoard" --host 0.0.0.0 --load_fast false &

We also provide two slurm submission scripts for training and monitoring in  ``scripts/``.

.. note::
    * Training time primarily depends on the number of genes and the density of the imposed GRN. It takes about five days with a very dense GRN (~20% density) containing 1000 genes on a single NVidia V100SXM2 (16G memory) GPU. 

    * GRouNdGAN supports multi-GPU training, but we suggest sticking to a single GPU to avoid excess overhead. 

    * GRouNdGAN trains for a million steps by default. It is not recommended to change this in the config file. 

    * You can resume training from a checkpoint by setting ``[EXPERIMENT]/checkpoint`` in the config file to the ``.pth`` checkpoint you wish to use. 

In-silico Single-Cell Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One training is done, populate the ``[EXPERIMENT]/checkpoint`` field with the path of the ``.pth`` checkpoint you want to use in the config file (usually the latest). 

You can change the number of cells to simulate in the config file (10000 by default)

.. code-block:: ini
    
    [Generation]
    number of cells to generate = 10000

Then run

.. code-block:: sh

    $ python src/main.py --config path/to/config_file --generate

This will output a ``simulated.h5ad`` file to ``[EXPERIMENT]/output directory`` containing the simulated expression matrix. 


References
----------

Marouf, M., Machart, P., Bansal, V., Kilian, C., Magruder, D. S., Krebs, C., & Bonn, S. (2020). Realistic in silico generation and augmentation of single-cell RNA-seq data using generative adversarial networks. Nature Communications, 11(1). https://doi.org/10.1038/s41467-019-14018-z

Paul, F., Arkin, Y., Giladi, A., Jaitin, D. A., Kenigsberg, E., Keren-Shaul, H., Winter, D. R., Lara-Astiaso, D., Gury, M., Weiner, A., David, E., Cohen, N., Lauridsen, F. K. B., Haas, S., Schlitzer, A., Mildner, A., Ginhoux, F., Jung, S., Trumpp, A., . . . Tanay, A. (2015). Transcriptional heterogeneity and lineage commitment in myeloid progenitors. Cell, 163(7), 1663–1677. https://doi.org/10.1016/j.cell.2015.11.013

Zheng, G., Terry, J. M., Belgrader, P., Ryvkin, P., Bent, Z., Wilson, R. J., Ziraldo, S. B., Wheeler, T. D., McDermott, G. P., Zhu, J., Gregory, M., Shuga, J., Montesclaros, L., Underwood, J. G., Masquelier, D. A., Nishimura, S. Y., Schnall-Levin, M., Wyatt, P., Hindson, C. M., . . . Bielas, J. H. (2017). Massively parallel digital transcriptional profiling of single cells. Nature Communications, 8(1). https://doi.org/10.1038/ncomms14049

Moerman, T., Aibar, S., González-Blas, C. B., Simm, J., Moreau, Y., Aerts, J., & Aerts, S. (2018). GRNBoost2 and Arboreto: efficient and scalable inference of gene regulatory networks. Bioinformatics, 35(12), 2159–2161. https://doi.org/10.1093/bioinformatics/bty916
