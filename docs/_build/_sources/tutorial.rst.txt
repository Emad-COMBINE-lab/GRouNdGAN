Tutorial 
========

Getting Started 
---------------
Start by cloning GRouNdGAN's official Github `repo <https://github.com/Emad-COMBINE-lab/GRouNdGAN>`_ to your local machine:

.. code-block:: sh

    git clone https://github.com/Emad-COMBINE-lab/GRouNdGAN.git
    cd GRouNdGAN/

.. note::
    You can optionally clone the BEELINE and scGAN submodules to reproduce our studies.
    
    .. code-block:: sh
    
        git clone --recurse-submodules https://github.com/Emad-COMBINE-lab/GRouNdGAN.git
        cd GRouNdGAN/


Dependencies 
------------
GRouNdGAN is developed and testes with Python version 3.9.6. After loading the appropriate python version, install required pip dependencies from the ``requirements.txt`` file: 

    .. code-block:: sh
    
        pip install -r requirements.txt

We recommend doing this step in a python virtual environment. 

How to run GRouNdGAN
--------------------

Preprocessing 
~~~~~~~~~~~~~
GRouNDGAN expects three files when processing a new dataset. You should put them in the same directory. We recomment putting them in ``data/raw/``.

* ``raw/barcodes.tsv``
* ``raw/genes.tsv``
* ``raw/matrix.mtx``

Then call the preprocessing script ``src/preprocessing/preprocess.py`` and provide the directory containing raw data and an output directory as positional arguments. If you have annotations, provide them through the ``--annotations`` option.

Example: 

.. code-block:: sh

    python src/preprocessing/preprocess.py data/raw/PBMC/ data/processed/PBMC/PBMC68k.h5ad --annotations data/raw/PBMC/barcodes_annotations.tsv

Config Files
~~~~~~~~~~~~
GRouNdGAN uses `python INI config files <https://docs.python.org/3/library/configparser.html>`_ to receive its inputs and allow users the flexibility to tune various parameters. Template config files are provided to run GRouNdGAN, scGAN, cscGAN, and cWGAN under the ``configs/`` directory. 

Below is an example cfg file for GRouNdGAN.

.. code:: INI
        
    [EXPERIMENT]
    output directory = results/GRouNdGAN
    device = cuda ; we will let the program choose what is available
    checkpoint  ; set value to use a trained model

        [Data]
        train = data/processed/train_data.h5ad
        validation = data/processed/validation_data.h5ad
        test = data/processed/test_data.h5ad
        number of genes = 1000

        ; this causal graph is a pickled nested dictionary
        ; nested dictionary keys are gene indices
        ; the dictionary is of this form:
        ; {381: {51, 65, 353, 664, 699},
        ; 16: {21, 65, 353, 605, 699},
        ; ...
        ; 565: {18, 51, 65, 552, 650}}
        ; In this example, 381, 16, and 565 are gene indices in the input dataset
        ; Each key's (gene's) value is the indiced of its regulating TFs in the input dataset
        ; A tutorial will be made available in the future.
        
        causal graph = data/processed/PBMC/causal_graph.pkl

        [Model]
        type = causal GAN
        noise per gene = 1
        depth per gene = 3
        width per gene = 2
        critic layers = 1024 512 256
        labeler layers = 2000 2000 2000
        latent dim = 128 ; noise vector dimensions
        library size = 20000 ; UMI count 
        lambda = 10 ; regularization hyper-parameter for gradient penalty


        [Training]
        batch size = 1024 
        critic iterations = 5 ; iterations to train the critic for each iteration of the generator
        maximum steps = 1000000
        labeler and antilabeler training intervals = 1

            [Optimizer]
            ; coefficients used for computing running averages of gradient and its square 
            beta1 = 0.5
            beta2 = 0.9

            [Learning Rate]
            generator initial = 0.001
            generator final = 0.0001
            critic initial = 0.001
            critic final = 0.001
            labeler = 0.0001
            antilabeler = 0.0001


            [Logging]
            summary frequency = 10000
            plot frequency = 10000
            save frequency = 100000

        [CC Model]
        type = GAN ; Non-conditional single-cell RNA-seq GAN
        generator layers = 256 512 1024
        critic layers = 1024 512 256
        latent dim = 128 ; noise vector dimensions
        library size = 20000 ; UMI count (hardcoded to None in the code)
        lambda = 10 ; regularization hyper-parameter for gradient penalty


        [CC Training]
        batch size = 128 
        critic iterations = 5 ; iterations to train the critic for each iteration of the generator
        maximum steps = 200000

            [CC Optimizer]
            ; coefficients used for computing running averages of gradient and its square 
            beta1 = 0.5
            beta2 = 0.9

            [CC Learning Rate]
            generator initial = 0.0001
            generator final = 0.00001
            critic initial = 0.0001
            critic final = 0.00001

            [CC Logging]
            summary frequency = 10000
            plot frequency = 10000
            save frequency = 100000


Training 
~~~~~~~~
GRouNdGAN can be trained by running ``main.py`` with the ``--train`` argument and providing a config file detailing training parameters. A template detailing every argument can be found here. This repository also implements  `scGAN <https://github.com/Emad-COMBINE-lab/GRouNdGAN/blob/master/configs/gan.cfg>`_, `_cscGAN with projection conditioning <https://github.com/Emad-COMBINE-lab/GRouNdGAN/blob/master/configs/conditional_gan.cfg>`, and a `Wasserstein gan with conditioning by concatenation <https://github.com/Emad-COMBINE-lab/GRouNdGAN/blob/master/configs/conditional_gan.cfg>`_.

.. code-block:: sh

    python src/main.py --config path/to/config_file --train



In-silico Single-Cell Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
One training is done, you can simulate cells by running the following command:

.. code-block:: sh

    python src/main.py --config path/to/config_file --generate

.. note::
    You have to first populate the ``checkpoint`` field with the path to the saved model (with ``.pth`` extension).

