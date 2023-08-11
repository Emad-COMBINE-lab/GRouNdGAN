Singularity Setup
~~~~~~~~~~~~~~~~~

Converting Docker Image to Singularity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. Install Singularity on your system if it's not already installed (`Installation Guide <https://docs.sylabs.io/guides/3.0/user-guide/installation.html/>`_).

2. Use the ``singularity pull`` command to convert the Docker image to a Singularity image:

.. code-block:: console

   $ singularity pull groundgan.sif docker://yazdanz/groundgan:4b98686

This command will create a Singularity image named ``groundgan.sif`` by pulling ``yazdanz/groundgan:4b98686`` from Docker Hub.

Running a Singularity Container
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After converting the Docker image to a Singularity image, you can run a Singularity container interactively:

1. Start an interactive shell session within the Singularity container:

.. code-block:: console

   $ singularity shell --nv groundgan.sif

* The ``--nv`` flag enables running CUDA application inside the container.

.. warning::
    There might be differences in directory structures and permissions between Singularity and Docker containers due to Singularity's bind-mounted approach.
