Docker Setup 
~~~~~~~~~~~~
**Prerequisite:** Before you begin, make sure you have Docker installed on your machine. You can download and install Docker from the official website: `Get Started | Docker <https://www.docker.com/get-started/>`_

Option A: Using Pre-built Docker Image (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. Pull the Docker image from Docker Hub:
   
   .. code-block:: console
    
    $ docker pull yazdanz/groundgan:4b98686 
    
2. Run the Docker container and pass GPU devices::

   $ docker run --gpus all -it yazdanz/groundgan:4b98686 /bin/bash

* The ``--gpus all`` flag enables GPU support within the container. Omit if you intend to use CPU only.
* The ``--it`` flag allows an interactive terminal session.

You're now inside the Docker container with CUDA support, ready to use GRouNdGAN!

Option B: Building Docker Image from Dockerfile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. Clone the GRouNdGAN repository::

   $ git clone https://github.com/Emad-COMBINE-lab/GRouNdGAN.git

2. Navigate to the project directory::

   $ cd GRouNdGAN

3. Build the Docker image using the provided Dockerfile::
   
   $ docker build -t yourusername/groundgan:custom -f docker/Dockerfile .

   This command will build a Docker image with the tag ``yourusername/groundgan:custom``.

   .. note::
        
      Building the image using this method may take approximately 15-30 minutes, depending on your system's performance.

4. Run the Docker container and pass GPU devices::

   $ docker run -itd --name yourusername/groundgan:custom --gpus all groundgan /bin/bash

Verifying GPU Acceleration
^^^^^^^^^^^^^^^^^^^^^^^^^^
Inside the Docker container, you can verify if the GPU is recognized using::

$ nvidia-smi

You should see detailed information about your GPU, including its name, memory usage, etc. This confirms that GPU acceleration is enabled inside the container.

.. code-block:: bash

    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 470.63.01    Driver Version: 470.63.01    CUDA Version: 11.4     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Tesla V100-SXM2...  Off  | 00000000:00:1E.0 Off |                    0 |
    | N/A   33C    P0    41W / 300W |      0MiB / 32480MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+

    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+
