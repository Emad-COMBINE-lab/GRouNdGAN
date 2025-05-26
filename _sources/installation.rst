Installation
------------
There are multiple ways to install GRouNdGAN, depending on your preferences and requirements. Choose the installation method that best suits your needs:

**1.** **Docker Setup**:
   If you want a quick and hassle-free installation with GPU support, you can use Docker. This method is recommended if you prefer a pre-configured environment.

   - **Option A:** `Using Pre-built Docker Image (Recommended) <#option-a-using-pre-built-docker-image-recommended>`_:
     We provide a pre-built Docker image for GRouNdGAN that is already configured with CUDA support for GPU acceleration. This option is the most convenient and straightforward way to get started.

   - **Option B:** `Building Docker Image from Dockerfile <#option-b-building-docker-image-from-dockerfile>`_: 
     Building the Docker image from the Dockerfile allows you to further finetune our provided environment to your specific requirements. It also provides the advantage of knowing exactly which dependencies and configurations are being used in your experiments. This option is ideal if you want fine-grained control or if you need to make modifications to GRouNdGAN's default setup. 

**2.** `Local Installation <#id1>`_: 
   If you prefer greater control over your environment, you can opt for a local installation of GRouNdGAN. This option is particularly recommended if you plan to use GRouNdGAN as a foundation for new projects.

**3.** `Singularity <#singularity-setup>`_:
   Most HPC clusters restrict the use of docker, as it can be used to gain root access to the host system. Singularity is a secure and compatible alternative for containerization.

.. include:: docker.rst
.. include:: local_installation.rst
.. include:: singularity.rst
