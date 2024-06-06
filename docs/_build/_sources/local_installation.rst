Local Installation
~~~~~~~~~~~~~~~~~~

**Prerequisites:** Before setting up GRouNdGAN locally, ensure you have Python 3.9.6 installed. If you do not have Python 3.9.6, you can use `pyenv` to manage multiple Python versions. Detailed installation instructions for various platforms can be found in the `pyenv` documentation: https://github.com/pyenv/pyenv#installation.  In the case you are using pyenv, please use `pyenv-virtualenv <https://github.com/pyenv/pyenv-virtualenv>`_ to manage and activate your virtual environment. 

1. Clone the GRouNdGAN repository to a directory of your choice::

   $ git clone https://github.com/Emad-COMBINE-lab/GRouNdGAN.git
   
   .. tip::
       You can optionally clone the scGAN, BEELINE, scDESIGN2, and SPARSim submodules to also get the specific version of repositories that we used in our study. 
        
       .. code-block:: sh
        
           git clone --recurse-submodules https://github.com/Emad-COMBINE-lab/GRouNdGAN.git
           
2. Navigate to the project directory::

   $ cd GRouNdGAN

3. Create a virtual environment for your project::

   $ python -m venv venv

In the case you are using pyenv, please use `pyenv-virtualenv <https://github.com/pyenv/pyenv-virtualenv>`_ to manage and activate your virtual environment. 
   
4. Activate the virtual environment

   - **Linux/macOS**::

     $ source venv/bin/activate

   - **Windows**::

     $ venv\Scripts\activate

5. Install the required dependencies from the ``requirements.txt`` file::

   (venv)$ pip install -r requirements.txt

   If you're a fellow Canadian using ComputeCanada, consider using ``requirements_computecanada.txt`` instead.

You're now ready to use GRouNdGAN locally!

.. admonition:: Troubleshooting

   **Known Installation Issues**

   If you’re encountering issues with installation to ubuntu, it might be because you are missing one or more of the following packages:

   .. code-block:: sh

      sudo apt install build-essential libffi-dev     zlib1g-dev     libncurses5-dev     libgdbm-dev     libnss3-dev     libssl-dev     libreadline-dev     libsqlite3-dev     libpng-dev     libjpeg-dev     libbz2-dev  liblzma-dev tk-dev