FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Set environment variable to prevent timezone configuration prompt
ENV DEBIAN_FRONTEND=noninteractive

# Install essential dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    libffi-dev \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libsqlite3-dev \
    libpng-dev \
    libjpeg-dev \
    libbz2-dev \ 
    liblzma-dev \
    vim 

# Download and install Python 3.9.6
RUN wget https://www.python.org/ftp/python/3.9.6/Python-3.9.6.tar.xz
RUN tar -xf Python-3.9.6.tar.xz
RUN cd Python-3.9.6/ && ./configure --enable-optimizations && make && make install


WORKDIR /
COPY .git groundgan.git
RUN git clone groundgan.git
WORKDIR /groundgan

RUN curl https://nextcloud.computecanada.ca/index.php/s/WqrCqkH5zjYYMw9/download --output demo_data.tar
RUN tar -xvf demo_data.tar -C data/raw/
RUN rm demo_data.tar
RUN mv data/raw/demo/* data/raw
RUN rm -rf data/raw/demo/

# just to double check. Should be v3.9.6.
RUN python3.9 --version 

# Install dependencies from requirements.txt
RUN python3.9 -m pip install -r requirements.txt


# Run the main Python script
CMD ["python3.9", "src/main.py", "--help"]