# GRouNdGAN
_**GR**N-guided in silic**o** sim**u**lation of single-cell R**N**A-seq **d**ata using Causal **G**enerative **A**dversarial **N**etworks_
![architecture figure](https://github.com/YazdanZ/GRouNdGAN/blob/master/docs/figs/architecture.svg)
## Dependencies
This repository has been tested with Python 3.9.6  A PIP environment is provided in [requirements.txt](https://github.com/YazdanZ/GRouNdGAN/blob/master/requirements.txt)

## How to run GRouNdGAN
### Preprocessing 
GRouNDGAN expects four files when processing a new dataset. You should put them in the same directory. We recomment putting them in  `data/raw/`.
* `raw/barcodes.tsv`
* `raw/genes.tsv`
* `raw/matrix.mtx`
  
Optionally, you can provide annotations through the 
* `barcodes_annotations.csv`

Then call the preprocessing script `src/preprocessing/preprocess.py` and provide the directory containing raw data and an output directory as positional arguments. If you have annotations, provide them through the option `--annotations`.

Example:
```
python src/preprocessing/preprocess.py data/raw/PBMC/ data/processed/PBMC/PBMC68k.h5ad --annotations data/raw/PBMC/barcodes_annotations.tsv

```
### Training 
GRouNdGAN can be trained by running  `main.py` with the `--train` argument and providing a config file detailing training parameters. A template detailing every argument can be found [here](https://github.com/YazdanZ/GRouNdGAN/blob/master/configs/causal_gan.cfg). This repository also implements scGAN [config file](https://github.com/YazdanZ/GRouNdGAN/blob/master/configs/gan.cfg), cscGAN with projection conditioning [config file](https://github.com/YazdanZ/GRouNdGAN/blob/master/configs/conditional_gan.cfg) and A wasserstein gan with conditioning by concatenation [config file](https://github.com/YazdanZ/GRouNdGAN/blob/master/configs/conditional_gan.cfg). 

```
python src/main.py --config {path/to/config_file} --train
```

## License 
Copyright (C) 2023 Emad's COMBINE Lab: Yazdan Zinati, Abdulrahman Takiddeen, and Amin Emad. 

GRouNdGAN is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

GRouNdGAN is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with GRouNdGAN. If not, see <https://www.gnu.org/licenses/>.
