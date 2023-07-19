#!/bin/bash

#SBATCH --job-name=tensorbaord
#SBATCH --time=1-00:00:00
#SBATCH --account={your-PI}
#SBATCH --mail-user={your-email}
#SBATCH --mail-type=ALL
#SBATCH --mem 4G
#SBATCH --cpus-per-task=1

source {VIRTUAL ENV ACTIVATE SCRIPT HERE}
cd {PROJECT DIR HERE}
tensorboard --logdir="{GAN OUTPUT DIR HERE}/TensorBoard" --host 0.0.0.0 --load_fast false &
sleep infinity

# Run the following locally to forward TensorBoard port from compute canada
# ssh -N -f -L localhost:6006:computenode:6006 userid@cluster.computecanada.ca
# Then go to http://localhost:6006
# Note: It's possible that TensorBoard is running on ports other than 6006 (6007, 6008, ...) on computecanada.
