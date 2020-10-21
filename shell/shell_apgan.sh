#! /bin/bash
#SBATCH "--job-name=apgan"
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --output files/output/job%J.out
#SBATCH --error files/output/job%J.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:1

module load cuda/10.0

python3 ./apgan_train_main.py
