#!/bin/sh
#SBATCH --job-name=SSD
#SBATCH -N 1
#SBATCH -n 14    ##14 cores(of28) so you get 1/2 of machine RAM (64 GB of 128GB)
#SBATCH --gres=gpu:1   ## Run on 2 GPU
#SBATCH --output job%j.out
#SBATCH --error job%j.err
#SBATCH -p v100-16gb-hiprio


# set to use first visible GPU in the machine
#export CUDA_VISIBLE_DEVICES=0
##export CUDA_VISIBLE_DEVICES=0,1  # if both GPUs

##Load your modules and run code here


##module load python3/anaconda/2020.02
module load cuda/10.0
module load gcc/6.4.0

##source activate /home/xl22/.conda/envs/python37

cd /work/xl22/code/structure_flow/StructureFlow/resample2d_package

python setup.py install
