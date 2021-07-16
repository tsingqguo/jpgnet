#!/bin/sh
#SBATCH --job-name=SSD
#SBATCH -N 1
#SBATCH -n 14    ##14 cores(of28) so you get 1/2 of machine RAM (64 GB of 128GB)
#SBATCH --gres=gpu:1   ## Run on 2 GPU
#SBATCH --output job%j.out
#SBATCH --error job%j.err
#SBATCH -p v100-16gb-hiprio
##SBATCH -p defq-48co


# set to use first visible GPU in the machine
#export CUDA_VISIBLE_DEVICES=0
##export CUDA_VISIBLE_DEVICES=0,1  # if both GPUs

##Load your modules and run code here


cd /work/xl22/code/stacked_hourglass/stacked_hourglass_Alibaba_any

python -u prediction.py
