#!/bin/bash
#SBATCH -J CyBERT-MSExchange
#SBATCH -a 1-5%5
#SBATCH --mail-type=END
#SBATCH -e /work/projects/project01762/CyBERT/logs_training/%x.err.MSExchange.%A_%a
#SBATCH -o /work/projects/project01762/CyBERT/logs_training/%x.out.MSExchange.%A_%a
#SBATCH --mem-per-cpu=8000
#SBATCH -t 0:30:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1

#---------------------------------------------------------------

module purge
module load gcc
module load python
module load cuda
module load cuDNN

cd /work/projects/project01762/CyBERT/

srun python cybert/code/MSExchange/run_msexchange_Ranada.py
