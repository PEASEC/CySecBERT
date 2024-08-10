#!/bin/bash
#SBATCH -J CyBERT-Training
#SBATCH --mail-type=END
#SBATCH -e /work/projects/project01762/CyBERT/logs_training/%x.err.%j
#SBATCH -o /work/projects/project01762/CyBERT/logs_training/%x.out.%j
#SBATCH --mem-per-cpu=800
#SBATCH -t 00:01:00
#SBATCH -n 1
#---------------------------------------------------------------

module load gcc
module load python
module load cuda
module load cuDNN

conda activate cybert
