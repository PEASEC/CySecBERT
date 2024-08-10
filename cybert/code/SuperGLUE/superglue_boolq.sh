#!/bin/bash
#SBATCH -J CyBERT-SuperGLUE-boolq
#SBATCH --mail-type=END
#SBATCH -e /work/projects/project02060/CyBERT/logs_training/%x.err.SuperGLUE_cybert.%A_%a
#SBATCH -o /work/projects/project02060/CyBERT/logs_training/%x.out.SuperGLUE_cybert.%A_%a
#SBATCH --mem-per-cpu=16000
#SBATCH -t 00:30:00
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

cd /work/projects/project02060/CyBERT/

srun python cybert/code/SuperGLUE/superglue_boolq.py -r 1 -m cybert_v100_dataset_15
