#!/bin/bash
#SBATCH -J CyBERT-SuperGLUE-Our-True
#SBATCH -a 1-5%5
#SBATCH --mail-type=END
#SBATCH -e /work/projects/project02060/CyBERT/logs_training/%x.err.SuperGLUE_cybert.%A_%a
#SBATCH -o /work/projects/project02060/CyBERT/logs_training/%x.out.SuperGLUE_cybert.%A_%a
#SBATCH --mem-per-cpu=16000
#SBATCH -t 72:30:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:v100:1

#---------------------------------------------------------------

module purge
module load gcc
module load python/3.7.4
module load cuda
module load cuDNN

cd /work/projects/project02060/CyBERT/

srun python cybert/code/SuperGLUE/superglue_baseline.py -r $SLURM_ARRAY_TASK_ID

