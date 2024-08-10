#!/bin/bash
#SBATCH -J CyBERT-Training
#SBATCH --mail-type=END
#SBATCH -e /work/projects/project02060/CyBERT/logs_training/%x.err.%j
#SBATCH -o /work/projects/project02060/CyBERT/logs_training/%x.out.%j
#SBATCH --mem-per-cpu=8000
#SBATCH -t 150:00:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 4
#SBATCH --gres=gpu:v100:4

#---------------------------------------------------------------

module purge
module load gcc
module load python
module load cuda
module load cuDNN

eval "$(/work/home/mb14sola/anaconda3/bin/conda shell.bash hook)"
conda activate cybert

cd /work/projects/project02060/CyBERT/

srun python -m torch.distributed.launch --nproc_per_node=4 cybert/code/run_mlm.py --model_name_or_path "bert-base-uncased" --train_file "cybert/input/Corpus/cysec_corpus_15.txt"  --do_train --num_train_epochs 30 --per_device_train_batch_size 16 --learning_rate "2e-5" --weight_decay 0.01 --adam_beta1 0.9 --adam_beta2 0.999 --adam_epsilon 0.000001 --output_dir 'model/cybert_v100_dataset_15' --save_strategy "epoch" --warmup_steps 10000 --cache_dir "cache" --report_to="wandb"
