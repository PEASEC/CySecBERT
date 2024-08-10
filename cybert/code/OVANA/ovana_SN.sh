#!/bin/bash
#SBATCH -J CyBERT-OVANA
#SBATCH -a 1-5%5
#SBATCH --mail-type=END
#SBATCH -e /work/projects/project01762/CyBERT/logs_training/%x.err.OVANA.%A_%a
#SBATCH -o /work/projects/project01762/CyBERT/logs_training/%x.out.OVANA.%A_%a
#SBATCH --mem-per-cpu=8000
#SBATCH -t 0:20:00
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

srun python cybert/code/OVANA/run_ner_SN_Ranada.py \
        --model_name_or_path "/work/projects/project01762/CyBERT/model/cybert_ranada" \
        --task_name "ner" \
        --train_file "cybert/input/NER/tagged_all_SN/train.json" \
        --validation_file "cybert/input/NER/tagged_all_SN/test.json" \
        --do_train --do_eval --evaluation_strategy "epoch" \
        --num_train_epochs 4 --per_device_train_batch_size 16 \
        --output_dir 'cybert/model/OVANA_SN_Ranada' \
        --save_strategy "epoch" --report_to="wandb" \
        --seed $SLURM_ARRAY_TASK_ID
