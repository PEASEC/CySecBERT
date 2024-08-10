import sys
sys.path.insert(0, "/work/projects/project02060/CyBERT/cybert/code/SuperGLUE/exp/")
#sys.path.append("/work/projects/project02060/CyBERT/cybert/code/SuperGLUE/ji")

import os

import jiant.utils.python.io as py_io
import jiant.proj.simple.runscript as simple_run
import jiant.scripts.download_data.runscript as downloader
from transformers import AutoTokenizer
from transformers import AutoModel
import torch


from argparse import ArgumentParser

import torch
import gc

parser = ArgumentParser()
parser.add_argument("-r", "--run", help="Run id")
parser.add_argument("-m", "--model", help="Model name")

args = parser.parse_args()
run_id = args.run
model = args.model

HF_PRETRAINED_MODEL_NAME = r"/work/projects/project02060/CyBERT/model/" + model

tokenizer = AutoTokenizer.from_pretrained(HF_PRETRAINED_MODEL_NAME)
model = AutoModel.from_pretrained(HF_PRETRAINED_MODEL_NAME, num_labels=2)
# make dir for model
os.makedirs(HF_PRETRAINED_MODEL_NAME + "/model", exist_ok=True) 
torch.save(model.state_dict(), HF_PRETRAINED_MODEL_NAME + "/model/model.p")

tasks = ["boolq"]

for TASK_NAME in tasks:

    # Remove forward slashes so RUN_NAME can be used as path
    MODEL_NAME = HF_PRETRAINED_MODEL_NAME.split("/")[-1]
    RUN_NAME = f"simple_{TASK_NAME}_{MODEL_NAME}_{run_id}"
    EXP_DIR = "/work/projects/project02060/CyBERT/cybert/code/SuperGLUE/boolq_exp_" + str(run_id)
    DATA_DIR = "/work/projects/project02060/CyBERT/cybert/code/SuperGLUE/boolq_exp_" + str(run_id) + "/tasks"

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(EXP_DIR, exist_ok=True)


    downloader.download_data([TASK_NAME], DATA_DIR)

    if TASK_NAME == "record":
        bs = 32
        epochs = 2
    else:
        bs = 64
        epochs = 3

    args = simple_run.RunConfiguration(
        run_name=RUN_NAME,
        exp_dir=EXP_DIR,
        data_dir=DATA_DIR,
        hf_pretrained_model_name_or_path=HF_PRETRAINED_MODEL_NAME,
        tasks=TASK_NAME,
        train_batch_size=bs,
        num_train_epochs=epochs
    )

    simple_run.run_simple(args)

    gc.collect()
    torch.cuda.empty_cache()
