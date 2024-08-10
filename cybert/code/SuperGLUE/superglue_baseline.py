import sys
sys.path.insert(0, "/work/projects/project02060/CyBERT/cybert/code/SuperGLUE/exp/")

import os

import jiant.utils.python.io as py_io
import jiant.proj.simple.runscript as simple_run
import jiant.scripts.download_data.runscript as downloader

from argparse import ArgumentParser

import torch
import gc

parser = ArgumentParser()
parser.add_argument("-r", "--run", help="Run id")

args = parser.parse_args()
run_id = args.run

HF_PRETRAINED_MODEL_NAME = r"/work/projects/project02060/CyBERT/model/cybert_v100_r"

tasks = ["superglue_broadcoverage_diagnostics", "record", "rte", "wic", "wsc", "superglue_winogender_diagnostics", "boolq", "cb", "copa", "multirc"]

for TASK_NAME in tasks:

    # Remove forward slashes so RUN_NAME can be used as path
    MODEL_NAME = HF_PRETRAINED_MODEL_NAME.split("/")[-1]
    RUN_NAME = f"simple_{TASK_NAME}_{MODEL_NAME}_{run_id}"
    EXP_DIR = "/work/projects/project02060/CyBERT/cybert/code/SuperGLUE/exp_" + str(run_id)
    DATA_DIR = "/work/projects/project02060/CyBERT/cybert/code/SuperGLUE/exp_" + str(run_id) + "/tasks"

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
