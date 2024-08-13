# CySecBERT

CySecBERT is a domain-adapted version of the BERT model tailored for cybersecurity tasks. This repository contains the experimental codebase for further training BERT, fine-tuning the resulting BERT model, running various experiments, and analyzing training logs. Please note that this repository does not include any datasets, pretrained models, or checkpoints. These resources can be obtained by contacting the authors.

Note: Internally, the project was called CyBERT, so all references to it normally refer to the CySecBERT model. 

## Contact and Support

If you have any questions, need access to datasets, model checkpoints, or the complete research data, or if you encounter any bugs, please feel free to contact me!

## Model

https://huggingface.co/markusbayer/CySecBERT

## Project Structure

The repository is organized as follows:

```
cysecbert/
│
├── logs_training/               # Contains logging files from training
├── results/                     # Contains checkpoints of trained models
├── cybert/
│   ├── input/                   # Directory for datasets
│   ├── model/                   # Directory for storing resulting models
│   ├── code/
│       ├── CySecAlert/          # Code for running the CySecAlert experiment
│       ├── MSExchange/          # Code for running the MSExchange experiment
│       ├── OVANA/               # Code for running the OVANA experiment
│       ├── SuperGLUE/           # Code for running the SuperGLUE experiment
│       ├── run_mlm.py           # Script for domain adapting BERT (CySecBERT)
│   ├── CyBERT.ipynb             # Jupyter notebook for domain adaptation and fine-tuning (deprecated)
```

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/PEASEC/CySecBERT.git
   cd CySecBERT
   ```

2. Install the required dependencies:
   ```bash
   pip install -r cybert/requirements.txt
   ```

## Usage

### Domain Adapting BERT

To adapt BERT to the cybersecurity domain, use the `run_mlm.py` script. This script allows you to further train the BERT model on a domain-specific corpus.

Example command:
```
python cybert/code/run_mlm.py \
    --model_name_or_path bert-base-uncased \
    --train_file /path/to/train.txt \
    --validation_file /path/to/validation.txt \
    --max_seq_length 512 \
    --mlm_probability 0.15 \
    --output_dir /path/to/save/model \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --save_steps 10_000 \
    --logging_dir /path/to/logging
```

#### Model Parameters

The `run_mlm.py` script contains several key parameters for model training and fine-tuning:

- **ModelArguments**:
  - `model_name_or_path`: The model checkpoint for weights initialization. Don't set if you want to train a model from scratch.
  - `model_type`: If training from scratch, pass a model type. 
  - `config_name`, `tokenizer_name`: Names or paths of the config and tokenizer if different from the model name.
  - `cache_dir`: Directory to store downloaded models.
  - `use_fast_tokenizer`: Whether to use a fast tokenizer backed by the `tokenizers` library.
  - `model_revision`: Specific model version to use (branch name, tag, or commit id).
  - `use_auth_token`: Use a Hugging Face authentication token for private models.

- **DataTrainingArguments**:
  - `dataset_name`: Name of the dataset (via the Hugging Face datasets library).
  - `dataset_config_name`: The configuration name of the dataset to use (via the datasets library).
  - `train_file`, `validation_file`: Paths to the training and validation data files (text files).
  - `validation_split_percentage`: Percentage of the training set used for validation.
  - `max_seq_length`: Maximum sequence length after tokenization.
  - `mlm_probability`: Ratio of tokens to mask for the masked language modeling loss.
  - `line_by_line`: Whether to treat each line in the dataset as a separate sequence.
  - `pad_to_max_length`: Whether to pad all sequences to the `max_seq_length`.

### Fine-Tuning and Experiments

The repository includes scripts for fine-tuning CySecBERT on various tasks. These are located under `cybert/code/` in directories named after specific experiments (e.g., `CySecAlert`, `MSExchange`, `OVANA`, `SuperGLUE`).

To run an experiment, navigate to the corresponding directory and execute the script provided. Ensure that you have configured the necessary environment variables and paths for datasets and models.

### Training Logs

Training logs are stored in the `logs_training/` directory. These logs provide detailed insights into the training process.

## Notebook for Domain Adaptation and Fine-Tuning

The repository includes an older Jupyter notebook, `CyBERT.ipynb`, that was used during the initial phases of domain adaptation and fine-tuning. While this notebook is deprecated, it may still provide useful insights into the process.

## Getting Help

For any questions, issues, or requests for datasets, models, or checkpoints, please contact the authors directly.

