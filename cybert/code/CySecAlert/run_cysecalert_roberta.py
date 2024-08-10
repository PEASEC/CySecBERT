import json
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import os
import time
import datetime
import wandb
wandb.login('true', '169903ffe3707fe1b844110ffb85887e0b1f1748')
os.chdir("/work/projects/project01762/CyBERT/")

model_name = "roberta-base"

wandb.init(project="CyBERT", entity="few_shot", group="CySecAlert_" + model_name)

with open("cybert/input/CySecAlert/dataset_1.json") as dataset_1_file:
    dataset_1 = json.load(dataset_1_file) 
with open("cybert/input/CySecAlert/dataset_2.json") as dataset_2_file:
    dataset_2 = json.load(dataset_2_file) 
        
dataset = [instance["text"] for instance in dataset_1] + [instance["text"] for instance in dataset_2]
labels = [instance["label"] for instance in dataset_1] + [instance["label"] for instance in dataset_2]
    
labels_binary = [1 if int(label) > 2 else 0 for label in labels]

X_train, X_test, y_train, y_test = train_test_split(dataset, labels_binary, test_size=0.2, random_state=42)
    
X_train_pos = [instance for i,instance in enumerate(X_train) if y_train[i] == 1]
X_train_neg = [instance for i,instance in enumerate(X_train) if y_train[i] == 0]

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

tokenized_dataset_train = tokenizer(X_train, padding='max_length', truncation=True)
tokenized_dataset_test = tokenizer(X_test, padding='max_length', truncation=True)
        

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = Dataset(tokenized_dataset_train, y_train)
test_dataset = Dataset(tokenized_dataset_test, y_test)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "Accuracy: " : accuracy_score(labels, predictions),
        "F1: " : f1_score(labels, predictions, pos_label=1), 
        "Precision_1: " : recall_score(labels, predictions, pos_label=1),
        "Recall_1: " : precision_score(labels, predictions, pos_label=1),
        "Precision_0: " : recall_score(labels, predictions, pos_label=0),
        "Recall_0: " : precision_score(labels, predictions, pos_label=0),
    }

training_args = TrainingArguments(
    output_dir= "./results",
    num_train_epochs=5,              
    per_device_train_batch_size=8,  
    per_device_eval_batch_size=64,   
    warmup_steps=500,                
    weight_decay=0.01,               
    logging_dir="./logs",            
    logging_steps=100,
    report_to="wandb",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)
        
trainer.train()
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
os.mkdir("cybert/model/CySecAlert_" + model_name + "_" + timestamp)
model.save_pretrained("cybert/model/CySecAlert_" + model_name + "_" + timestamp + "/")

trainer.evaluate()

