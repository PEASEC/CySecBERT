import json
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import os
import time
import datetime
os.chdir("/work/projects/project01762/CyBERT/")

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

print("Full dataset:", len(dataset))
print("Pos:", len([instance for i,instance in enumerate(dataset) if labels_binary[i] == 1]))
print("Neg:", len([instance for i,instance in enumerate(dataset) if labels_binary[i] == 0]))
