import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import os
import time
import datetime
os.chdir("/work/projects/project01762/CyBERT/")

df_train_full = pd.read_csv("cybert/input/MSExchange/df_train_full.csv")
df_dev_full = pd.read_csv("cybert/input/MSExchange/df_dev_full.csv")
df_test = pd.read_csv("cybert/input/MSExchange/df_test.csv")

X_train, y_train = df_train_full["text"].tolist(), df_train_full["label"].tolist()
X_dev, y_dev = df_dev_full["text"].tolist(), df_dev_full["label"].tolist()
X_test, y_test = df_test["text"].tolist(), df_test["label"].tolist()


X_all = X_train + X_dev + X_test
y_all = y_train + y_dev + y_test

print("Full dataset: ", len(X_all))
print("Pos: ", len([instance for i,instance in enumerate(X_all) if y_all[i] == 1]))
print("Neg: ", len([instance for i,instance in enumerate(X_all) if y_all[i] == 0]))
