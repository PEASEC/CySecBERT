import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

with open("../../../input/CySecAlert/dataset_1.json") as dataset_1_file:
    dataset_1 = json.load(dataset_1_file) 
with open("../../../input/CySecAlert/dataset_2.json") as dataset_2_file:
    dataset_2 = json.load(dataset_2_file) 
        
dataset = [instance["text"] for instance in dataset_1] + [instance["text"] for instance in dataset_2]
labels = [instance["label"] for instance in dataset_1] + [instance["label"] for instance in dataset_2]
    
labels_binary = [1 if int(label) > 2 else 0 for label in labels]

X_train, X_test, y_train, y_test = train_test_split(dataset, labels_binary, test_size=0.2, random_state=42)
    
X_train_pos = [instance for i,instance in enumerate(X_train) if y_train[i] == 1]
X_train_neg = [instance for i,instance in enumerate(X_train) if y_train[i] == 0]

# create bag of words with TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# create random forest classifier with standard parameters
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()

# train the model
clf.fit(X_train, y_train)

# predict the labels
y_pred = clf.predict(X_test)

# evaluate the model
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("F1: ", f1_score(y_test, y_pred, pos_label=1))
print("Precision_1: ", recall_score(y_test, y_pred, pos_label=1))
print("Recall_1: ", precision_score(y_test, y_pred, pos_label=1))
print("Precision_0: ", recall_score(y_test, y_pred, pos_label=0))
print("Recall_0: ", precision_score(y_test, y_pred, pos_label=0))

