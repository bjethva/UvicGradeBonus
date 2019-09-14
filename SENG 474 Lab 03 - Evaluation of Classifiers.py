#!/usr/bin/env python
# coding: utf-8

# In[54]:


get_ipython().run_line_magic('matplotlib', 'inline')
import csv

import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[29]:


# data from http://archive.ics.uci.edu/ml/datasets/Anuran+Calls+%28MFCCs%29
label_names = ['AdenomeraAndre', 'AdenomeraHylaedactylus', 'Ameeregatrivittata', 'HylaMinuta', 'HypsiboasCinerascens', 'HypsiboasCordobae',
               'LeptodactylusFuscus', 'OsteocephalusOophagus', 'Rhinellagranulosa', 'ScinaxRuber']

def load_frog_data():
    '''Function for loading Frog MFCC data.  See data/Readme.txt for details about the data set'''

    # Load Data from CSV File
    with open("data/Frogs_MFCCs.csv") as f:
        reader = csv.reader(f, delimiter=",")

        X = []   # array for feature vectors
        y = []   # array for ground truth labels
        for i, row in enumerate(reader):
            # ignore first row since it is just headers
            if i ==0:
                continue

            X.append(row[:-4])                    # append feature vector from CSV row
            y.append(label_names.index(row[-2]))  # append ground truth label from CSV row (converting from categorical to int)

    return np.array(X).astype(float), np.array(y).astype(int)


# In[30]:


# Load the Data
X, y = load_frog_data()
print ("X Shape:", X.shape, "\tType:", X.dtype)
print ("y Shape:", y.shape, "\tType:", y.dtype)


# In[94]:


# Print a few rows of the data
float_formatter = lambda x: "%7.4f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

for i in range(0, 7195, 700):
    print (y[i], ":", label_names[y[i]])
    print (X[i],end="\n\n")
    
print (np.unique(y, return_counts=True))


# In[56]:


# Import evaluation functions
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


# In[57]:


# Initialize a few classifiers for comparison
clfs = [DecisionTreeClassifier(), SVC(), BernoulliNB()]


# In[72]:


# Train and Score each classifier on a standard single training/test split of the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
for clf in clfs:
    clf.fit(X_train, y_train)
    print(type(clf).__name__, clf.score(X_test, y_test))


# In[77]:


# Evaluate using 5-Fold Cross Evaluation
for clf in clfs:
    score = cross_val_score(clf, X, y, cv=5)
    print (type(clf).__name__, score, np.average(score))


# In[103]:


# print confusion matrices for each classifier
plt.figure(figsize=(10,4))
for i, clf in enumerate(clfs):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
        
    print (type(clf).__name__)
    print (cm)
    print()
    
    plt.subplot(1, len(clfs), i+1)
    plt.imshow(cm, cmap="Blues")


# In[101]:


# visualize confusion matrices
float_formatter = lambda x: "%5.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

plt.figure(figsize=(10,4))
for i, clf in enumerate(clfs):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    print (type(clf).__name__)
    print (cm)
    print()
    
    plt.subplot(1, len(clfs), i+1)
    plt.imshow(cm, cmap="Blues")


# In[ ]:


# Calculate Precision and Recall

