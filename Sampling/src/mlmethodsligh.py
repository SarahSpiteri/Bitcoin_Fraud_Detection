# Requirements
import pandas as pd
import numpy as np
import networkx as nx

import pickle
import os

from node2vec import Node2Vec
import sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

import random
random.seed(42)

import warnings
warnings.filterwarnings('ignore')

SEED = 2022

# Function to reweight probabilities
def reweight_proba(pi,q1=0.5,r1=0.5):
    r0 = 1-r1
    q0 = 1-q1
    tot = pi*(q1/r1)+(1-pi)*(q0/r0)
    w = pi*(q1/r1)
    w /= tot
    return w


# Rebalance the data
def rebalance(X, y):
    smotenn_tomek = SMOTETomek(random_state=0, n_jobs=-1)
    X_resampled, y_resampled = smotenn_tomek.fit_resample(X, y)
    return X_resampled, y_resampled


# Logistic Regression
def logistic_regresion(X, y):
      
    # Outsample cross validated performance with the best parameters form the gridsearch
    splitter = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)

    results = []

    for train_index, val_index in splitter.split(X, y):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index] 
            y_train, y_val = y.iloc[train_index], y.iloc[val_index] 

            # Rebalancing the training data
            X_resampled, y_resampled = rebalance(X_train, y_train)
            
            # Standarization
            scaler = StandardScaler()
            scaler.fit(X_resampled)
            X_resampled_s = scaler.transform(X_resampled)
            X_val_s = scaler.transform(X_val)

            # Predictions
            model = LogisticRegression(n_jobs=-1)
            model.fit(X_resampled_s, y_resampled)
            predictions = model.predict(X_val_s)
            
            results.append({
                            'F1': f1_score(y_val, predictions),
                            'Precision': precision_score(y_val, predictions),
                            'Recall': recall_score(y_val, predictions)
            })
            
    return pd.DataFrame(results)


# Random forest
def random_forest(X, y):
    
    # Outsample cross validated performance with the best parameters form the gridsearch
    splitter = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)

    results = []

    for train_index, val_index in splitter.split(X, y):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index] 
            y_train, y_val = y.iloc[train_index], y.iloc[val_index] 

            # Rebalancing the training data
            X_resampled, y_resampled = rebalance(X_train, y_train)
            
            # Standarization
            scaler = StandardScaler()
            scaler.fit(X_resampled)
            X_resampled_s = scaler.transform(X_resampled)
            X_val_s = scaler.transform(X_val)

            # Predictions
            model = RandomForestClassifier(n_jobs=-1)
            model.fit(X_resampled_s, y_resampled)
            predictions = model.predict(X_val_s)
            
            results.append({
                            'F1': f1_score(y_val, predictions),
                            'Precision': precision_score(y_val, predictions),
                            'Recall': recall_score(y_val, predictions)
            })

    return pd.DataFrame(results)
