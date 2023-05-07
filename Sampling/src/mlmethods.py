from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
#from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from imblearn.combine import SMOTEENN, SMOTETomek


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
    
    # Grid search on the resampled data
    X_re, y_re = rebalance(X, y)
    
    pipeline = Pipeline([('standardizer', StandardScaler()),
                         ('classifier', LogisticRegression())])
    
    
    logistic_params = {'classifier__penalty': ['l1', 'l2', 'elasticnet'],
                 'classifier__C' : [0.01, 0.1, 1, 3, 5, 10, 30, 50],
                 'classifier__random_state' : [SEED]}

    
    grid_s = GridSearchCV(pipeline, logistic_params, scoring='f1', n_jobs=-1, cv=5, verbose=0)
    grid_s.fit(X_re, y_re)
    
    # Best params in a form that will work later
    keys = [key.replace('classifier__', '') for key in grid_s.best_params_.keys()]
    values = [value for value in grid_s.best_params_.values()]
    best_params = dict(zip(keys, values))
   
    
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
            model = LogisticRegression(**best_params, n_jobs=-1)
            model.fit(X_resampled_s, y_resampled)
            predictions = model.predict_proba(X_val_s)
            
            predictions_p_rw = reweight_proba(predictions[:,1], y_val.mean(), 0.5) # Reweight of proba
            y_hat_rw = (predictions_p_rw >= 0.5).astype(int) # Predictions after reweighting the proba

            results.append({
                            'F1': f1_score(y_val, y_hat_rw),
                            'Precision': precision_score(y_val, y_hat_rw),
                            'Recall': recall_score(y_val, y_hat_rw),
                            'AUC': roc_auc_score(y_val, predictions_p_rw),
                            'Best_params' : best_params
            })

            
            
    return pd.DataFrame(results)


# Random forest
def random_forest(X, y):
    
    # Grid search on the resampled data
    X_re, y_re = rebalance(X, y)
    
    pipeline = Pipeline([('standardizer', StandardScaler()),
                         ('classifier', RandomForestClassifier())])
    
    
    rf_params = {'classifier__max_depth': [3, 5, 7, 10], 
                 'classifier__bootstrap': [True], 
                 'classifier__random_state' : [SEED]
                 }
    
    
    grid_s = GridSearchCV(pipeline, rf_params, scoring='f1', n_jobs=-1, cv=5, verbose=0)
    grid_s.fit(X_re, y_re)
    
    # Best params in a form that will work later
    keys = [key.replace('classifier__', '') for key in grid_s.best_params_.keys()]
    values = [value for value in grid_s.best_params_.values()]
    best_params = dict(zip(keys, values))
   
    
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
            model = RandomForestClassifier(**best_params, n_jobs=-1)
            model.fit(X_resampled_s, y_resampled)
            predictions = model.predict_proba(X_val_s)
            
            predictions_p_rw = reweight_proba(predictions[:,1], y_val.mean(), 0.5) # Reweight of proba
            y_hat_rw = (predictions_p_rw >= 0.5).astype(int) # Predictions after reweighting the proba

            results.append({
                            'F1': f1_score(y_val, y_hat_rw),
                            'Precision': precision_score(y_val, y_hat_rw),
                            'Recall': recall_score(y_val, y_hat_rw),
                            'AUC': roc_auc_score(y_val, predictions_p_rw),
                            'Best_params' : best_params
            })

    return pd.DataFrame(results)
