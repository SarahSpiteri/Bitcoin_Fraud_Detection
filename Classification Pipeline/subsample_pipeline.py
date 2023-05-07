# General imports
import os,sys,inspect
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import pandas as pd
import numpy as np
import zipfile
import seaborn as sns 
from datetime import datetime
from functools import reduce
from time import time
import matplotlib.dates as mdates
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import plotly.offline as py
import string
import pickle
import re
import math
import random as rd
rd.seed(42)

# Pre-processing
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.ensemble import RandomForestClassifier
import keras
import tensorflow as tf
from tensorflow import keras

# Classification model
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVC
from sklearn.base import BaseEstimator
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

# Accuracy check
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, recall_score, precision_score, f1_score, precision_recall_fscore_support, classification_report, confusion_matrix, ConfusionMatrixDisplay, mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelBinarizer
from plotly.offline import iplot, plot, init_notebook_mode

# Graph embedding
import networkx as nx
from networkx.algorithms.community.modularity_max import greedy_modularity_communities
from sklearn.metrics.pairwise import pairwise_distances
from gensim.models import Word2Vec
from networkx.algorithms.community.label_propagation import label_propagation_communities
from networkx.algorithms.community import k_clique_communities
import community # python-louvain
import stellargraph as sg
from stellargraph import StellarGraph
from stellargraph.data import BiasedRandomWalk, UnsupervisedSampler
from stellargraph.mapper import Node2VecLinkGenerator, Node2VecNodeGenerator
from stellargraph.layer import Node2Vec as n2v_stellar
from stellargraph.layer import link_classification
from node2vec import Node2Vec # Leskovec

# Visualization 
import igraph as ig
import plotly.graph_objs as go
import plotly.express as px
from plotly.offline import iplot, plot, init_notebook_mode
from bokeh.io import output_notebook, show, save
from bokeh.models import *
from bokeh.plotting import figure, show, from_networkx
from bokeh.palettes import viridis, Viridis8
from bokeh.transform import linear_cmap
from networkx.algorithms import community
from sklearn.manifold import TSNE
from community import community_louvain
output_notebook()

from custom_functions import *

# Instantiate 
sc = StandardScaler()


def subsample_pipeline(filename, test_size, walk_count, walk_length, p, q, batch_size, epochs, emb_size, columns_to_drop, target_variable):
    
    algorithms = {'LR': LogisticRegression(),
                  'RF': RandomForestClassifier()}
    metrics = {
        'ROC AUC':  roc_auc_score,
        'Recall': recall_score,
        'Precision': precision_score,
        'F1': f1_score  
    }
    print(f'Execute pipeline for file: {filename}')
    
    
    # Initializing tables of results with their corresponding indices 
    results_table = pd.DataFrame(index=['ROC AUC','Recall','Precision', 'F1 Score'])

    # 1. Loading initial files
    trans_3w = load_data(filename)

    # 2. Extract original graph properties 
    original_graph_properties = extract_original_graph_properties(trans_3w)
    
    # 3. Extract number of self loops for each address
    self_loops = extract_self_loops(trans_3w)
     
    # 4. Create dataframe with non-embbeded features
    df_features = extract_df_features(trans_3w)
    
    # 5. Ten different random walks of 50k nodes resulting in ten different sampled subgraphs
    for iter_ in range(10):
        # Load the sampled subgraph
        x = pd.read_csv(r'Outputs/sample_'  + str(iter_ + 1) + '.csv')
        subgraph = nx.from_pandas_edgelist(x[['source', 'target']])
        print("Sample: ", str(iter_ + 1))
        print('Number of nodes:', subgraph.number_of_nodes())
        print('Number of edges:', subgraph.number_of_edges())

        # 6. Extract edges from subgraph
        sampled_edges = extract_graph_edges(subgraph)

        # 7. Extract nodes from subgraph
        sampled_nodes = extract_graph_nodes(subgraph)

        # 8. Create non-embedded dataframe
        df_non_embedded_feat = merge_df(sampled_nodes, self_loops, df_features)

        # 9. Extract subgraph embeddings with node2vec algorithm
        node_embeddings, df_embedded_feat = node2vec_2(subgraph, sampled_nodes, walk_count, walk_length, p, q, emb_size)

        # 10. Create training dataframe by concatenating embedded and non-embedded features
        df = create_train_df(df_embedded_feat, df_non_embedded_feat)

        # 11. Split data into train/test sets
        X_train_original, X_train, X_test, y_train, y_test = split_df_train_test(df, target_variable, columns_to_drop, test_size)

        # 12. Preprocess dataframe
        X_train, X_test, y_train = preprocess_dataframe(X_train, X_test, y_train)

        # 13. Train model and get results for each algorithm in the list
        for name, algorithm in algorithms.items():
            # Train 
            model = train_model(algorithm, X_train, y_train)
            # Predict
            y_test_pred = test_model(model, X_test)
            # Assess results
            results = model_evaluation(y_test, y_test_pred, metrics)
            results_table[name] = results.values()

        # Display results
        print("Results sample ", str(iter_ + 1))
        display(pd.DataFrame(results_table))                                                                                                      