
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
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split, HalvingGridSearchCV
from imblearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.experimental import enable_halving_search_cv  # noqa
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
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, precision_score,f1_score, confusion_matrix, roc_curve, roc_auc_score, precision_recall_fscore_support, classification_report, mean_squared_error
from sklearn.model_selection import cross_val_score,train_test_split
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
from trans2vec import *

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

# Instantiate 
sc = StandardScaler()



# Load original dataframe & exclude NAs from output_address and negative amounts 
def load_data(df):
    trans_3w = pd.read_csv(df)
    print('Initial shape: ', trans_3w.shape)
    
    # Drop the 139 NaN observations from output_address
    trans_3w = trans_3w[trans_3w['output_address'].notna()]
    print('Shape without NaN: ', trans_3w.shape)
    
    # Drop negative amounts as they are abnormal
    trans_3w = trans_3w.drop(trans_3w[trans_3w.ammount < 0].index)
    print('Shape without negative amounts: ', trans_3w.shape)
    
    # Convert date to the right format
    trans_3w['block_time'] = pd.to_datetime(trans_3w['block_time'], format="%Y-%m-%d %H:%M:%S")
    
    return trans_3w


# Extract original graph properties 
def extract_original_graph_properties(df):
    # Create multi directed and weighted graph 
    Direct_Weight_G = nx.MultiDiGraph()
    Direct_Weight_G.add_weighted_edges_from(df[['input_address', 'output_address', 'ammount']].values)

    # Create node properties dataframe
    degree = [val for (node, val) in Direct_Weight_G.degree()]
    in_degree = [val for (node, val) in Direct_Weight_G.in_degree()]
    out_degree = [val for (node, val) in Direct_Weight_G.out_degree()]
    address = [node for (node, val) in Direct_Weight_G.degree()]

    original_graph_properties = pd.DataFrame()
    original_graph_properties['address'] = address
    original_graph_properties['degree'] = degree
    original_graph_properties['in_degree'] = in_degree
    original_graph_properties['out_degree'] = out_degree
    #print(original_graph_properties.shape)
    # display(original_graph_properties.head(3))
    
    return original_graph_properties


# Add 1 for the input_addresses that are equal to the output_address
def self_loops(df):
    df['self_loops'] = 0
    df.loc[(df.input_address == df.output_address), 'self_loops'] = 1
    
    return df


# Extract self loops
def extract_self_loops(df):
    data = self_loops(df)
    
    # Create new dataframe with unique input_address and the sum of their self loops
    selfloops = pd.DataFrame(df.groupby(['input_address'], as_index=False)['self_loops'].sum()) 
    selfloops.rename(columns={selfloops.columns[0]: "address" }, inplace = True)
    print('Number of self-loops: ', selfloops['self_loops'].sum())
    
    return selfloops


# Create dataframe with non-embbeded features
def extract_df_features(df):
    # Get key descriptive statistics of transaction ammounts where the node appeared as the input address
    in_amounts = df.groupby(['input_address']).agg(ina_total=('ammount', 'sum'), ina_std=('ammount', pd.Series.std), 
                                                      ina_max=('ammount', pd.Series.max), ina_min=('ammount', pd.Series.min),
                                                      ina_median=('ammount', 'median'))
    in_amounts['address'] = in_amounts.index
    
    # Get key descriptive statistics of transaction ammounts where the node appeared as the output address
    out_amounts = df.groupby(['output_address']).agg(outa_total=('ammount', 'sum'), outa_std=('ammount',pd.Series.std), 
                                                  outa_max=('ammount', pd.Series.max), outa_min=('ammount', pd.Series.min),
                                                  outa_median=('ammount', 'median'))
    out_amounts['address'] = out_amounts.index
    
    # Get key descriptive statistics of transaction fees where the node appeared as the input address
    in_fees = df.groupby(['input_address']).agg(inf_total=('fees', 'sum'), inf_std=('fees', pd.Series.std), 
                                                  inf_max=('fees', pd.Series.max), inf_min=('fees', pd.Series.min),
                                                  inf_median=('fees', 'median'))
    in_fees['address'] = in_fees.index
    
    # Get key descriptive statistics of transaction fees where the node appeared as the output address
    out_fees = df.groupby(['output_address']).agg(outf_total=('fees', 'sum'), outf_std=('fees', pd.Series.std), 
                                                  outf_max=('fees', pd.Series.max), outf_min=('fees', pd.Series.min),
                                                  outf_median=('fees', 'median'))
    out_fees['address'] = out_fees.index

    # Merge properties of transactions
    dfs = [in_amounts, out_amounts, in_fees, out_fees]
    df_features = reduce(lambda left,right: pd.merge(left,right, on = ['address'], how = 'outer'), dfs)
    
    # Difference in total ammounts (potential indicator of obfuscation patterns)
    df_features['diff_amount'] = df_features['outa_total'] - df_features['ina_total']
    
    # Difference in total fees 
    df_features['diff_fees'] = df_features['outf_total'] - df_features['inf_total']
    
    # Save to csv
    df_features.to_csv("df_features.csv")
    # df_features = pd.read_csv(r"df_features.csv")
    # df_features = df_features.drop(['Unnamed: 0'], axis=1)

    return df_features


# Load input/output label files
def load_input_labels(path):
    input_labels = pickle.load(open(path,'rb'))
    return input_labels

def load_output_labels(path):
    output_labels = pickle.load(open(path,'rb'))
    return output_labels

# Append fraudulent/non-fraudulent flag to df
def assign_flag(sampled_df):
    sampled_df['flag'] = 0
    sampled_df.loc[(sampled_df.input_flag == 1) | (sampled_df.output_flag == 1), 'flag'] = 1

# Extract edges from subgraph
def extract_graph_edges(subgraph):
    # Transform networkx edges to dataframe
    sampled_trans_3w = nx.to_pandas_edgelist(subgraph)
    # Add flags
    sampled_trans_3w = sampled_trans_3w.rename(columns = {"source": "input_address", "target": "output_address"})
    sampled_trans_3w['input_flag'], sampled_trans_3w['output_flag'], sampled_trans_3w['flag'] = 0, 0, 0
    # Create flag for fraudulent/high-risk input address
    input_labels = load_input_labels('data/input_labels.txt')
    sampled_trans_3w.loc[sampled_trans_3w.input_address.isin(input_labels), 'input_flag'] = 1
    sampled_trans_3w['input_flag'].value_counts()
    # Create flag for fraudulent/high-risk output address
    output_labels = load_output_labels('data/output_labels.txt')
    sampled_trans_3w.loc[sampled_trans_3w.output_address.isin(output_labels), 'output_flag'] = 1
    sampled_trans_3w['output_flag'].value_counts()
    
    assign_flag(sampled_trans_3w)
    
    print('Target variable distribution within the subsampled edges:')
    display(sampled_trans_3w['flag'].value_counts())
    
    sampled_trans_3w.to_csv('sampled_edges.csv')
    return(sampled_trans_3w)


# Extract nodes from subgraph
def extract_graph_nodes(subgraph):
    # Extract nodes addressses from the subsampled graph
    address = [node for (node, val) in subgraph.degree()]
    nodes = pd.DataFrame()
    nodes['address'] = address
    
    # Add flags
    nodes['input_flag'], nodes['output_flag'], nodes['flag'] = 0, 0, 0
    # Create flag for fraudulent/high-risk input address
    input_labels = load_input_labels('data/input_labels.txt')
    nodes.loc[nodes.address.isin(input_labels), 'input_flag'] = 1
    nodes['input_flag'].value_counts()
    # Create flag for fraudulent/high-risk output address
    output_labels = load_output_labels('data/output_labels.txt')
    nodes.loc[nodes.address.isin(output_labels), 'output_flag'] = 1
    nodes['output_flag'].value_counts()
    
    assign_flag(nodes)
    
    print('Target variable distribution within the subsampled nodes:')
    display(nodes['flag'].value_counts())
    
    nodes.to_csv('sampled_nodes.csv')
    return(nodes)


# Check the distribution of target variable 
def plot_target_distribution(df):
    fig, ax = plt.subplots(figsize=(8,5), dpi=100)
    patches, texts, autotexts = ax.pie(df['flag'].value_counts(), autopct= '%1.1f%%', shadow=True, 
                                           startangle=90, explode=(0.1, 0), labels=['Licit','Fraudulent'])
    plt.setp(autotexts, size=12, color = 'black', weight='bold')
    autotexts[1].set_color('white');
    plt.title('Target variable distribution', fontsize=14)
    plt.show()


# Concatenate sampled_nodes with self loops and non-embedded features 
def merge_df(sampled_nodes, self_loops, features):
    df_non_embedded_feat = pd.merge(sampled_nodes, self_loops, on = "address", how = "left")
    df_non_embedded_feat = pd.merge(df_non_embedded_feat, features, on = "address", how = "left")
    
    # Replace NAs with 0
    df_non_embedded_feat = df_non_embedded_feat.fillna(0)
    
    return (df_non_embedded_feat)


# Node2vec algorithm for graph embedding using Stellar Graph library 
def node2vec_1(subgraph, sampled_edges, sampled_nodes, walk_count, walk_length, p, q, batch_size, epochs, emb_size): 
    # Create the directed multigraph of the network using stellargraph library
    graph = StellarGraph.from_networkx(subgraph)
    
    # Generate random walks for mapping the network
    walk_count = walk_count
    walk_length = walk_length
    karate_walk = BiasedRandomWalk(
        graph,
        n=walk_count,
        length=walk_length,
        p=p,  
        q=q, )

    unsupervised_samples = UnsupervisedSampler(graph, nodes=list(graph.nodes()), walker=karate_walk)
    batch_size = batch_size
    epochs = epochs
    generator = Node2VecLinkGenerator(graph, batch_size)
    emb_size = emb_size
    
    # Train the model for embeddings size 100
    node2vec = n2v_stellar(emb_size, generator=generator)
    x_inp, x_out = node2vec.in_out_tensors()
    predictions = prediction = link_classification(
        output_dim=1, output_act="sigmoid", edge_embedding_method="dot"
    )(x_out)

    model = keras.Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.binary_crossentropy,
        metrics=[keras.metrics.binary_accuracy],)
    
    # Fit the model
    history = model.fit(
        generator.flow(unsupervised_samples),
        epochs=epochs,
        verbose=1,
        use_multiprocessing=False,
        workers=4,
        shuffle=True,)
    
    x_inp_src = x_inp[0]
    x_out_src = x_out[0]
    embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
    unique_ids = np.unique(list(df["input_address"]) + 
                         list(df["output_address"]))
    unique_ids = pd.DataFrame({"Node_ids":unique_ids })
    unique_ids = pd.merge(unique_ids,df, how = "left", left_on = "Node_ids", right_on = "input_address")
    unique_ids = pd.merge(unique_ids,df, how = "left", left_on = "Node_ids", right_on = "output_address")
    
    source_vals = df[["input_address","flag"]].drop_duplicates()
    source_vals.columns = ["Node_ID","flag"]
    target_vals = df[["output_address","flag"]].drop_duplicates()
    target_vals.columns = ["Node_ID","flag"]
    testing = source_vals.append(target_vals)
    testing = testing.drop_duplicates(subset = "Node_ID",keep = "first")
    node_gen = Node2VecNodeGenerator(graph, batch_size).flow(graph.nodes())
    node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1)
    
    # Convert to dataframe 
    node_embeddings_df = pd.DataFrame(node_embeddings)
    
    # Add header for features by renaming columns
    column_group = {str(ii): "feature_" + str(ii+1) for ii in range(200)}
    column_all = dict(column_group)
    column_all = {int(jj): item_kk for jj,item_kk in column_all.items()}
    node_embeddings_df = node_embeddings_df.rename(columns=column_all)
    
    # Save to csv
    print('Embeddings dataframe shape: ', node_embeddings_df.shape)
    node_embeddings_df.to_csv("embeddings.csv")
    
    # Create the embedded features dataframe for further predictions
    df_embedded_feat = pd.concat([sampled_nodes, node_embeddings_df], axis=1)
    
    return(node_embeddings, df_embedded_feat)


# Node2vec algorithm for graph embedding using Leskovec's library 
def node2vec_2(subgraph, sampled_nodes, walk_count, walk_length, p, q, emb_size): # LESKOVEC
    # Adjustment to make embedding work
    for u,v,d in subgraph.edges(data=True):
        d['weight'] = 1

    node2vec = Node2Vec(subgraph, dimensions=emb_size, p=p, q=q, walk_length=walk_length, num_walks=walk_count, seed=42)
    model = node2vec.fit(window=10)

    # Represent embedding as DataFrame
    node_embeddings = ([model.wv.get_vector(str(n)) for n in subgraph.nodes()])
    node_embeddings_df = pd.DataFrame(node_embeddings)
    
    # Add header for features by renaming columns
    column_group = {str(ii): "feature_" + str(ii+1) for ii in range(200)}
    column_all = dict(column_group)
    column_all = {int(jj): item_kk for jj,item_kk in column_all.items()}
    node_embeddings_df = node_embeddings_df.rename(columns=column_all)
    
    # Save to csv
    print('Embeddings dataframe shape: ', node_embeddings_df.shape)
    # node_embeddings_df.to_csv("embeddings.csv")
    
    # Create the embedded features dataframe for further predictions
    df_embedded_feat = pd.concat([sampled_nodes, node_embeddings_df], axis=1)
    
    return(node_embeddings, df_embedded_feat)


# Node2vec algorithm for graph embedding using word2vec 
def node2vec_3(graph, sampled_nodes, walk_count, walk_length, p, q, emb_size):
    # Create the directed multigraph of the network using stellargraph library
    graph = StellarGraph.from_networkx(subgraph)
    display(graph.info())
    
    rw = BiasedRandomWalk(graph, p = p, q = q, n = 10, length = walk_length, seed=42, weighted = True)

    # Generate random walk
    walks = rw.run(nodes=list(graph.nodes())
                   # root nodes
                  )

    # Pass the random walks to a list
    str_walks = [[str(n) for n in walk] for walk in walks]

    # Train the model for 20 epochs with a window size of 10
    model = Word2Vec(str_walks, vector_size=emb_size, window=10, min_count=1, sg=1, workers=4, epochs=epochs)

    # Retrieve node embeddings and their corresponding transaction ids
    node_ids = model.wv.index_to_key  # list of node ids
    node_embeddings = (model.wv.vectors)

    # Represent embedding as DataFrame
    node_embeddings = ([model.wv.get_vector(str(n)) for n in graph.nodes()])
    node_embeddings_df = pd.DataFrame(node_embeddings)
    
    # Add header for features by renaming columns
    column_group = {str(ii): "feature_" + str(ii+1) for ii in range(200)}
    column_all = dict(column_group)
    column_all = {int(jj): item_kk for jj,item_kk in column_all.items()}
    node_embeddings_df = node_embeddings_df.rename(columns=column_all)

    # Save to csv
    print('Embeddings dataframe shape: ', node_embeddings_df.shape)
    node_embeddings_df.to_csv("embeddings.csv")
    
    # Create the embedded features dataframe for further predictions
    df_embedded_feat = pd.concat([sampled_nodes, node_embeddings_df], axis=1)
    
    return(node_embeddings, df_embedded_feat)


# Trans2vec algorithm for graph embedding using word2vec 
def trans2vec(subsample, sampled_nodes):
    # Retrieve node embeddings and their corresponding transaction ids
    t2v = Trans2vec(subsample)
    x = t2v.embeddings
    node_embeddings = x.drop(['address'], axis=1)

    # Represent embedding as DataFrame
    node_embeddings_df = pd.DataFrame(x)
    
    # Add header for features by renaming columns
    column_group = {str(ii): "feature_" + str(ii+1) for ii in range(200)}
    column_all = dict(column_group)
    column_all = {int(jj): item_kk for jj,item_kk in column_all.items()}
    node_embeddings_df = node_embeddings_df.rename(columns=column_all)

    # Save to csv
    print('Embeddings dataframe shape: ', node_embeddings_df.shape)
    node_embeddings_df.to_csv("embeddings.csv")
    
    # Create the embedded features dataframe for further predictions
    df_embedded_feat = pd.concat([sampled_nodes, node_embeddings_df], axis=1)
    
    return(node_embeddings, df_embedded_feat)


# Visualize the new graph by applying t-SNE transformation on node embeddings
def tsne(node_embeddings, df):
    tsne = TSNE(n_components=2)
    node_embeddings_2d = tsne.fit_transform(node_embeddings)
    node_targets = df['flag']

    # Draw the points
    alpha = 0.7
    label_map = {l: i for i, l in enumerate(np.unique(node_targets))}
    node_colours = [label_map[target] for target in node_targets]

    plt.figure(figsize=(10, 8))
    plt.title('t-SNE transformation on node embeddings', fontsize=14)
    plt.scatter(
        node_embeddings_2d[:, 0],
        node_embeddings_2d[:, 1],
        c=node_colours,
        cmap="jet",
        alpha=alpha,)


# Visualize fraudulent ids 
def visualize_fraudulent_nodes(sampled_nodes, sampled_edges):
    illicit_ids = sampled_nodes[sampled_nodes['flag'] == 1]['address']

    # make list of illicit accounts ids in edge set 
    illicit_edges = sampled_edges[sampled_edges['input_address'].isin(illicit_ids)]
    graph_illicit = nx.from_pandas_edgelist(illicit_edges, source = 'input_address', target = 'output_address', create_using = nx.DiGraph())

    plot = figure(tools="pan,wheel_zoom,save,reset", active_scroll='wheel_zoom',
                x_range=Range1d(-10.1, 10.1), y_range=Range1d(-10.1, 10.1), 
                  title='Fraudulent Addresses Interaction', plot_width=850)

    network_graph = from_networkx(graph_illicit, nx.spring_layout, scale=10, center=(0, 0))
    network_graph.node_renderer.glyph = Circle(size=15, fill_color='teal')
    network_graph.edge_renderer.glyph = MultiLine(line_alpha=0.5, line_width=1)
    plot.renderers.append(network_graph)

    show(plot)
    #save(plot, filename=f"{title}.html")


# Create training df by concatenating embedded and non-embedded features
def create_train_df(df_embedded_feat, df_non_embedded_feat):
    df = df_embedded_feat.merge(df_non_embedded_feat,  left_index=True, right_index=True, how = "outer",  suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
    
    return (df)


# Run correlogram to identify highly features
def correlogram(df):
    corr = df.corr()
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True 
    corr[mask] = np.nan 
    return (corr
     .style
     .background_gradient(cmap='coolwarm', axis=None, vmin=-1, vmax=1)
     .highlight_null(null_color='#f1f1f1')  # Color NaNs grey
     .set_precision(3))
 

# Alternative correlogram to identify highly features
def corr(df):
    corr = X.corr()
    return corr.style.background_gradient(cmap='coolwarm').set_precision(3)


# Get highly correlated features 
def correlated_features(df):
    corr_max = 0.025
    corr_min = -0.025
    df2 = df.corr().unstack().reset_index()
    
    display(df2[(df2[0] != 1) & ((df2[0]>corr_max) | (df2[0]<corr_min))])
  

 # Vizualize distribution of classes 
def flag_distribution (df):
    display(df['flag'].value_counts())
    
    label = df[['flag']].groupby(['flag']).size().to_frame().reset_index()
    illicit_count = label[label['flag'] == 1]
    licit_count = label[label['flag'] == 0]

    fig = go.Figure(data = [
        go.Bar(name="Licit",y=licit_count[0],marker = dict(color = 'rgba(60, 179, 113, 0.6)',
            line = dict(
                color = 'rgba(58, 190, 120, 1.0)',width=1))),
        go.Bar(name="Illicit",y=illicit_count[0],marker = dict(color = 'rgba(255, 0, 0, 0.6)',
            line = dict(
                color = 'rgba(246, 78, 139, 1.0)',width=1)))])

    fig.update_layout(
        margin=dict(l=30, r=20, t=60, b=20), width=700, height = 400,
        title={'text': "Distribution of Classes",
               'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
        yaxis_title="Number of Labels",
        legend_title="Legend")
    py.iplot(fig)   


# Split into X/y train/test dataframes 
def split_df_train_test(df, target_variable, columns_to_drop, test_size):
    # Drop ids from training dataframe
    train = df.drop(columns_to_drop, axis=1)

    # Split between features and target
    X = train.drop([target_variable], axis = 1)
    y = train[target_variable]

    # Reconvert X to dataframe from numpy array
    X = pd.DataFrame(X)
    
    # Split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train_original = X_train
    
    return (X_train_original, X_train, X_test, y_train, y_test)


# Preprocess dataframe
def preprocess_dataframe(X_train, X_test, y_train):
    # Balance training data
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    y_train.value_counts()
    
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    return (X_train, X_test, y_train)


# Function to reweight probabilities
def reweight_proba(pi,q1=0.5,r1=0.5):
    r0 = 1-r1
    q0 = 1-q1
    tot = pi*(q1/r1)+(1-pi)*(q0/r0)
    w = pi*(q1/r1)
    w /= tot
    return w


# Train model
def train_model(algorithm, X_train, y_train):
    # obtain prediction by using a cross-validation 10 folds
    # y_pred = cross_val_predict(algorithm, X_train, y_train, cv=KFold(n_splits=10, random_state=42, shuffle=True))
    # entrenamos el modelo definitivo
    model = algorithm.fit(X_train, y_train)
    return model


# Test model on test set
def test_model(model, X_test):
    y_test_pred = model.predict(X_test)
    return (y_test_pred)


# Assess de accuracy metrics
def evaluation(y_true, y_pred, metrics):
    res = {}
    for name, function in metrics.items():
        res[name] = function(y_true, y_pred)
    return res


# Evaluate the model out of sample
def model_evaluation(y_test, y_test_pred, metrics):
    # evaluate the model on the test set 
    return evaluation(y_test, y_test_pred, metrics)


# Prints a confusion matrix as a figure
def plot_confusion_matrix(cm, class_labels):
    df_cm = pd.DataFrame(cm, index = [i for i in class_labels],
                  columns = [i for i in class_labels])
    sns.set(font_scale=1)
    sns.heatmap(df_cm, annot=True, fmt='g', cmap='Blues')
    plt.xlabel("Predicted label")
    plt.ylabel("Real label")
    plt.show()
    
# Customized classification report including AUC
def class_report(y_true, y_pred, y_score=None, average='micro'):
    if y_true.shape != y_pred.shape:
        print("Error! y_true %s is not the same shape as y_pred %s" % (
              y_true.shape,
              y_pred.shape)
        )
        return

    lb = LabelBinarizer()

    if len(y_true.shape) == 1:
        lb.fit(y_true)

    #Value counts of predictions
    labels, cnt = np.unique(
        y_pred,
        return_counts=True)
    n_classes = len(labels)
    pred_cnt = pd.Series(cnt, index=labels)

    metrics_summary = precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels)

    avg = list(precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            average='weighted'))
    
    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index,
        columns=labels)
    support = class_report_df.loc['support']
    total = support.sum()
    class_report_df['avg / total'] = avg[:-1] + [total]
    class_report_df = class_report_df.T
    class_report_df['pred'] = pred_cnt
    class_report_df['pred'].iloc[-1] = total
    if not (y_score is None):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for label_it, label in enumerate(labels):
            fpr[label], tpr[label], _ = roc_curve(
                (y_true == label).astype(int),
                y_score[:, label_it])
            roc_auc[label] = auc(fpr[label], tpr[label])
        if average == 'micro':
            if n_classes <= 2:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                    lb.transform(y_true).ravel(),
                    y_score[:, 1].ravel())
            else:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                        lb.transform(y_true).ravel(),
                        y_score.ravel())
            roc_auc["avg / total"] = auc(
                fpr["avg / total"],
                tpr["avg / total"])
        elif average == 'macro':
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([
                fpr[i] for i in labels]
            ))
            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in labels:
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            # Finally average it and compute AUC
            mean_tpr /= n_classes
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["avg / total"] = auc(fpr["macro"], tpr["macro"])
        class_report_df['AUC'] = pd.Series(roc_auc)
    return class_report_df

# Plots ROC curve
def get_auc(y, y_pred_probabilities, class_labels, column =1, plot = True):
    fpr, tpr, _ = roc_curve(y == column, y_pred_probabilities[:,column],drop_intermediate = False)
    roc_auc = roc_auc_score(y_true=y, y_score=y_pred_probabilities[:,1])
    print ("AUC: ", roc_auc)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()    


# Grid search over different algorithm and create dictionary of scores (score = recall)
def run_models(X_train, X_test, y_train, y_test):
    SEED = 9876
    
    # instantiate algorithms
    models = [
              KNeighborsClassifier(),
              LogisticRegression(),
              RandomForestClassifier(),
              XGBClassifier()]

    # models parameters
    knn_params = {'n_neighbors': [5, 10, 20],
                      'weights': ['uniform', 'distance']}

    lr_params = {'penalty': ['l2', 'elasticnet'],
                 'C' : [0.1, 5, 10],
                 'solver': ['lbfgs', 'saga'],
                'max_iter': [10000]}

    randomforest_params = {'max_features': range(1, 7),
                      'n_estimators': [70, 100, 130],
                      'max_depth': range(3, 6),
                      'min_samples_leaf': range(5, 10),
                      'random_state': [SEED]}

    xgb_params = {'objective':['binary:logistic'],
                  'eval_metric':['auc', 'error'],
                  'learning_rate': [0.05, 1],
                  'max_depth': range(6, 10),
                  'lambda':[0.5, 1.5],
                  'use_label_encoder': [False],
                  'seed': [SEED]}

    params = [knn_params, lr_params, randomforest_params, xgb_params]
    names = ['KNN', 'LogisticRegression', 'RandomForest', 'XGB']
    scores = {}

    # run gridsearch
    for i, model in enumerate(models):
        print(f"Grid-Searching for model {names[i]}...")
        best_model = GridSearchCV(model, params[i], n_jobs=-1, cv=5, scoring='f1', verbose = 0)
        best_model.fit(X_train, y_train)
        print(f"Best model fitted")
        y_hat = best_model.predict(X_test)

        f1 = f1_score(y_test, y_hat)
        # assign best parameters to the models
        models[i].set_params(**best_model.best_params_)

        # print scores 
        scores[names[i]] = (best_model.best_estimator_, f1)
        print(f'{names[i]} chosen hyperparameters: {best_model.best_params_}')
        print(f'{names[i]} F1 score on train sample: {best_model.best_score_}')
        print(f'{names[i]} F1 score on test sample: {f1}\n')
    
    return(scores)      


# Ranking the accuracy of the models based on scores
def ranking_accuracy (scores):
    ranking = pd.DataFrame.from_dict(scores, orient='index').reset_index()

    ranking.columns = ['Model', 'Setting', 'F1 Score']
    ranking = ranking.sort_values(by='F1 Score', ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.suptitle("Performance on test set", fontsize=25)
    sns.barplot(x="F1 Score", y="Model", data=ranking, ax=ax, palette = 'Blues_r');
    
    return (ranking)
 

# Check performance of best model
def best_model_performance(ranking, model_number, class_labels, X_train, X_test, y_train, y_test):
    best_model = ranking.Setting[model_number].fit(X_train, y_train)
    y_train_proba = best_model.predict_proba(X_train)
    y_train_pred = best_model.predict(X_train)
    y_test_proba = best_model.predict_proba(X_test)
    y_test_pred = best_model.predict(X_test)

    # In sample accuracy
    print("In sample accuracy:")

    # Customized classification report including AUC
    report_with_auc = class_report(
        y_true = y_train, 
        y_pred = y_train_pred, 
        y_score = y_train_proba)
    print(report_with_auc)

    cm =  confusion_matrix(y_train_pred, y_train)
    plot_confusion_matrix(cm, class_labels)

    # Assess expected accuracy using AUC/ROC 
    get_auc(y_train, y_train_proba, class_labels, column=1, plot=True)
    
    # Out sample accuracy
    print("Out sample accuracy:")

    # Customized classification report including AUC
    report_with_auc = class_report(
        y_true = y_test, 
        y_pred = y_test_pred, 
        y_score = y_test_proba)
    print(report_with_auc) 

    cm =  confusion_matrix(y_test_pred, y_test)
    plot_confusion_matrix(cm, class_labels)

    # Assess expected accuracy using AUC/ROC 
    get_auc(y_test, y_test_proba, class_labels, column=1, plot=True)

 
# Get features importance 
def feature_imp(coef, names):
    imp = coef
    indexes = np.argsort(imp)[-15:]
    indexes = list(indexes)
    
    plt.barh(range(len(indexes)), imp[indexes], align='center')
    plt.yticks(range(len(indexes)), [names[i] for i in indexes])
    plt.show()
   
    return indexes


# Check feature importance for best model
def feature_imp_best_model(X_train_original, model_number, ranking):
    feature_names = X_train_original.columns
    model_opt = ranking.Setting[model_number].fit(X_train, y_train)
    feat_indexes_model = feature_imp(abs(model_opt.feature_importances_), feature_names)

    display (feat_indexes_model)








