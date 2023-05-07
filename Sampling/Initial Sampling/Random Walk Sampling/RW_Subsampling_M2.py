###################################################################################################################################################
###################################################################################################################################################
# This script shows the process of creating 10 sampled graphs of a size of 50.000 (nodes)
# Here the idea is to represent the network as better as possible by taking in to account the time stamps
# and that multiple transactions can happen in the same time stamp
# The process developed in this script takes into account the following ideas:
## 1. Avoid self loops by deleting rows of the original data frame that has the same address as input and output
## 2. If an adresses sends BTC to another one in the same block multiple times, sum the total money sent to have an 'unique' transaction
## 3. If an adresses sends BTC to another one multiple times in different blocks, take into account that using nx.MultiDiGraph() (multiple edges 
#### between the same pair of addresses)
## 4. Following the concept of trans2vec from Wu et al (2022) [https://arxiv.org/ftp/arxiv/papers/1911/1911.09259.pdf] the edges are weighted by
### a mix between the amount and the time stap (hihger values when timestamp is more recent)
## 5. Finally the sampled method follows the description of random walk sampling based methods from Chen et al (2021) https://dl.acm.org/doi/10.1145/3398071


Note: Given the size of the data to run sucessfully this notebook a Computer with over 8GB of RAM should be used
###################################################################################################################################################
###################################################################################################################################################

# %%
# Load the libraries

import pandas as pd
import numpy as np

import os
import pickle
import time
import networkx as nx
import random
import Graph_Sampling as gs
from datetime import datetime

# %%
# Read the dataframe
df = pd.read_csv('/data/trans_3w.csv')
print(df.shape)
print(df.info())

# %%
# Delete self loops (same sender and receiver)
df = df[df['input_address'] != df['output_address']]
print(df.shape)

# %%
# Group multiple transactions in the same block. If a pair of addresses has the are involve in different transactions
# under the same block, work with them as an unique transaction
df = df.groupby(['input_address', 'output_address', 'block_index'])['ammount'].sum().reset_index()
print(df.shape)

# %%
# Creation of the normalized transfered funds (the sum over out-coming neighbors is 1)
df['total_transfered'] = df.groupby(['input_address', 'output_address'])['ammount'].transform('sum')
df['ammount_n'] =  df['ammount'] / df['total_transfered']


# %%
# Creation of a measure in which the mosts recent block has a higher value
# Creation of a timestamp values
df['timestap'] = df['block_index'] - df['block_index'].min() + 1

# Creation of the normalized timestap
df['total_timestap'] = df.groupby(['input_address', 'output_address'])['timestap'].transform('sum')
df['timestap_n'] =  df['timestap'] / df['total_timestap']

# %%
# Creation of the search bias parameter with alpha = 0.5
df['search_b'] = df['ammount_n'] * 0.5 + df['timestap_n'] * 0.5 

# %%
# Keep only important features and save the dataframe
df = df[['input_address', 'output_address', 'ammount_n', 'timestap_n', 'search_b']].copy()
df.to_csv('outputs/processed_df.csv')

# %%
# Create multi directed and weighted graph most suitable for our dataset
start = datetime.now()
Direct_Weight_G = nx.MultiDiGraph()
Direct_Weight_G.add_weighted_edges_from(df[['input_address', 'output_address', 'search_b']].values)
del df
print(datetime.now() - start)

# %%
# largest weakly conneceted component
start = datetime.now()
giant = max(nx.weakly_connected_components(Direct_Weight_G), key=len)
print(len(giant))
print(datetime.now() - start)

# %%
# save the set that contains the information of the largets connected component
with open('output/s/lcc_set.pickle', 'wb') as handle:
    pickle.dump(giant, handle, protocol=pickle.HIGHEST_PROTOCOL)

del handle

# %%
# Load the set with the nodes in the largest weakly connected component
# giant = pickle.load(open('outputs/lcc_set.pickle','rb'))

# %%
# Create the graph that is the largest weekly connected component of the full data
start = datetime.now()
G_giant = Direct_Weight_G.subgraph(giant).copy()
del Direct_Weight_G
del giant
print(datetime.now() - start)

# %%
# Trying to save the the graph
nx.write_gml(G_giant, 'outputs/lwcc_grahp.gml')


# %%
# A function that does the random walk sampling. Unfortunatly this function receives a directed graph and the undirected version of that graph
# this is for creating the walk not only with the outcoming neighbors but also incoming ones as in Chen et al (2021) https://dl.acm.org/doi/10.1145/3398071


def sampling_the_graph(graph_d, graph_u, n_nodes):
    
    selected_nodes = []
    
    list_of_nodes = [node for node in graph_u.nodes()]
    first_node = random.choice(list_of_nodes)
    selected_nodes.append(first_node)
    
    while len(selected_nodes) != n_nodes:
        #print(len(selected_nodes))
        next_candidates =  [node for node in graph_u.neighbors(selected_nodes[-1])]
        #int(len(next_candidates))
        if next_candidates != []:
            next_node = random.choice(next_candidates)
            if next_node not in selected_nodes:
                selected_nodes.append(next_node)
            else: 
                next_node = random.choice(list_of_nodes)
                if next_node not in selected_nodes:
                    selected_nodes.append(next_node)
                else:
                    pass
        else:
            next_node = random.choice(list_of_nodes)
            if next_node not in selected_nodes:
                selected_nodes.append(next_node)
            else:
                pass
        
    return graph_d.subgraph(selected_nodes).copy()
    #return selected_nodes
        
# %%
# Get the undirected version of the graph
start = datetime.now()
G_giant_u = G_giant.to_undirected().copy()    
print(datetime.now() - start)

# %%
# Ten different random walks of 50.000
start = datetime.now()

for iter_ in range(10):
    g_sampled = sampling_the_graph(G_giant, G_giant_u, 50000)
    print('Number of edges:', g_sampled.number_of_nodes())
    print('Number of edges:', g_sampled.number_of_edges())
    
    # Save the graph
    nx.write_gml(g_sampled, path='outputs/g_sb_'  + str(iter_ + 1) + '.gml')

print(datetime.now() - start)
