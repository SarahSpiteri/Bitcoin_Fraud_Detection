###################################################################################################################################################
###################################################################################################################################################
# This script shows the process of creating 5 sampled graphs of a size of 10K, 20K, 30K, 40K and 50K (nodes)
# Here the initital dataframe is used as input in the creation of an graph object using the library Networkx
# is this case the graph is created with the class nx.DiGraph() that when the same edge is given twice as it 
# happens many times in the data, then the graph is created with the last one wich represents a loss of information 

# The process of sampling the walk tries to follow the description of Chen et al (2021) https://dl.acm.org/doi/10.1145/3398071


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
start = datetime.now()

df = pd.read_csv('data/trans_3w.csv')
print(df.shape)
print(df.info())
df.head()

# Convert date to the right format
df['block_time'] = pd.to_datetime(df['block_time'], format="%Y-%m-%d %H:%M:%S")
print(datetime.now() - start)

# %%
# Create directed and weighted graph most suitable for our dataset
start = datetime.now()
Direct_Weight_G = nx.DiGraph()
Direct_Weight_G.add_weighted_edges_from(df[['input_address', 'output_address', 'ammount']].values)
del df
print(datetime.now() - start)

# %%
# largest component (is also the largest weakly conneected component)
start = datetime.now()
giant = max(nx.connected_components(Direct_Weight_G.to_directed()), key=len)
print(len(giant))
print(datetime.now() - start)

# %%
# save the set that contains the information of the largets connected component
with open('ouputs/lcc_set.pickle',pip 'wb') as handle:
    pickle.dump(giant, handle, protocol=pickle.HIGHEST_PROTOCOL)

del handle

# %%
# Load the set with the nodes in the largest weakly connected component
giant = pickle.load(open('outputs/lcc_set.pickle','rb'))

# %%
# Create the graph that is the largest weekly connected component of the full data
start = datetime.now()
G_giant = Direct_Weight_G.subgraph(giant).copy()
del Direct_Weight_G
del giant
print(datetime.now() - start)

# %%
# Trying to save the the graph
#nx.write_gpickle(G_giant, 'ouputs/lwcc_grahp.gpickle')
nx.write_gml(G_giant, 'outputs/lwcc_grahp.gml')


# %%
# A function that does the random walk sampling
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
    
    
# %%
# Create the undirected version of the graph, to run the random walk with more neihgbors
start = datetime.now()
G_giant_u = G_giant.to_undirected().copy()
print(datetime.now() - start)


# %%
# Performing random walk subsampling of 10000 nodes
start = datetime.now()
g_1 = sampling_the_graph(G_giant,G_giant_u, 10000)
print('Number of edges:', g_1.number_of_nodes())
print('Number of edges:', g_1.number_of_edges())

# Save the graph
nx.write_gml(g_1, 'outputs/g_s1.gml')

print(datetime.now() - start)


# %%
# Performing random walk subsampling of 20000 nodes
start = datetime.now()
g_2 = sampling_the_graph(G_giant,G_giant_u, 20000)
print('Number of edges:', g_2.number_of_nodes())
print('Number of edges:', g_2.number_of_edges())

# Save the graph
nx.write_gml(g_2, 'outputs/g_s2.gml')

print(datetime.now() - start)

# %%
# Performing random walk subsampling of 30000 nodes
start = datetime.now()
g_3 = sampling_the_graph(G_giant,G_giant_u, 30000)
print('Number of edges:', g_3.number_of_nodes())
print('Number of edges:', g_3.number_of_edges())

# Save the graph
nx.write_gml(g_3, 'outputs/g_s3.gml')

print(datetime.now() - start)

# %%
# Performing random walk subsampling of 40000 nodes
start = datetime.now()
g_4 = sampling_the_graph(G_giant,G_giant_u, 40000)
print('Number of edges:', g_4.number_of_nodes())
print('Number of edges:', g_4.number_of_edges())

# Save the graph
nx.write_gml(g_4, 'outputs/g_s4.gml')

print(datetime.now() - start)

# %%
# Performing random walk subsampling of 50000 nodes
start = datetime.now()
g_5 = sampling_the_graph(G_giant,G_giant_u, 50000)
print('Number of edges:', g_5.number_of_nodes())
print('Number of edges:', g_5.number_of_edges())

# Save the graph
nx.write_gml(g_5, 'outputs/g_s5.gml')

print(datetime.now() - start)
