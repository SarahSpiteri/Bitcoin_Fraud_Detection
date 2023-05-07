
import pandas as pd
import numpy as np
import networkx as nx


from node2vec import Node2Vec


def df_node2vec(df, fraudulent_nodes):
    """
    This function takes a data frame (of the graph (source and target) and a list of fraudulent nodes to create the embbeding using node2vc)
    """
    G = nx.from_pandas_edgelist(df[['source', 'target']])
    
    # Adjustment to make embedding work
    for u,v,d in G.edges(data=True):
        d['weight'] = 1
        
    # Node2Vec Embedding from Leskovec‬ - Use parameters set in trans2vec paper
    trans2vec = Node2Vec(G, dimensions=64, p=0.25, q=0.75, walk_length=5, num_walks=20, seed=42, workers=7, quiet=True)
    model = trans2vec.fit(window=10)
    
    # Represent embedding as DataFrame
    trans2vec = (pd.DataFrame([model.wv.get_vector(str(n)) for n in G.nodes()]))
    trans2vec['address'] = list(G.nodes)
    
    
    # Include the label of the addresses
    trans2vec['label'] = np.where(trans2vec['address'].isin(fraudulent_nodes), 1, 0)
    trans2vec.set_index('address', inplace=True)
    
    return trans2vec