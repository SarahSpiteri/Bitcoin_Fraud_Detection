# Requirements
import random
import numpy as np
import pandas as pd
import networkx as nx

import scipy.sparse as sp
from gensim.models import Word2Vec
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


class BiasedRandomWalker:
    """Biased second order random walks in Node2Vec.
    Parameters:
    -----------
    walk_number (int): Number of random walks. Default is 10.
    walk_length (int): Length of random walks. Default is 80.
    p (float): Return parameter (1/p transition probability) to move towards from previous node.
    q (float): In-out parameter (1/q transition probability) to move away from previous node.
    """

    def __init__(self, walk_length: int = 80, walk_number: int = 10, p: float = 0.5, q: float = 0.5):
        self.walk_length = walk_length
        self.walk_number = walk_number
        try:
            _ = 1 / p
        except ZeroDivisionError:
            raise ValueError("The value of p is too small or zero to be used in 1/p.")
        self.p = p
        try:
            _ = 1 / q
        except ZeroDivisionError:
            raise ValueError("The value of q is too small or zero to be used in 1/q.")
        self.q = q

    def random_choice(self, arr, p):
        return arr[np.searchsorted(np.cumsum(p), np.random.random(), side="right")]    
    
    def walk(self, graph: sp.csr_matrix):
        data = graph.data
        indices = graph.indices
        indptr = graph.indptr
        walk_length = self.walk_length
        walk_number = self.walk_number

        def random_walk():
            N = len(indptr) - 1
            for _ in range(walk_number):
                nodes = np.arange(N, dtype=np.int32)
                np.random.shuffle(nodes)
                for n in nodes:
                    walk = [n]
                    current_node = n
                    for _ in range(walk_length - 1):
                        neighbors = indices[indptr[current_node]:indptr[current_node + 1]]
                        if neighbors.size == 0:
                            break

                        probability = data[indptr[current_node]: indptr[current_node + 1]].copy()
                        norm_probability = probability / np.sum(probability)
                        current_node = self.random_choice(neighbors, norm_probability)
                        walk.append(current_node)
                    yield walk

        walks = [list(map(str, walk)) for walk in random_walk()]
        return walks

    
    
class trans2vec(object):
    def __init__(self, DataFrame, alpha=0.5, p=0.5, q=0.5, dimensions=64, num_walks=20, walk_length=5,
                 window_size=10, workers=1, seed=2022):
        self.alpha = alpha
        self.p = p
        self.q = q
        self.dimensions = dimensions
        self.window_size = window_size
        self.workers = workers
        self.seed = seed
        self.walk_length = walk_length
        self.num_walks = num_walks
        
        # DataFrame must include: source and target nodes, transaction amounts, and timestamps
        self.DataFrame = DataFrame

        self.walks = None
        self.word2vec_model = None
        self.embeddings = None
        self.do()

    def do(self):
        self.load_data()
        self.walk()

    def load_data(self):
        # Make Sure that all column labels match
        self.G = nx.from_pandas_edgelist(self.DataFrame, 'source', 'target', create_using=nx.MultiDiGraph())
        self.adj_matrix = nx.to_scipy_sparse_matrix(self.G) # to match graph: sp.csr_matrix
        self.amount_data = self.DataFrame['weight'].values # weight is the amount of BTC
        self.timestamp_data = self.DataFrame['timestamp'].values # 
        self.adj_matrix.data = self.get_amount_timestamp_data()
        
    def normalized_probs(self, unnormalized_probs):
        if len(unnormalized_probs) > 0:
            global normalized_probs
            normalized_probs = unnormalized_probs / unnormalized_probs.sum()
        return normalized_probs

    def combine_probs(self, p1, p2, alpha):
        probs1 = self.normalized_probs(p1)
        probs2 = self.normalized_probs(p2)
        assert len(probs1) == len(probs2), "combine_probs invalid"
        combine_probs = np.multiply(np.power(probs1, alpha), np.power(probs2, 1 - alpha))
        return combine_probs

    def get_amount_timestamp_data(self):
        """Preprocessing transition probability: alpha * TBS * (1-alpha) * WBS
            refer to <https://ieeexplore.ieee.org/document/9184813>
            Returns
            -------
            amount_timestamp_data.data : sp.csr_matrix.data
        """
        self.N = self.adj_matrix.shape[0]
        amount_timestamp_data = sp.lil_matrix((self.N, self.N), dtype=np.float64)
        nodes = np.arange(self.N, dtype=np.int32)
        indices = self.adj_matrix.indices
        indptr = self.adj_matrix.indptr
        amount_data = self.amount_data.data
        timestamp_data = self.timestamp_data.data
        for node in nodes:
            nbrs = indices[indptr[node]: indptr[node + 1]]
            nbrs_amount_probs = np.asarray(amount_data[indptr[node]: indptr[node + 1]]).copy()
            nbrs_timestamp_probs = np.asarray(timestamp_data[indptr[node]: indptr[node + 1]]).copy()
            nbrs_unnormalized_probs = self.combine_probs(nbrs_amount_probs, nbrs_timestamp_probs, self.alpha)

            for i, nbr in enumerate(nbrs):
                amount_timestamp_data[node, nbr] = nbrs_unnormalized_probs[i]
        amount_timestamp_data = amount_timestamp_data.tocsr()
        return amount_timestamp_data.data

    def walk(self):
        walks = BiasedRandomWalker(walk_length=self.walk_length, walk_number=self.num_walks, p=self.p,
                                  q=self.q).walk(self.adj_matrix)
        word2vec_model = Word2Vec(sentences=walks, vector_size=self.dimensions, window=self.window_size,
                                  min_count=0, sg=1, hs=1, workers=self.workers, seed=self.seed)
        # embeddings = word2vec_model.wv.vectors[np.fromiter(map(int, word2vec_model.wv.index_to_key), np.int32).argsort()]
        embeddings = (pd.DataFrame([word2vec_model.wv.get_vector(str(n)) for n in range(self.N)]))
        embeddings['address'] = list(self.G.nodes)
        self.walks = walks
        self.word2vec_model = word2vec_model
        self.embeddings = embeddings

        
def df_trans2vec(df, fraudulent_nodes):
    """
    This function takes a data frame (of the graph (source and target) and a list of fraudulent nodes to create the embbeding using trans2vc)
    """
    
    t2v = trans2vec(df)
    trans_emb = t2v.embeddings

    trans_emb['label'] = np.where(trans_emb['address'].isin(fraudulent_nodes), 1, 0)
    trans_emb.set_index('address', inplace=True)
    
    return trans_emb