import random
import numpy as np
import networkx as nx
from stellargraph import StellarGraph, StellarDiGraph

# Random node sampling
class RandomNodeSampler:
    def __init__(self, graph, fraud_sample, licit_sample, number_of_nodes):
        self.number_of_nodes = number_of_nodes
        self.fraud_sample = fraud_sample
        self.licit_sample = licit_sample 
        
    def sampler(self, graph, fraud_sample, licit_sample, number_of_nodes):
        s_size = self.number_of_nodes - len(fraud_sample) # Number of licit nodes to sample
        rd_nodes = random.sample(self.licit_sample, s_size) # Randomly select licit of nodes
        rd_nodes = list(set(rd_nodes + self.fraud_sample))  # Get full list of nodes
        g_sampler = graph.subgraph(rd_nodes) # Obtain sub-graph from randomly selected nodes
        return g_sampler
        
    def sample(self, graph, fraud_sample, licit_sample, number_of_nodes):
        new_graph = self.sampler(graph, self.fraud_sample, self.licit_sample, self.number_of_nodes) # Get randomly sampled graph
        # Edge condition: Number of Edges >= Number of Nodes (Ensures graph is not too sparse)
        while new_graph.number_of_edges() < self.number_of_nodes: # If condition not met
            new_graph = self.sampler(graph, self.fraud_sample, self.licit_sample, self.number_of_nodes) # Get new sub-graph
            if new_graph.number_of_edges() >= self.number_of_nodes:
                        break
        return new_graph

# Random walk based sampling method
def rw_sampling(graph, n_nodes):
    selected_nodes = []
    
    list_of_nodes = [node for node in graph.nodes()]
    first_node = random.choice(list_of_nodes)
    selected_nodes.append(first_node)
    

    while len(selected_nodes) != n_nodes:
        next_candidates =  [node for node in graph.neighbors(selected_nodes[-1])] # Neigbors of the last node
        if next_candidates != []:
            next_node = random.choice(next_candidates) # Select between neighbors
            if next_node not in selected_nodes:
                selected_nodes.append(next_node)
            else: 
                while len(next_candidates) > 1:
                    next_candidates.remove(next_node)  # If next_node is problematic remove it from the neigbors
                    next_node = random.choice(next_candidates) # Select between neighbors
                    if next_node not in selected_nodes:
                        selected_nodes.append(next_node)
                        break
                    else:
                        pass
                    
                next_node = random.choice(list_of_nodes) # once it has tested all the neigbors without sucess go for other
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

    return graph.subgraph(selected_nodes)
    

# Mixed method (hhannes idea with random walked)
def mixed_sampling(graph, fraud_sample, number_of_nodes):
    
    selected_nodes = []
    
    for i in fraud_sample:                                                 # for each fraudulent/high-risk node
        fr_neighbors = [node for node in graph.neighbors(i)]    # find neighbors
        if fr_neighbors != []:                                             # if the node has neighbors
            fr_adj = random.choice(fr_neighbors)                           # choose 1 neighbor
            if fr_adj not in fraud_sample:                                 # check if neighbor isn't already included
                if fr_adj not in selected_nodes:
                    selected_nodes.append(fr_adj)                          # add to selection
            else:                                                          # if selection not made
                if len(fr_neighbors) > 1:                                  # if node has more than 1 neighbor
                    fr_neighbors.remove(fr_adj)                            # remove previous invalid choice as neighbor
                    fr_adj2 = random.choice(fr_neighbors)                  # choose new neighbor
                    if fr_adj2 not in fraud_sample:                        # check if already included
                        if fr_adj2 not in selected_nodes:
                            selected_nodes.append(fr_adj2)                 # if not included, add neighbor as selection
                else:
                    fraud_sample.remove(i)                                 # if no neighbor found remove node (reduces isolates)
    
    selected_nodes.extend(fraud_sample)                                    # combine fraud and selected neighbors list
    list_of_nodes = list(set(list(graph.nodes())) - set(selected_nodes)) # determine amount required to reach the set number of nodes
    
    first_node = random.choice(list_of_nodes)                              # randomly choose starting node
    selected_nodes.append(first_node)                                      # add node to selection
    
    while len(selected_nodes) != number_of_nodes:                          # until the set number of nodes is reached
                                                                           # follow similar neighbor sampling procedure as before
        next_candidates =  [node for node in graph.neighbors(selected_nodes[-1])] 
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
        
    return graph.subgraph(selected_nodes)


# Neighbors of a set of root nodes  (licit and fraudulent ones)
def neighbors_sampling(graph, fraud_list, licit_list, ini_size):    
    r_nodes = []
    
    # Fraudulent nodes part
    f_nodes = []
    rs_fraud_list = random.sample(fraud_list, ini_size)
    for node in rs_fraud_list:
        r_nodes.append(node)
        f_nodes.append(graph.neighbors(node))

    f_nodes = [x for xs in f_nodes for x in xs] # flattened list    
    f_nodes = list(set(f_nodes)) # Unique nodes
    
    # Non-fraudulent nodes part    
    l_nodes = []
    rs_nfraud_list = random.sample(licit_list, ini_size)
    for node in rs_nfraud_list:
        r_nodes.append(node)
        l_nodes.append(graph.neighbors(node))

    l_nodes = [x for xs in l_nodes for x in xs] # flattened list    
    l_nodes = list(set(l_nodes)) # Unique nodes
    
    t_nodes = list(set(f_nodes + l_nodes + r_nodes))
    
    return graph.subgraph(t_nodes)

class DegreeBasedSampler:
    def __init__(self, graph, g_licit, fraud_sample, licit_sample, number_of_nodes, p_distribution):
        self.number_of_nodes = number_of_nodes
        self.fraud_sample = fraud_sample
        self.licit_sample = licit_sample
        self.p_distribution = p_distribution

    def sampler(self, graph, g_licit, fraud_sample, licit_sample, number_of_nodes, p_distribution):
        s_size = self.number_of_nodes - len(fraud_sample) # Number of licit nodes to sample
        
        # Sample nodes and create sub-graph
        sampled_nodes = np.random.choice(self.licit_sample, s_size, replace=False, p=self.p_distribution)
        sampled_nodes = list(sampled_nodes) + list(self.fraud_sample)
        g_sampler = graph.subgraph(sampled_nodes)
        return g_sampler
    
    def sample(self, graph, g_licit, fraud_sample, licit_sample, number_of_nodes, p_distribution):
        new_graph = self.sampler(graph, g_licit, self.fraud_sample, self.licit_sample, self.number_of_nodes,
                                 self.p_distribution) # Get randomly sampled graph
        # Edge condition: Number of Edges >= Number of Nodes (Ensures graph is not too sparse)
        while new_graph.number_of_edges() < self.number_of_nodes: # If condition not met
            new_graph = self.sampler(graph, g_licit, self.fraud_sample, self.licit_sample, self.number_of_nodes, 
                                     self.p_distribution) # Get new sub-graph
            if new_graph.number_of_edges() >= self.number_of_nodes:
                        break
        return new_graph
