"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023
"""

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

from models import GNN
from utils import sparse_mx_to_torch_sparse_tensor

# Initializes device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
hidden_dim = 32
output_dim = 4
dropout = 0.0
neighbor_aggr = 'mean'
readout = 'mean'


############## Task 4

Gs = [nx.cycle_graph(i) for i in range(10, 20)]


############## Task 5
        
adj = sp.block_diag([nx.adjacency_matrix(G) for G in Gs])

X = np.ones((adj.shape[0], 1))

idx = np.repeat(np.arange(len(Gs)), [G.number_of_nodes() for G in Gs])


adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
X = torch.FloatTensor(X).to(device)
idx = torch.LongTensor(idx).to(device)


############## Task 8

print("="*20)
print("Task 8")
for neighbor_aggr, readout in [("sum", "sum"), ("mean", "mean"), ("mean", "sum"), ("sum", "mean")]:
    print("neighbor_aggr: {}, readout: {}".format(neighbor_aggr, readout))
    model = GNN(1, hidden_dim, output_dim, neighbor_aggr, readout, dropout).to(device)
    print(model(X, adj, idx))
    print()
print("="*20)
############## Task 9
G1 = nx.union(nx.cycle_graph(3), nx.cycle_graph(3), rename=('G1-', 'G2-'))
G2 = nx.cycle_graph(6)
############## Task 10
Gs = [G1, G2]

adj = sp.block_diag([nx.adjacency_matrix(G) for G in Gs])

X = np.ones((adj.shape[0], 1))

idx = np.repeat(np.arange(len(Gs)), [G.number_of_nodes() for G in Gs])


adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
X = torch.FloatTensor(X).to(device)
idx = torch.LongTensor(idx).to(device)




############## Task 11
print("="*20)
print("Task 11")
model = GNN(1, hidden_dim, output_dim, neighbor_aggr="sum", readout="sum", dropout=dropout).to(device)
print(model(X, adj, idx))
print("="*20)