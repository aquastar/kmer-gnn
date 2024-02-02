import pandas as pd
import torch
from torch_geometric.data import Data
from multiprocessing import Pool
import os

def split_by_antibiotic(data, antibiotic_names):

    datasets = {}
    for antibiotic in antibiotic_names:
        datasets[antibiotic] = data[data['Antibiotic'] == antibiotic].drop(columns=['Antibiotic'])
        datasets[antibiotic].reset_index(drop=True, inplace=True)
    
    return datasets

def graph_creation(adj, node_index, threshold=None):

    if threshold:
        edges = []
        edge_attr = []
        for i in adj.columns:
            for j in adj.columns:
                if j != i and adj.loc[i, j] >= threshold:
                    edges.append([node_index.index(i), node_index.index(j)])
                    edge_attr.append([1])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edges = []
        edge_attr = []
        for i in adj.columns:
            for j in adj.columns:
                if j != i:
                    edges.append([node_index.index(i), node_index.index(j)])
                    edge_attr.append([adj.loc[i, j]/1])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return edge_index, edge_attr


