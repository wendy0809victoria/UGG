import argparse
import os
import random
import scipy as sp
import scipy.sparse as sparse
from tqdm import tqdm
from torch import Tensor
import networkx as nx
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import pandas as pd
import pickle as pk
import csv
from scipy.spatial.distance import cdist

def generate(gamma_1, gamma_2):
    G = pk.load(open('dataset/NBA/nba-G.pkl', 'rb'))
    graphs = pk.load(open('dataset/NBA/nba-0617-100.pkl', 'rb'))

    T_f, T_cf, adj_cf, edges_cf_t0, edges_cf_t1 = pk.load(open('dataset/NBA/nba_kcore2-euclidean30.0-mvgrl-auto.pkl', 'rb'))

    T_f = T_f.todense()
    T_f = torch.tensor(T_f)
    T_f = T_f.to(torch.long)

    T_cf = T_cf.todense()
    T_cf = torch.tensor(T_cf)
    T_cf = T_cf.to(torch.long)

    adj_cf = adj_cf.todense()
    adj_cf = torch.tensor(adj_cf)
    adj_cf = adj_cf.to(torch.long)

    idx_features_labels = pd.read_csv('dataset/NBA/nba.csv')
    user_id = np.array(idx_features_labels["user_id"], dtype=int)

    node_dict = {}
    for i in range(403):
        node_dict[int(user_id[i])] = i
    print(user_id[0])

    edges = np.loadtxt('dataset/NBA/nba_relationship.txt', dtype=np.int64)
    print(f'edges shape: {edges.shape}') # (16570, 2)
    print(edges[0][0])

    nba_adj = np.zeros((403, 403), dtype=np.int64)

    for i in range(edges.shape[0]):
        nba_adj[int(node_dict[int(edges[i][0])])][int(node_dict[int(edges[i][1])])] = 1
        nba_adj[int(node_dict[int(edges[i][1])])][int(node_dict[int(edges[i][0])])] = 1

    adj = torch.tensor(nba_adj)
    adj = adj.to(torch.long)           

    nba_src = []
    nba_dst = []
    bias_init = 0
    edge_init = 0
    for i in range(403):
        for j in range(i, 403):
            if int(adj[i][j]) == 1:
                nba_src.append(i)
                nba_dst.append(j)
                edge_init += 1
                if T_f[i][j] == 1:
                    bias_init += 1
    np.savetxt('edge_index-nba.txt', np.array([nba_src, nba_dst]), fmt='%d')
    print(f'edge_init, bias_init, and their ratio: {edge_init}, {bias_init}, {bias_init/edge_init}')

    train_idx = np.loadtxt('dataset/NBA/nba-train-idx.txt')
    val_idx = np.loadtxt('dataset/NBA/nba-val-idx.txt')
    test_idx = np.loadtxt('dataset/NBA/nba-test-idx.txt')

    whole_idx = []
    for i in range(len(train_idx)):
        whole_idx.append(train_idx[i])
    for i in range(len(val_idx)):
        whole_idx.append(val_idx[i])
    for i in range(len(test_idx)):
        whole_idx.append(test_idx[i])

    adjs = torch.load(f'adj-gen-nba-gamma1-{gamma_1}-gamma2-{gamma_2}.pt')

    data = torch.load('processed_data/NBA.pt')
    # data = torch.load('NBA.pt') # correct

    test_batch_idx = []

    for i in range(65):
        if i != 64:
            test_batch_idx.append(train_idx[5*i:5*(i+1)])
        else:
            test_batch_idx.append(train_idx[5*i:])
    for i in range(8):
        if i != 7:
            test_batch_idx.append(val_idx[5*i:5*(i+1)])
        else:
            test_batch_idx.append(val_idx[5*i:])
    for i in range(9):
        if i != 8:
            test_batch_idx.append(test_idx[5*i:5*(i+1)])
        else:
            test_batch_idx.append(test_idx[5*i:])

    for batch in range(82):
    # for batch in range(65):
        num = 0
        for idx in test_batch_idx[batch]:
            data_gen = data[int(idx)]
            prevA = data_gen.A[0]
            data_gen.A = adjs[batch][num]
            num += 1

    ##############################################################################################
    # Update Generated Graph
    print("Start generating...")
    bias_total = []
    bias = []
    ct_zero = 0
    visited = np.zeros((403, 403), dtype=np.int64)
    for idx in whole_idx:
        g = graphs[int(idx)]
        nodes = list(g.nodes)
        nodes_dict = {}
        for n in range(len(nodes)):
            nodes_dict[nodes[n]] = n
        src = []
        dst = []
        bias_gen = 0
        bias_init = 0
        edge_init = 0
        for i in range(len(nodes)):
            for j in range(i, len(nodes)):
                if visited[int(nodes[i])][int(nodes[j])] == 0:
                    if data[int(idx)].A[i][j] == 1:
                        src.append(i)
                        dst.append(j)
                    if data[int(idx)].A[i][j] == 1 and T_f[int(nodes[i])][int(nodes[j])] == 1:
                        bias_gen += 1
                    if adj[int(nodes[i])][int(nodes[j])] == 1 and T_f[int(nodes[i])][int(nodes[j])] == 1:
                        bias_init += 1
                    if adj[int(nodes[i])][int(nodes[j])] == 1:
                        edge_init += 1
                    adj[int(nodes[i])][int(nodes[j])] = data[int(idx)].A[i][j]
                    adj[int(nodes[j])][int(nodes[i])] = data[int(idx)].A[i][j]
                    visited[int(nodes[i])][int(nodes[j])] = 1
                    visited[int(nodes[j])][int(nodes[i])] = 1
        if len(src) == 0:
            ct_zero += 1
            bias_total.append(0)
        else:
            bias_total.append(bias_gen/len(src))
        if edge_init != 0:
            bias.append(bias_init/edge_init)
        else:
            bias.append(0)
    bias_diff = np.array(bias_total) - np.array(bias)
    print(f'bias_diff: {bias_diff}')
    print(f'Counting no edges: {ct_zero}')
    print(f'Bias_gen ratio mean: {np.array(bias_total).mean()}')
    print(f'Bias_gen ratio variance: {np.var(np.array(bias_total))}')
    print(f'Bias_init ratio mean: {np.array(bias).mean()}')
    print(f'Bias_init ratio variance: {np.var(np.array(bias))}')
    print(f'len(src): {len(src)}')
    print(f'edge_init: {edge_init}')
    print("Done generating...")

    #############################################################################################
    # Construct Generated Edge_index
    print("Start constructing...")
    src = []
    dst = []
    edge_num_final = 0
    edge_index = 0
    for i in range(0, 403):
        for j in range(i, 403):
            if adj[i][j] == 1:
                edge_num_final += 1
                src.append(i)
                dst.append(j)
                edge_index += 1     
    print("Done counting...")
    print(f'Final edge number: {edge_num_final}')
    edge_index_numpy = np.array([src, dst], dtype=np.int64)
    np.savetxt(f'edge_index-nba-ugg-gamma1-{gamma_1}-gamma2-{gamma_2}.txt', edge_index_numpy, fmt='%d')
    
gamma_1 = [1, 2, 5, 10, 20]
gamma_2 = [0.1, 0.2, 0.5, 0.75, 1]

# for i in range(5):
for i in range(1):
    # for j in range(5):
    for j in range(1):
        generate(gamma_1[i], gamma_2[j])
