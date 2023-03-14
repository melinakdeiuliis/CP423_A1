import argparse
import gzip 
import numpy as np
import argparse
def summation(node,pageranks,nodes):
    x=10 
    return
def read_graph(filename):
    edges = []
    nodes = set()
    with gzip.open(filename, 'rt') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) == 2:
                node1, node2 = tokens
                edges.append((node1, node2))
                nodes.add(node1)
                nodes.add(node2)
    return edges, nodes

def page_rank(max,lamb,thr,nodes):
    G_edges,G_nodes=read_graph('web-Stanford.txt.gz')
    node_index = {node: i for i, node in enumerate(nodes)}
    edges = []
    for node1, node2 in G_edges:
        if node1 in nodes and node2 in nodes:
            edges.append((node1, node2))
    print(edges)
    row, col = zip(*[(node_index[node1], node_index[node2]) for node1, node2 in edges])
    data = np.ones(len(row))
    A = np.coo_matrix((data, (row, col)), shape=(len(nodes), len(nodes)))


    col_sum = np.array(A.sum(axis=0)).flatten()
    col_sum[col_sum == 0] = 1  
    D = np.reciprocal(col_sum)
    D[np.isinf(D)] = 0 
    D = np.sqrt(D)
    AD = A.multiply(D).transpose().multiply(D)
    x = np.ones(len(nodes)) / len(nodes)

    for i in range(max):
        x_new = lamb * AD.dot(x) + (1 - lamb) / len(nodes)
        if np.linalg.norm(x_new - x, ord=1) < thr:
            break
        x = x_new

    for node, rank in zip(nodes, x):
        print(f"{node}: {rank}")
parser = argparse.ArgumentParser()
parser.add_argument("--maxiteration",type=int)
parser.add_argument("--lamb",type=float)
parser.add_argument("--thr",type=float)
parser.add_argument("--nodes")
args = parser.parse_args()
values=args.nodes.split(',')
page_rank(args.maxiteration,args.lamb,args.thr,values)
 
