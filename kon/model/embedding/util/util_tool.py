# _*_ coding:utf-8 _*_
import os
import networkx as nx
import pandas as pd


def get_node_information(all_nodes):
    node2idx = {}
    idx2node = []
    node_size = 0
    for node in all_nodes:
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1
    idx2node = idx2node
    node2idx = node2idx
    return idx2node, node2idx

def save_edgelist(edgelist_list,save_path):
    if os.path.exists(save_path):
        os.remove(save_path)

    file=open(save_path,mode='a+')
    for edgelist in edgelist_list:
        file.writelines(edgelist)

def read_graph(edgelist_path='../wiki/Wiki_edgelist.txt'):
    DG=nx.read_edgelist(
        edgelist_path,
        create_using=nx.DiGraph(),
        nodetype=None,
        data=[('weight',int)]
    )

    return DG

def read_node_label(filename, skip_head=False):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        if skip_head:
            fin.readline()
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y

def read_label(label_path):
    data=pd.read_csv(label_path,header=None,sep=' ')
    nodes=data[0].tolist()
    label=data[1].tolist()

    return nodes,label

if __name__=='__main__':
    pass
    # edgelist_list=['1 2\n','4 3\n','5 6\n','2 3\n','2 1\n','3 5\n','1 2\n']
    # save_path='kon.txt'
    # save_edgelist(edgelist_list,save_path)
    # read_graph(save_path)

