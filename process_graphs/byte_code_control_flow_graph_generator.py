import os
import collections
from os.path import join
from copy import deepcopy

import json
import dgl
from numpy import source
import torch
import networkx as nx
from tqdm import tqdm


EDGE_DICT = {('None', 'None', 'None'): '0', ('None', 'None', 'orange'): '1', ('Msquare', 'None', 'gold'): '2', ('None', 'None', 'lemonchiffon'): '3', ('Msquare', 'crimson', 'crimson'): '4', ('None', 'None', 'crimson'): '5', ('Msquare', 'crimson', 'None'): '6', ('Msquare', 'crimson', 'lemonchiffon'): '7'}
DRY_RUNS = 0

def creat_node_label(g,d):
    nodes = g.nodes()
    int2label = {}
    for idx,node_idx in enumerate(nodes):
        obj = g._node[node_idx]
        if 'shape' not in obj:
            shape = 'None'
        else:
            shape = obj['shape']
        if 'color' not in obj:
            color = 'None'
        else:
            color = obj['color']
        if 'fillcolor' not in obj:
            fillcolor = 'None'
        else:
            fillcolor = obj['fillcolor']
        t = (shape,color,fillcolor)
        node_type = d[t]
        int2label[node_idx] = node_type
    return int2label


def creat_edge_label(g):
    edgeLabel = {}
    edgedata = g.edges.data()
    for u,v,fe in edgedata:
        edgeLabel[(u, v, 0)] = {'edge_type': 'CF'}
    return edgeLabel


def dot2gpickle(dot_file, gpickle_file):
    source_file = gpickle_file.split('/')[-1]
    nx_g = nx.drawing.nx_pydot.read_dot(dot_file)
    node_lables = creat_node_label(nx_g, EDGE_DICT)
    edge_labels = creat_edge_label(nx_g)
    nx.set_node_attributes(nx_g, node_lables, name='node_type')
    nx.set_node_attributes(nx_g, source_file, name='source_file')
    nx.set_edge_attributes(nx_g, edge_labels)
    nx.write_gpickle(nx_g, gpickle_file)


def merge_byte_code_cfg(source_path, graph_list, output):
    merged_graph = None
    for graph in graph_list:
        nx_graph = nx.read_gpickle(join(source_path, graph))
        if merged_graph is None:
            merged_graph = deepcopy(nx_graph)
        else:
            merged_graph = nx.disjoint_union(merged_graph, nx_graph)
    nx.write_gpickle(merged_graph, output)


def travelsalDir(filepath):
    count_0, count_1 = 0, 0
    with open(filepath,'r') as f:
        load_dict = json.load(f)
    location = './data/bytecode_cfg_set/'

    pathDict = {}
    labels = collections.defaultdict(int)
    name_list = []
    for name in load_dict:
        if load_dict[name] == 0 and count_0 < 100:
            count_0 += 1
            labels[name] = torch.LongTensor([load_dict[name]])
            pathDict[name] = location + name + '.dot'
            name_list.append(name)
        elif load_dict[name] == 1 and count_1 < 100:
            count_1 += 1
            labels[name] = torch.LongTensor([load_dict[name]])
            pathDict[name] = location + name + '.dot'
            name_list.append(name)

    return pathDict, labels, name_list


def format_label(label_file, output):
    with open(label_file, 'r') as f:
        labels = json.load(f)
    new_labels = []
    for contract, target in labels.items():
        new_labels.append({'targets': target, 'contract_name': contract+'.sol'})
    with open(output, 'w') as f:
        json.dump(new_labels, f)


def forencis_gpickle(graph_path):
    nx_graph = nx.read_gpickle(graph_path)
    print(nx_graph.is_multigraph())
    print(nx_graph.nodes()[0])
    for idx, node in nx_graph.nodes(data=True):
        print(idx, node['node_type'])
    print(list(nx_graph.edges.data())[0])
    # print(nx_graph.edges.data())
    # for source, target, data in list(nx_graph.edges(data=True)):
        # print(source, target, data)


if __name__ == '__main__':
    graph_path = '../HAN_DGL/data/bytecode_cfg_set'
    gpickle_path = './experiments/ge-sc-data/byte_code/ethor/'
    merge_graph_output_path = './experiments/ge-sc-data/byte_code/ethor/compressed_graphs'
    compressed_graph = join(merge_graph_output_path, 'cfg_compressed_graphs.gpickle')
    os.makedirs(merge_graph_output_path, exist_ok=True)
    dot_files = [f for f in os.listdir(graph_path) if f.endswith('.dot')]
    print(len(dot_files))
    SCALE = DRY_RUNS if DRY_RUNS else len(dot_files)

    # # Convert dot to gpickle
    # for dot in dot_files[:SCALE]:
    #     dot2gpickle(join(graph_path, dot), join(gpickle_path, dot.replace('.dot', '.sol')))

    # # Merge gpickle files
    # gpickle_files = [f for f in os.listdir(gpickle_path) if f.endswith('.sol')]
    # merge_byte_code_cfg(gpickle_path, gpickle_files, compressed_graph)

    # Forencis gpickle graph
    source_compressed_graph = './experiments/ge-sc-data/source_code/access_control/clean_57_buggy_curated_0/cfg_compressed_graphs.gpickle'
    # forencis_gpickle(source_compressed_graph)
    forencis_gpickle(compressed_graph)