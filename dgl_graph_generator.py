import torch
import dgl
import networkx as nx
import numpy as np

def add_node_type_feature(nx_graph):
    nx_g = nx_graph
    list_node_type = []
    node_type_feat_attrs = dict()
    for node, data in nx_graph.nodes(data=True):
        if data['node_type'] is not None:
            if data['node_type'] not in list_node_type:
                list_node_type.append(data['node_type'])
            node_type_feat = torch.tensor(list_node_type.index(data['node_type']))
            node_type_feat_attrs[node] = node_type_feat
            print(node_type_feat)

    nx.set_node_attributes(nx_g, node_type_feat_attrs, '_TYPE')

    return nx_g

def add_edge_type_feature(nx_graph):
    nx_g = nx_graph
    list_edge_type = []

    for source, target, data in nx_graph.edges(data=True):
        if data['edge_type'] is not None:
            if data['edge_type'] not in list_edge_type:
                list_edge_type.append(data['edge_type'])
            edge_type_feat = torch.tensor(list_edge_type.index(data['edge_type']))
            nx_g[source][target][0]['_TYPE'] = edge_type_feat

    return nx_g

nx_graph = nx.read_gpickle('data/reentrancy/source_code/Bank_merge_contract_graph.gpickle')
nx_graph = add_node_type_feature(nx_graph)
nx_graph = add_edge_type_feature(nx_graph)
print(nx.info(nx_graph))
print(nx_graph.nodes(data=True))
print(nx_graph.edges(data=True))

dgl_graph = dgl.from_networkx(nx_graph, node_attrs=['_TYPE'], edge_attrs=['_TYPE'])
# dgl_graph = dgl.from_networkx(nx_graph, node_attrs=['label'])
print(dgl_graph)
print(dgl_graph.nodes(), dgl_graph.ndata)
print(dgl_graph.edges(), dgl_graph.edata)

# hetero_dgl_graph = dgl.to_heterogeneous(dgl_graph, ntypes=dgl_graph.ntypes, etypes=dgl_graph.etypes)
# print(hetero_dgl_graph)

# temp = (torch.tensor([0, 1]), torch.tensor([1, 2]))
# print(temp)
