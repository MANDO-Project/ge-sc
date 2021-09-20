import torch
import dgl
import networkx as nx
import numpy as np

def add_hetero_ids(nx_graph):
    nx_g = nx_graph
    dict_hetero_id = {}

    for node, node_data in nx_g.nodes(data=True):
        if node_data['node_type'] not in dict_hetero_id:
            dict_hetero_id[node_data['node_type']] = 0
        else:
            dict_hetero_id[node_data['node_type']] += 1
        
        nx_g.nodes[node]['node_hetero_id'] = dict_hetero_id[node_data['node_type']]

    print(dict_hetero_id)

    return nx_g

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
            # print(node_type_feat)

    nx.set_node_attributes(nx_g, node_type_feat_attrs, '_TYPE')

    return nx_g, list_node_type

def add_edge_type_feature(nx_graph):
    nx_g = nx_graph
    list_edge_type = []

    for source, target, data in nx_graph.edges(data=True):
        if data['edge_type'] is not None:
            if data['edge_type'] not in list_edge_type:
                list_edge_type.append(data['edge_type'])
            edge_type_feat = torch.tensor(list_edge_type.index(data['edge_type']))
            nx_g[source][target][0]['_TYPE'] = edge_type_feat

    return nx_g, list_edge_type

def convert_edge_data_to_tensor(dict_egdes):
    dict_three_cannonical_egdes = dict_egdes

    for key, val in dict_three_cannonical_egdes.items():
        list_source = []
        list_target = []
        for source, target in val:
            list_source.append(source)
            list_target.append(target)
        # print(list_source, list_target)
        dict_three_cannonical_egdes[key] = (torch.tensor(list_source), torch.tensor(list_target))

    return dict_three_cannonical_egdes

def generate_hetero_graph_data(nx_graph):
    nx_g = nx_graph
    dict_three_cannonical_egdes = dict()
    for source, target, data in nx_g.edges(data=True):
        edge_type = data['edge_type']
        source_node_type = nx_g.nodes[source]['node_type']
        target_node_type = nx_g.nodes[target]['node_type']
        three_cannonical_egde = (source_node_type, edge_type, target_node_type)
        # print(dict_three_cannonical_egdes)
        # print(three_cannonical_egde, source, target)
        if three_cannonical_egde not in dict_three_cannonical_egdes.keys():
            dict_three_cannonical_egdes[three_cannonical_egde] = [(nx_g.nodes[source]['node_hetero_id'], nx_g.nodes[target]['node_hetero_id'])]
        else:
            current_val = dict_three_cannonical_egdes[three_cannonical_egde]
            temp_edge = (nx_g.nodes[source]['node_hetero_id'], nx_g.nodes[target]['node_hetero_id'])
            current_val.append(temp_edge)
            dict_three_cannonical_egdes[three_cannonical_egde] = current_val
    
    dict_three_cannonical_egdes = convert_edge_data_to_tensor(dict_three_cannonical_egdes)

    return dict_three_cannonical_egdes

nx_graph = nx.read_gpickle('data/extracted_source_code/compress_graphs.gpickle')
print(nx.info(nx_graph))
# nx_graph, list_node_type = add_node_type_feature(nx_graph)
# print(list_node_type)
# nx_graph, list_edge_type = add_edge_type_feature(nx_graph)
# print(list_edge_type)
nx_graph = nx.convert_node_labels_to_integers(nx_graph)
nx_graph = add_hetero_ids(nx_graph)
# print(nx_graph.nodes(data=True))

nx_g_data = generate_hetero_graph_data(nx_graph)
# print(nx_g_data)

dgl_hete_graph = dgl.heterograph(nx_g_data)
print(dgl_hete_graph)
# print(dgl_hete_graph.ntypes, dgl_hete_graph.num_nodes())
# print(dgl_hete_graph.etypes, dgl_hete_graph.num_edges())
# for ntype in dgl_hete_graph.ntypes:
#     print(ntype,dgl_hete_graph.num_nodes(ntype))


# homo_g = dgl.to_homogeneous(dgl_hete_graph)
# print(homo_g)
# print('nodes:', homo_g.nodes(), homo_g.ndata)
# print('edges:', homo_g.edges(), homo_g.edata)

# sub_g = dgl.node_subgraph(dgl_hete_graph, {'END_IF': [24]})
# print(sub_g)
# print(sub_g.ntypes)
# print(sub_g.etypes)
# print(nx.info(nx_graph))
# print(nx_graph.nodes(data=True))
# print(nx_graph.edges(data=True))

# dgl_graph = dgl.from_networkx(nx_graph, node_attrs=['_TYPE'], edge_attrs=['_TYPE'])
# dgl_graph.ndata[dgl.NID] = dgl_graph.nodes()
# dgl_graph.edata[dgl.EID] = dgl_graph.edges()[0]

# dgl_graph = dgl.from_networkx(nx_graph, node_attrs=['label'])
# print(dgl_graph)
# print('nodes:', dgl_graph.nodes(), dgl_graph.ndata)
# print('edges:', dgl_graph.edges(), dgl_graph.edata)

# hetero_dgl_graph = dgl.to_heterogeneous(dgl_graph, ntypes=dgl_graph.ntypes, etypes=dgl_graph.etypes)
# print(hetero_dgl_graph)

# temp = (torch.tensor([0, 1]), torch.tensor([1, 2]))
# print(temp)

# Test heterograph
# graph_data = {
#    ('drug', 'interacts', 'drug'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
#    ('drug', 'interacts', 'gene'): (torch.tensor([0, 1]), torch.tensor([2, 3])),
#    ('drug', 'treats', 'disease'): (torch.tensor([1]), torch.tensor([2])),
#    ('drug', 'treats', 'abc'): (torch.tensor([], dtype=torch.int32), torch.tensor([], dtype=torch.int32))
# }

# graph_data = {
#    ('drug', 'interacts', 'drug'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
#    ('drug', 'interacts', 'gene'): (torch.tensor([0, 1]), torch.tensor([2, 3])),
#    ('drug', 'treats', 'disease'): (torch.tensor([1]), torch.tensor([2]))
# }

# hg = dgl.heterograph(graph_data, idtype=torch.int32)
# print(hg)
# print(hg.ntypes)
# print(hg.etypes)

# g = dgl.to_homogeneous(dgl_hete_graph)
# print(g)
# print('nodes:', g.nodes(), g.ndata)
# print('edges:', g.edges(), g.edata)

# g = dgl.to_homogeneous(graph_data)
# print(g)
# print('nodes:', g.nodes(), g.ndata)
# print('edges:', g.edges(), g.edata)

