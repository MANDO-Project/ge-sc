import os
import torch
import dgl
import networkx as nx
import numpy as np
from torch import tensor


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

def generate_hetero_graph_data(nx_graph, filename_mapping):
    nx_g = nx_graph
    dict_three_cannonical_egdes = dict()
    node_tracker = {}
    retrived_node = []
    for source, target, data in nx_g.edges(data=True):
        edge_type = data['edge_type']
        source_node_type = nx_g.nodes[source]['node_type']
        target_node_type = nx_g.nodes[target]['node_type']
        three_cannonical_egde = (source_node_type, edge_type, target_node_type)
        source_filename = nx_g.nodes[source]['source_file']
        target_filename = nx_g.nodes[target]['source_file']
        # if source_node_type not in node_tracker.keys():
        #     node_tracker[source_node_type] = torch.tensor([filename_mapping[source_filename]], dtype=torch.int64)
        # else:
        #     node_tracker[source_node_type] = torch.cat((node_tracker[source_node_type], torch.tensor([filename_mapping[source_filename]], dtype=torch.int64)))
        # if target_node_type not in node_tracker.keys():
        #     node_tracker[target_node_type] = torch.tensor([filename_mapping[target_filename]], dtype=torch.int64)
        # else:
        #     node_tracker[target_node_type] = torch.cat((node_tracker[target_node_type], torch.tensor([filename_mapping[target_filename]], dtype=torch.int64)))

        if three_cannonical_egde not in dict_three_cannonical_egdes.keys():
            dict_three_cannonical_egdes[three_cannonical_egde] = [(nx_g.nodes[source]['node_hetero_id'], nx_g.nodes[target]['node_hetero_id'])]
        else:
            current_val = dict_three_cannonical_egdes[three_cannonical_egde]
            temp_edge = (nx_g.nodes[source]['node_hetero_id'], nx_g.nodes[target]['node_hetero_id'])
            current_val.append(temp_edge)
            dict_three_cannonical_egdes[three_cannonical_egde] = current_val

        # if source not in retrived_node:
        #     retrived_node.append(source)
        #     if source_node_type not in node_tracker.keys():
        #         node_tracker[source_node_type] = torch.tensor([filename_mapping[source_filename]], dtype=torch.int64)
        #     else:
        #         node_tracker[source_node_type] = torch.cat((node_tracker[source_node_type], torch.tensor([filename_mapping[source_filename]], dtype=torch.int64)))
        # if target not in retrived_node:
        #     retrived_node.append(target)
        #     if target_node_type not in node_tracker.keys():
        #         node_tracker[target_node_type] = torch.tensor([filename_mapping[target_filename]], dtype=torch.int64)
        #     else:
        #         node_tracker[target_node_type] = torch.cat((node_tracker[target_node_type], torch.tensor([filename_mapping[target_filename]], dtype=torch.int64)))
    
    node_tracker = {}
    for node, node_data in nx_g.nodes(data=True):
        node_type = node_data['node_type']
        filename = node_data['source_file']
        if node_type not in node_tracker.keys():
            node_tracker[node_type] = torch.tensor([filename_mapping[filename]], dtype=torch.int64)
        else:
            node_tracker[node_type] = torch.cat((node_tracker[node_type], torch.tensor([filename_mapping[filename]], dtype=torch.int64)))

    dict_three_cannonical_egdes = convert_edge_data_to_tensor(dict_three_cannonical_egdes)

    return dict_three_cannonical_egdes, node_tracker


def filename_mapping(extracted_graph):
    return {file: idx for idx, file in enumerate(os.listdir(extracted_graph))}


def get_number_of_nodes(nx_graph):
    nx_g = nx_graph
    number_of_nodes = {}
    for node, data in nx_g.nodes(data=True):
        if data['node_type'] not in number_of_nodes.keys():
            number_of_nodes[data['node_type']] = 1
        else:
            number_of_nodes[data['node_type']] += 1
    return number_of_nodes


def load_nx_graph(nx_graph_path):
    nx_graph = nx.read_gpickle(nx_graph_path)
    nx_graph = nx.convert_node_labels_to_integers(nx_graph)
    nx_graph = add_hetero_ids(nx_graph)
    return nx_graph


def add_cfg_mapping(nx_call_graph, nx_cfg_graph):
    nx_graph = nx_call_graph
    for call_node, call_node_data in nx_graph.nodes(data=True):
        cfg_mapping = {}
        for cfg_node, cfg_data in nx_cfg_graph.nodes(data=True):
            if call_node_data['function_fullname'] == cfg_data['function_fullname'] and \
               call_node_data['contract_name'] == cfg_data['contract_name'] and \
               call_node_data['source_file'] == cfg_data['source_file']:
                if cfg_data['node_type'] not in cfg_mapping.keys():
                    cfg_mapping[cfg_data['node_type']] = [cfg_data['node_hetero_id']]
                else:
                    cfg_mapping[cfg_data['node_type']].append(cfg_data['node_hetero_id'])
        nx_graph.nodes[call_node]['cfg_mapping'] = cfg_mapping
    return nx_graph


if __name__ == '__main__':
    extracted_graph = './dgl_models/pytorch/han/dataset/smartbugs/source_code'
    compressed_graph = './dgl_models/pytorch/han/dataset/smartbugs/compressed_graphs/compressed_graphs.gpickle'
    # compressed_graph = './data/smartbug-dataset/compressed_graphs/compress_graphs.gpickle'
    filename_mapping = filename_mapping(extracted_graph)
    cfg_nx_graph = load_nx_graph(compressed_graph)

    for n, data in cfg_nx_graph.nodes(data=True):
        if 'node_info_vulnerabilities' not in data:
            print(data)
        
    # print(count)
    # nx_g_data, node_tracker = generate_hetero_graph_data(nx_graph, filename_mapping)
    # count_node = {}
    # max_node_id = {}
    # print('number of metapath: ', len(list(nx_g_data.keys())))
    # for k, v in nx_g_data.items():
    #     if k[0] not in count_node.keys():
    #         count_node[k[0]] = torch.unique(v[0])
    #         max_node_id[k[0]] = torch.max(v[0]).item()
    #     else:
    #         count_node[k[0]] = torch.unique(torch.cat((count_node[k[0]], torch.unique(v[0]))))
    #         max_node_id[k[0]] = max(max_node_id[k[0]], torch.max(v[0]).item())
    #     if k[2] not in count_node.keys():
    #         count_node[k[2]] = torch.unique(v[1])
    #         max_node_id[k[2]] = torch.max(v[1])
    #     else:
    #         count_node[k[2]] = torch.unique(torch.cat((count_node[k[2]], torch.unique(v[1]))))
    #         max_node_id[k[2]] = max(max_node_id[k[2]], torch.max(v[1]).item())
    # print('count by unique node id')
    # for k, v in count_node.items():
    #     print(k, v.shape)
    # print('count by max node id')
    # print(max_node_id)
    # total_nodes = 0
    # print('node tracker')
    # for k, v in node_tracker.items():
    #     print(f'{k} - {v.shape}')
    # # node_tracker['EXPRESSION'] = torch.cat((node_tracker['EXPRESSION'], torch.ones(4) * -1))
    # # node_tracker['FUNCTION_NAME'] = torch.cat((node_tracker['FUNCTION_NAME'], torch.ones(1465) * -1))

    # number_of_nodes = get_number_of_nodes(nx_graph)
    # print('number of nodes: ', number_of_nodes)
    # dgl_hete_graph = dgl.heterograph(nx_g_data, num_nodes_dict=number_of_nodes)
    # print(dgl_hete_graph)
    # dgl_hete_graph.ndata['filename'] = node_tracker
    # # print(node_tracker)
