from collections import defaultdict

import torch
import networkx as nx
import dgl

from .opcodes import int2op


def add_hetero_ids(nx_graph):
    nx_g = nx_graph
    dict_hetero_id = {}

    for node, node_data in nx_g.nodes(data=True):
        if node_data['node_type'] not in dict_hetero_id:
            dict_hetero_id[node_data['node_type']] = 0
        else:
            dict_hetero_id[node_data['node_type']] += 1
        nx_g.nodes[node]['node_hetero_id'] = dict_hetero_id[node_data['node_type']]
    return nx_g


def add_hetero_subgraph_ids(nx_graph):
    nx_g = nx_graph
    dict_hetero_subgraph_id = {}
    for node, node_data in nx_g.nodes(data=True):
        filename = node_data['source_file']
        nodetype = node_data['node_type']
        if filename not in dict_hetero_subgraph_id:
            dict_hetero_id = {}
            dict_hetero_id[nodetype] = 0
        else:
            dict_hetero_id = dict_hetero_subgraph_id[filename]
            if nodetype not in dict_hetero_id:
                dict_hetero_id[nodetype] = 0
            else:
                dict_hetero_id[nodetype] += 1
        dict_hetero_subgraph_id[filename] = dict_hetero_id
        nx_g.nodes[node]['node_hetero_subgraph_id'] = dict_hetero_subgraph_id[filename][nodetype]
    return nx_g


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


def map_node_embedding(nx_graph, embedding):
    nx_g = nx_graph
    features = {}
    assert len(nx_g.nodes) == embedding.shape[0]
    for node_ids, node_data in nx_g.nodes(data=True):
        node_type = node_data['node_type']
        if node_type not in features:
            features[node_type] = embedding[node_ids].unsqueeze(0)
        else:
            features[node_type] = torch.cat((features[node_type], embedding[node_ids].unsqueeze(0)))
    return features


def generate_random_node_features(nx_graph, feature_dims):
    nx_g = nx_graph
    features = {}
    for node_ids, node_data in nx_g.nodes(data=True):
        node_type = node_data['node_type']
        if node_type not in features:
            features[node_type] = torch.rand(1, feature_dims)
        else:
            features[node_type] = torch.cat((features[node_type], torch.rand(1, feature_dims)))
    return features


def generate_zeros_node_features(nx_graph, feature_dims):
    nx_g = nx_graph
    features = {}
    for node_ids, node_data in nx_g.nodes(data=True):
        node_type = node_data['node_type']
        if node_type not in features:
            features[node_type] = torch.zeros((1, feature_dims))
        else:
            features[node_type] = torch.cat((features[node_type], torch.zeros((1, feature_dims))))
    return features


def op2onehot(op):
    s = '0123456789abcdef'
    str2int = {s[i]: i for i in range(len(s))}
    onehot_line = [0 for _ in range(256)]
    if op != 'EXIT BLOCK':
        op2int = {int2op[key]:key for key in int2op}
        hexstr = op2int[op]
        number = str2int[hexstr[0]] * 16 + str2int[hexstr[1]]
        onehot_line[number] = 1
    return onehot_line

def generate_lstm_node_features(nx_graph):
    nx_g = nx_graph
    graph_seq_feature = []
    features = {}
    for node_idx, node_data in nx_g.nodes(data=True):
        node_type = node_data['node_type']
        node_seq_feature = []
        seq = node_data.get('label', '').split(':')[1:]
        for unit in seq:
            op = unit[1:unit.find('\\')]
            if 'PUSH' in op:
                op = op.split(' ')[0]
            node_seq_feature.append(op2onehot(op))
        avg_seq_node = torch.mean(torch.tensor(node_seq_feature, dtype=torch.float), 0)
        # print(avg_seq_node.shape)
        # graph_seq_feature.append(torch.tensor([node_seq_feature]).float())
        if node_type not in features:
            features[node_type] = avg_seq_node.unsqueeze(0)
        else:
            features[node_type] = torch.cat((features[node_type], avg_seq_node.unsqueeze(0)))
    return features


def reveert_map_node_embedding(nx_graph, features):
    nx_g = nx_graph
    embedded = torch.zeros(len(nx_g.nodes), 128)
    feature_count = {}
    for node_ids, node_data in nx_g.nodes(data=True):
        node_type = node_data['node_type']
        if node_type not in feature_count:
            feature_count[node_type] = 0
        embedded[node_ids] = features[node_type][feature_count[node_type]]
        feature_count[node_type] += 1
    return embedded


def get_node_label(nx_graph):
    nx_g = nx_graph
    node_labels = []
    label_ids = {'valid': 0}
    labeled_node_ids = {'buggy': [], 'valid': []}
    for node_id, node_data in nx_g.nodes(data=True):
        node_type = node_data['node_type']
        node_label = node_data['node_info_vulnerabilities']
        target = 0
        if node_label is None:
            target = 0
            labeled_node_ids['valid'].append(node_id)
        else:
            bug_type = node_label[0]['category']
            if bug_type not in label_ids:
                label_ids[bug_type] = len(label_ids)
            target = label_ids[bug_type]
            # if bug_type == 'time_manipulation':
            #     target = 1
            target = 1
            labeled_node_ids['buggy'].append(node_id)
        node_labels.append(target)
    return node_labels, labeled_node_ids, label_ids


def get_node_label_by_nodetype(nx_graph):
    node_labels = defaultdict(list)
    for _, node_data in nx_graph.nodes(data=True):
        node_type = node_data['node_type']
        node_label = node_data['node_info_vulnerabilities']
        target = 0 if node_label is None else 1
        node_labels[node_type].append(target)
    return node_labels


def get_node_ids(graph, source_files):
    file_ids = []
    for node_ids, node_data in graph.nodes(data=True):
        filename = node_data['source_file']
        if filename in source_files:
            file_ids.append(node_ids)
    return file_ids


def get_node_tracker(nx_graph, filename_mapping):
    nx_g = nx_graph
    node_tracker = {}
    for _, node_data in nx_g.nodes(data=True):
        node_type = node_data['node_type']
        filename = node_data['source_file']
        if node_type not in node_tracker.keys():
            node_tracker[node_type] = torch.tensor([filename_mapping[filename]], dtype=torch.int64)
        else:
            node_tracker[node_type] = torch.cat((node_tracker[node_type], torch.tensor([filename_mapping[filename]], dtype=torch.int64)))
    return node_tracker


def get_number_of_nodes(nx_graph):
    nx_g = nx_graph
    number_of_nodes = {}
    for node, data in nx_g.nodes(data=True):
        if data['node_type'] not in number_of_nodes.keys():
            number_of_nodes[data['node_type']] = 1
        else:
            number_of_nodes[data['node_type']] += 1
    return number_of_nodes


def get_node_ids_dict(nx_graph):
    nx_g = nx_graph
    node_ids_dict = {}
    for node_ids, node_data in nx_g.nodes(data=True):
        ntype = node_data['node_type']
        if ntype not in node_ids_dict:
            node_ids_dict[ntype] = [node_ids]
        else:
            node_ids_dict[ntype].append(node_ids)
    return node_ids_dict


def get_nodetype_mask(nx_graph, nodetype_dict):
    nx_g = nx_graph
    nodetype_mask = []
    for node_ids, node_data in nx_g.nodes(data=True):
        ntype = node_data['node_type']
        nodetype_mask.append(nodetype_dict[ntype])
    return nodetype_mask


def get_node_ids_by_filename(nx_graph):
    nx_g = nx_graph
    node_ids_dict = {}
    for node_ids, node_data in nx_g.nodes(data=True):
        ntype = node_data['node_type']
        filename = node_data['source_file']
        if filename not in node_ids_dict:
            node_ids_dict[filename] = [node_ids]
        else:
            node_ids_dict[filename].append(node_ids)
    return node_ids_dict


def get_nx_subgraphs(nx_graph):
    nx_g = nx_graph
    nx_subgraphs_dict = {}
    for node, node_data in nx_g.nodes(data=True):
        filename = node_data['source_file']
        nodetype = node_data['node_type']
        if filename not in nx_subgraphs_dict:
            nx_subgraphs_dict[filename] = nx.MultiDiGraph()
        nx_subgraphs_dict[filename].add_node(node, **node_data)
    for k, v in nx_subgraphs_dict.items():
        nx_subgraphs_dict[k] = add_hetero_ids(nx_subgraphs_dict[k])
    return nx_subgraphs_dict


def load_hetero_nx_graph(nx_graph_path):
    nx_graph = nx.read_gpickle(nx_graph_path)
    nx_graph = nx.convert_node_labels_to_integers(nx_graph)
    nx_graph = add_hetero_ids(nx_graph)
    return nx_graph


def convert_edge_data_to_tensor(dict_egdes):
    dict_three_cannonical_egdes = dict_egdes
    for key, val in dict_three_cannonical_egdes.items():
        list_source = []
        list_target = []
        for source, target in val:
            list_source.append(source)
            list_target.append(target)
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

        if three_cannonical_egde not in dict_three_cannonical_egdes.keys():
            dict_three_cannonical_egdes[three_cannonical_egde] = [(nx_g.nodes[source]['node_hetero_id'], nx_g.nodes[target]['node_hetero_id'])]
        else:
            current_val = dict_three_cannonical_egdes[three_cannonical_egde]
            temp_edge = (nx_g.nodes[source]['node_hetero_id'], nx_g.nodes[target]['node_hetero_id'])
            current_val.append(temp_edge)
            dict_three_cannonical_egdes[three_cannonical_egde] = current_val

    dict_three_cannonical_egdes = convert_edge_data_to_tensor(dict_three_cannonical_egdes)
    return dict_three_cannonical_egdes


def generate_hetero_subgraph_data(nx_graph):
    nx_g = nx_graph
    subgraph_dict = dict()
    for source, target, data in nx_g.edges(data=True):
        edge_type = data['edge_type']
        source_node_type = nx_g.nodes[source]['node_type']
        target_node_type = nx_g.nodes[target]['node_type']
        source_filename = nx_g.nodes[source]['source_file']
        target_filename = nx_g.nodes[target]['source_file']
        if source_filename == target_filename:
            filename = source_filename
            three_cannonical_egde = (source_node_type, edge_type, target_node_type)
            if filename not in subgraph_dict.keys():
                dict_three_cannonical_egdes = dict()
                dict_three_cannonical_egdes[three_cannonical_egde] = [(nx_g.nodes[source]['node_hetero_subgraph_id'], nx_g.nodes[target]['node_hetero_subgraph_id'])]
            else:
                dict_three_cannonical_egdes = subgraph_dict[filename]
                if three_cannonical_egde not in dict_three_cannonical_egdes.keys():
                    dict_three_cannonical_egdes[three_cannonical_egde] = [(nx_g.nodes[source]['node_hetero_subgraph_id'], nx_g.nodes[target]['node_hetero_subgraph_id'])]
                else:
                    current_val = dict_three_cannonical_egdes[three_cannonical_egde]
                    temp_edge = (nx_g.nodes[source]['node_hetero_subgraph_id'], nx_g.nodes[target]['node_hetero_subgraph_id'])
                    current_val.append(temp_edge)
                    dict_three_cannonical_egdes[three_cannonical_egde] = current_val
            subgraph_dict[filename] = dict_three_cannonical_egdes
    for k, v in subgraph_dict.items():
        subgraph_dict[k] = convert_edge_data_to_tensor(v)
    return subgraph_dict


def generate_filename_ids(nx_graph):
    file_ids = {}
    for _, node_data in nx_graph.nodes(data=True):
        filename = node_data['source_file']
        if filename not in file_ids:
            file_ids[filename] = len(file_ids)
    return file_ids
    # return {node_data['source_file']: idx for idx, node_data in nx_graph.nodes(data=True)}

def filename_mapping(extracted_graph):
    return {file: idx for idx, file in enumerate(extracted_graph)}

# HAN defined metapaths are the path between the same type nodes.
# We need undirect the global graph.
def reflect_graph(nx_g_data):
    symmetrical_data = {}
    for metapath, value in nx_g_data.items():
        if metapath[0] == metapath[-1]:
            symmetrical_data[metapath] = (torch.cat((value[0], value[1])), torch.cat((value[1], value[0])))
        else:
            if metapath not in symmetrical_data.keys():
                symmetrical_data[metapath] = value
            else:
                symmetrical_data[metapath] = (torch.cat((symmetrical_data[metapath][0], value[0])), torch.cat((symmetrical_data[metapath][1], value[1])))
            if metapath[::-1] not in symmetrical_data.keys():
                symmetrical_data[metapath[::-1]] = (value[1], value[0])
            else:
                symmetrical_data[metapath[::-1]] = (torch.cat((symmetrical_data[metapath[::-1]][0], value[1])), torch.cat((symmetrical_data[metapath[::-1]][1], value[0])))
    return symmetrical_data


# Get all the pair of symmetrical metapath from the symmetrical graph. 
def get_symmatrical_metapaths(symmetrical_global_graph):
    meta_paths = []
    for mt in symmetrical_global_graph.canonical_etypes:
        if mt[0] == mt[-1]:
            ref_mt = [mt]
        else:
            ref_mt = [mt, mt[::-1]]
        if ref_mt not in meta_paths:
            meta_paths.append(ref_mt)
    return meta_paths


def get_length_2_metapath(symmetrical_global_graph):
    begin_by = {}
    end_by = {}
    for mt in symmetrical_global_graph.canonical_etypes:
        if mt[0] not in begin_by:
            begin_by[mt[0]] = [mt]
        else:
            begin_by[mt[0]].append(mt)
        if mt[-1] not in end_by:
            end_by[mt[-1]] = [mt]
        else:
            end_by[mt[-1]].append(mt)
    metapath_list = []
    for mt_0 in symmetrical_global_graph.canonical_etypes:
        source = mt_0[0]
        dest = mt_0[-1]
        if source == dest:
            metapath_list.append([mt_0])
        first_metapath = [mt_0]
        if dest in begin_by:
            for mt_1 in begin_by[dest]:
                if mt_1 != mt_0 and mt_1[-1] == source:
                    second_metapath = first_metapath + [mt_1]
                    metapath_list.append(second_metapath)
    return metapath_list


def get_length_3_metapath(symmetrical_global_graph):
    begin_by = {}
    end_by = {}
    for mt in symmetrical_global_graph.canonical_etypes:
        if mt[0] not in begin_by:
            begin_by[mt[0]] = [mt]
        else:
            begin_by[mt[0]].append(mt)
        if mt[-1] not in end_by:
            end_by[mt[-1]] = [mt]
        else:
            end_by[mt[-1]].append(mt)
    metapath_list = []
    for mt_0 in symmetrical_global_graph.canonical_etypes:
        source = mt_0[0]
        first_metapath = [mt_0]
        for mt_1 in begin_by[mt_0[-1]]:
            if mt_1 != mt_0:
                second_metapath = first_metapath + [mt_1]
                intermediate = mt_1[-1]
                for mt_2 in end_by[source]:
                    if mt_2[0] == intermediate and mt_1 != mt_2:
                        third_metapath = second_metapath + [mt_2]
                        metapath_list.append(third_metapath)
    return metapath_list


def get_subgraph_by_metapath(nx_graph, dgl_graph, metapath):
    nx_g = nx_graph
    number_of_nodes = 0
    source_nodes = []
    target_nodes = []
    edges = []
    for source, edge, target, data in nx_g.edges(data=True, keys=True):
        edge_type = data['edge_type']
        source_node_type = nx_g.nodes[source]['node_type']
        target_node_type = nx_g.nodes[target]['node_type']
        # print((source_node_type, edge_type, target_node_type))
        if (source_node_type, edge_type, target_node_type) == metapath:
            print('this metapath')
            number_of_nodes += 1
            # source_nodes.append(int(nx_g.nodes[source]['node_hetero_id']))
            # target_nodes.append(int(nx_g.nodes[target]['node_hetero_id']))
            edges.append(edge)
    # print(source_nodes)
    # print(target_nodes)
    return dgl.edge_subgraph(dgl_graph, {metapath: edges}, preserve_nodes=True)

# if __name__ == '__main__':
#     canonical_edges = [('0', 'a', '0'), ('0', 'a', '1'), ('1', 'a', '0'),
#                        ('1', 'b', '1'), ('1', 'b', '0'),('0', 'b', '1'), 
#                        ('1', 'a', '2'),('2', 'a', '1'),('1', 'a', '3'),('3', 'a', '1'),
#                        ('0', 'a', '3'), ('3', 'a', '0'), ('2', 'b', '3'), ('3', 'b', '2')]
#     print('canonical_etypes: ', len(canonical_edges))
#     metapath_length_2 = get_length_2_metpath(canonical_edges)
#     metapath_length_3 = get_length_3_metapath(canonical_edges)
#     print('Meta path length <=2: ', len(metapath_length_2))
#     print(metapath_length_2)
#     print('Meta path length 3', len(metapath_length_3))
#     print(metapath_length_3)
