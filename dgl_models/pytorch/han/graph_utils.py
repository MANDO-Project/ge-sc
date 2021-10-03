import torch
import networkx as nx

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


def generate_hetero_graph_data(nx_graph, filename_mapping):
    nx_g = nx_graph
    dict_three_cannonical_egdes = dict()
    node_tracker = {}
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

    for _, node_data in nx_g.nodes(data=True):
        node_type = node_data['node_type']
        filename = node_data['source_file']
        if node_type not in node_tracker.keys():
            node_tracker[node_type] = torch.tensor([filename_mapping[filename]], dtype=torch.int64)
        else:
            node_tracker[node_type] = torch.cat((node_tracker[node_type], torch.tensor([filename_mapping[filename]], dtype=torch.int64)))

    first_cannonical = list(dict_three_cannonical_egdes.keys())[0]
    dict_three_cannonical_egdes = convert_edge_data_to_tensor(dict_three_cannonical_egdes)

    return dict_three_cannonical_egdes, node_tracker


def get_number_of_nodes(nx_graph):
    nx_g = nx_graph
    number_of_nodes = {}
    for node, data in nx_g.nodes(data=True):
        if data['node_type'] not in number_of_nodes.keys():
            number_of_nodes[data['node_type']] = 1
        else:
            number_of_nodes[data['node_type']] += 1
    return number_of_nodes


def filename_mapping(extracted_graph):
    return {file: idx for idx, file in enumerate(extracted_graph)}
