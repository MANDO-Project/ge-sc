from collections import defaultdict
from operator import index

import networkx as nx
import torch
from torch_geometric.data import HeteroData
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

from sco_models.model_hetero import MANDOGraphClassifier as GraphClassifier
from sco_models.graph_utils import load_hetero_nx_graph

class GESCData:
    def __init__(self, input_graph, split=None ,gpu=None):
        self.nx_graph = load_hetero_nx_graph(input_graph)
        self.original_graph = self.nx_graph.copy()
        if split is not None:
            amount_nodes = int(split  * len(self.nx_graph))
            self.nx_graph = self.nx_graph.subgraph(range(amount_nodes))
        self.ntypes = set([node for node in self.get_node_types()])
        self.ntypes_idx = {node_type: idx for idx, node_type in enumerate(self.ntypes)}
        self.etypes = set([edge for edge in self.get_edge_types()])
        self.etypes_idx = {edge_type: idx for idx, edge_type in enumerate(self.etypes)}
        targets = [t for t in self.get_node_targets()]
        self.process_node_attribution()
        self.process_edge_attribution()
        self.data = from_networkx(self.nx_graph, group_node_attrs=self._get_node_attribution(), group_edge_attrs=self._get_edge_attribution())
        self.data.update({'y': torch.tensor(targets, dtype=torch.int64),
                          'name': 'GeSc',
                          'num_classes': 2})

    def get_edge_types(self):
        for edge in self.nx_graph.edges(data=True):
            yield edge[2]['edge_type']

    def get_node_types(self):
        for node in self.nx_graph.nodes(data=True):
            yield node[1]['node_type']

    def get_node_targets(self):
        for node in self.nx_graph.nodes(data=True):
            yield 0 if node[1]['node_info_vulnerabilities'] is None else 1

    def _get_node_attribution(self):
        return list(next(iter(self.nx_graph.nodes(data=True)))[-1].keys())

    def _get_edge_attribution(self):
        return list(next(iter(self.nx_graph.edges(data=True)))[-1].keys())

    def process_node_attribution(self):
        node_attrs = self._get_node_attribution()
        for nid, node_data in self.nx_graph.nodes(data=True):
            node_feat = self._nodetype2onehot(node_data['node_type'])
            for attr in node_attrs:
                del self.nx_graph.nodes[nid][attr]
            self.nx_graph.nodes[nid]['node_feature'] = node_feat.tolist()

    def process_edge_attribution(self):
        edge_attrs = self._get_edge_attribution()
        for src, dst, edge_data in self.nx_graph.edges(data=True):
            edge_feat = self._edgetype2onehot(edge_data['edge_type'])
            for attr in edge_attrs:
                del self.nx_graph.edges[(src, dst, 0)][attr]
            self.nx_graph.edges[(src, dst, 0)]['edge_feature'] = edge_feat.tolist()

    def _nodetype2onehot(self, nodetype):
        onehot = torch.zeros(len(self.ntypes))
        onehot[self.ntypes_idx[nodetype]] = 1
        return onehot

    def _edgetype2onehot(self, edgetype):
        onehot = torch.zeros(len(self.etypes))
        onehot[self.etypes_idx[edgetype]] = 1
        return onehot

    def get_nodes_of_source(self, source_file):
        indexes = []
        for id, node in self.original_graph.nodes(data=True):
            if node['source_file'] == source_file:
                indexes.append(id)       
        return indexes


if __name__ == '__main__':
    graph_path = './experiments/ge-sc-data/source_code/reentrancy/buggy_curated/cfg_cg_compressed_graphs.gpickle'
    checkpoint = './models/node_classification/source_code/nodetype/reentrancy/logs_hgt.pth'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # model = GraphClassifier(graph_path, feature_extractor=None, node_feature='nodetype', device=device)
    # model.load_state_dict(torch.load(checkpoint))
    # model.to(device)
    # model.eval()

    ge_sc_data = GESCData(graph_path, device)
    print(ge_sc_data.data)
