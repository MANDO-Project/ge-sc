import os

import networkx as nx
import pickle
from dgl import subgraph
import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch_geometric.nn import MetaPath2Vec
import dgl
import dgl.nn.pytorch as dglnn

from .graph_utils import load_hetero_nx_graph, generate_hetero_graph_data, reflect_graph, \
                         get_number_of_nodes, map_node_embedding, get_symmatrical_metapaths, \
                         get_node_tracker, generate_hetero_subgraph_data, \
                         add_hetero_subgraph_ids, get_nx_subgraphs, \
                         add_hetero_ids


class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs is features of nodes
        h = self.conv1(graph, inputs)
        h = {k: relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h


class RGCNVulClassifier(nn.Module):
    def __init__(self,compressed_global_graph_path, source_path, feature_extractor=None, node_feature='nodetype', hidden_size=32, out_size=2, device='cpu'):
        super(RGCNVulClassifier, self).__init__()
        self.compressed_global_graph_path = compressed_global_graph_path
        self.hidden_size = hidden_size
        self.source_path = source_path
        self.extracted_graph = [f for f in os.listdir(self.source_path) if f.endswith('.sol')]
        self.filename_mapping = {file: idx for idx, file in enumerate(self.extracted_graph)}
        self.device = device
        # Get Global graph
        nx_graph = load_hetero_nx_graph(compressed_global_graph_path)
        # nx_graph = add_hetero_subgraph_ids(nx_graph)
        nx_g_data = generate_hetero_graph_data(nx_graph)
        _node_tracker = get_node_tracker(nx_graph, self.filename_mapping)

        # # Get subgraphs
        # self.nx_subgraph_dict = get_nx_subgraphs(nx_graph)
        # assert len(list(self.nx_subgraph_dict.keys())) == len(self.extracted_graph)
        # self.dgl_subgraph_dict = {}
        # for filename, subgraph_data in  self.nx_subgraph_dict.items():
        #     _nx_subgraph = nx.convert_node_labels_to_integers(subgraph_data)
        #     _nx_subgraph = add_hetero_ids(_nx_subgraph)
        #     _nx_subgraph_data = generate_hetero_graph_data(_nx_subgraph)
        #     _node_subtracker = get_node_tracker(_nx_subgraph, self.filename_mapping)
        #     _symmetrical_subgraph_data = reflect_graph(_nx_subgraph_data)
        #     _sub_number_of_nodes = get_number_of_nodes(_nx_subgraph)
        #     _dgl_subgraph = dgl.heterograph(_symmetrical_subgraph_data, num_nodes_dict=_sub_number_of_nodes)
        #     _dgl_subgraph.ndata['filename'] = _node_subtracker
        #     self.dgl_subgraph_dict[filename] = _dgl_subgraph

        # Reflect graph data
        self.symmetrical_global_graph_data = reflect_graph(nx_g_data)
        self.number_of_nodes = get_number_of_nodes(nx_graph)
        self.symmetrical_global_graph = dgl.heterograph(self.symmetrical_global_graph_data, num_nodes_dict=self.number_of_nodes)
        self.symmetrical_global_graph.ndata['filename'] = _node_tracker
        self.meta_paths = get_symmatrical_metapaths(self.symmetrical_global_graph)
        # Concat the metapaths have the same begin nodetype
        self.full_metapath = {}
        for metapath in self.meta_paths:
            ntype = metapath[0][0]
            if ntype not in self.full_metapath:
                self.full_metapath[ntype] = [metapath]
            else:
                self.full_metapath[ntype].append(metapath)
        self.node_types = set([meta_path[0][0] for meta_path in self.meta_paths])
        self.edge_types = set([meta_path[0][1] for meta_path in self.meta_paths])
        self.ntypes_dict = {k: v for v, k in enumerate(self.node_types)}
        features = {}
        if node_feature == 'nodetype':
            for ntype in self.symmetrical_global_graph.ntypes:
                features[ntype] = self._nodetype2onehot(ntype).repeat(self.symmetrical_global_graph.num_nodes(ntype), 1).to(self.device)
            self.in_size = len(self.node_types)
        elif node_feature == 'metapath2vec':
            embedding_dim = 128
            self.in_size = embedding_dim
            for metapath in self.meta_paths:
                _metapath_embedding = MetaPath2Vec(self.symmetrical_global_graph_data, embedding_dim=embedding_dim,
                        metapath=metapath, walk_length=50, context_size=7,
                        walks_per_node=5, num_negative_samples=5, num_nodes_dict=None,
                        sparse=False)
                ntype = metapath[0][0]
                if ntype not in features.keys():
                    features[ntype] = _metapath_embedding(ntype).unsqueeze(0)
                else:
                    features[ntype] = torch.cat((features[ntype], _metapath_embedding(ntype).unsqueeze(0)))
            features = {k: torch.mean(v, dim=0).to(self.device) for k, v in features.items()}
        elif node_feature in ['gae', 'node2vec', 'line']:
            embedding_dim = 128
            self.in_size = embedding_dim
            with open(feature_extractor, 'rb') as f:
                embedding = pickle.load(f, encoding="utf8")
            embedding = torch.tensor(embedding, device=device)
            features = map_node_embedding(nx_graph, embedding)
        
        self.symmetrical_global_graph = self.symmetrical_global_graph.to(self.device)
        self.symmetrical_global_graph.ndata['feat'] = features

        # Init Model
        self.rgcn = RGCN(self.in_size, self.hidden_size, self.hidden_size, self.edge_types)
        self.classify = nn.Linear(self.hidden_size, out_size)

    def _nodetype2onehot(self, ntype):
        feature = torch.zeros(len(self.ntypes_dict), dtype=torch.float)
        feature[self.ntypes_dict[ntype]] = 1
        return feature

    def get_assemble_node_features(self):
        features = {}
        for han in self.layers:
            ntype = han.meta_paths[0][0][0]
            feature = han(self.symmetrical_global_graph, self.symmetrical_global_graph.ndata['feat'][ntype].to(self.device))
            if ntype not in features.keys():
                features[ntype] = feature.unsqueeze(0)
            else:
                features[ntype] = torch.cat((features[ntype], feature.unsqueeze(0)))
        # Use mean for aggregate node hidden features
        return {k: torch.mean(v, dim=0) for k, v in features.items()}

    def get_node_features(self):
        features = {}
        for ntype in self.node_types:
            features[ntype] = self.layers_dict[ntype](self.symmetrical_global_graph, self.symmetrical_global_graph.ndata['feat'][ntype].to(self.device))
        return features

    def reset_parameters(self):
        for layer in self.rgcn.children():
            if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        for layer in self.classify.children():
            if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

    def forward(self, batched_graph):
        batched_graph_embedded = []
        for g_name in batched_graph:
            file_ids = self.filename_mapping[g_name]
            node_mask = {}
            for node_type in self.node_types:
                node_mask[node_type] = self.symmetrical_global_graph.ndata['filename'][node_type] == file_ids
            sub_graph = dgl.node_subgraph(self.symmetrical_global_graph, node_mask)
            h = sub_graph.ndata['feat']
            # sub_graph = self.dgl_subgraph_dict[g_name]
            h = self.rgcn(sub_graph, h)
            graph_embedded = 0
            for node_type, feature in h.items():
                graph_embedded += feature.mean(0)
            if isinstance(graph_embedded, int):
                batched_graph_embedded.append([0] * self.hidden_size)
            else:
                batched_graph_embedded.append(graph_embedded.tolist())
        batched_graph_embedded = torch.tensor(batched_graph_embedded).to(self.device)
        output = self.classify(batched_graph_embedded)
        return output


if __name__ == '__main__':
    from dataloader import EthIdsDataset
    from dgl.dataloading import GraphDataLoader
    dataset = './dataset/aggregate/source_code'
    compressed_graph = './dataset/call_graph/compressed_graph/compress_call_graphs_no_solidity_calls.gpickle'
    label = './dataset/aggregate/labels.json'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    ethdataset = EthIdsDataset(dataset, compressed_graph, label)
    dataloader = GraphDataLoader(ethdataset, batch_size=128)
    model = RGCNVulClassifier(compressed_graph, ethdataset.filename_mapping, device=device)
    model.to(device)
