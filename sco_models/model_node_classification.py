"""This model shows an example of using dgl.metapath_reachable_graph on the original heterogeneous
graph.

Because the original HAN implementation only gives the preprocessed homogeneous graph, this model
could not reproduce the result in HAN as they did not provide the preprocessing code, and we
constructed another dataset from ACM with a different set of papers, connections, features and
labels.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import dgl
from dgl.nn.pytorch import GATConv
from torch.nn.modules.sparse import Embedding
from torch_geometric.nn import MetaPath2Vec

from graph_utils import load_hetero_nx_graph, generate_hetero_graph_data, get_number_of_nodes, add_cfg_mapping, get_node_label, get_node_tracker


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)

        return (beta * z).sum(1)                       # (N, D * K)


class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : DGLHeteroGraph
        The heterogeneous graph
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """
    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu,
                                           allow_zero_in_degree=True))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(
                        g, meta_path)

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)                  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)                            # (N, D * K)


class HANVulNodeClassifier(nn.Module):
    def __init__(self, compressed_global_graph_path, source_path, feature_extractor=None, node_feature='han', hidden_size=16, out_size=2,num_heads=8, dropout=0.6, device='cpu'):
        super(HANVulNodeClassifier, self).__init__()
        self.compressed_global_graph_path = compressed_global_graph_path
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.source_path = source_path
        self.extracted_graph = [f for f in os.listdir(self.source_path) if f.endswith('.sol')]
        self.filename_mapping = {file: idx for idx, file in enumerate(self.extracted_graph)}
        self.device = device
        # Get Global graph
        nx_graph = load_hetero_nx_graph(compressed_global_graph_path)
        nx_g_data = generate_hetero_graph_data(nx_graph)
        _node_tracker = get_node_tracker(nx_graph, self.filename_mapping)

        # Get None Labels
        self.node_labels = get_node_label(nx_graph)

        # Reflect graph data
        self.symmetrical_global_graph_data = self.reflect_graph(nx_g_data)
        self.number_of_nodes = get_number_of_nodes(nx_graph)
        self.symmetrical_global_graph = dgl.heterograph(self.symmetrical_global_graph_data, num_nodes_dict=self.number_of_nodes)
        self.symmetrical_global_graph.ndata['filename'] = _node_tracker
        self.meta_paths = self.get_symmatrical_metapaths()
        self.node_types = set([meta_path[0][0] for meta_path in self.meta_paths])
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
                        walks_per_node=5, num_negative_samples=5, num_nodes_dict=self.number_of_nodes,
                        sparse=False)
                ntype = metapath[0][0]
                if ntype not in features.keys():
                    features[ntype] = _metapath_embedding(ntype).unsqueeze(0)
                else:
                    features[ntype] = torch.cat((features[ntype], _metapath_embedding(ntype).unsqueeze(0)))
            # Use mean for aggregate node features
            features = {k: torch.mean(v, dim=0).to(self.device) for k, v in features.items()}
        elif node_feature == 'han':
            assert feature_extractor is not None, "Please pass features extraction model"
            nx_cfg_graph = load_hetero_nx_graph(feature_extractor.compressed_global_graph_path)
            self.in_size = feature_extractor.hidden_size * feature_extractor.num_heads
            nx_graph = add_cfg_mapping(nx_graph, nx_cfg_graph)
            han_features = feature_extractor.get_node_features()
            features = {}
            for node, node_data in nx_graph.nodes(data=True):
                han_mapping = node_data['cfg_mapping']
                for k, v in han_mapping.items():
                    # change torch operation to change the compressed method of cfg subgraph of a call node.
                    node_features = torch.mean(han_features[k][v], dim=0).unsqueeze(0).to(self.device)
                if node_data['node_type'] not in features.keys():
                    features[node_data['node_type']] = node_features
                else:
                    features[node_data['node_type']] = torch.cat((features[node_data['node_type']], node_features))

        # self.symmetrical_global_graph = self.symmetrical_global_graph.to('cpu')
        self.symmetrical_global_graph = self.symmetrical_global_graph.to(self.device)
        self.symmetrical_global_graph.ndata['feat'] = features

        # Init Model
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer([self.meta_paths[0]], self.in_size, hidden_size, num_heads, dropout))
        for meta_path in self.meta_paths[1:]:
            self.layers.append(HANLayer([meta_path], self.in_size, hidden_size, num_heads, dropout))
        self.classify = nn.Linear(hidden_size * num_heads , out_size)

    # HAN defined metapaths are the path between the same type nodes.
    # We need undirect the global graph.
    def reflect_graph(self, nx_g_data):
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
    def get_symmatrical_metapaths(self):
        meta_paths = []
        for mt in self.symmetrical_global_graph.canonical_etypes:
            if mt[0] == mt[1]:
                ref_mt = [mt]
            else:
                ref_mt = [mt, mt[::-1]]
            if ref_mt not in meta_paths:
                meta_paths.append(ref_mt)
        return meta_paths

    def _nodetype2onehot(self, ntype):
        feature = torch.zeros(len(self.ntypes_dict), dtype=torch.float)
        feature[self.ntypes_dict[ntype]] = 1
        return feature

    def _han_feature_extractor(self, han_global_graph):
        pass

    def get_node_features(self):
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

    def forward(self):
        features = self.get_node_features()
        targets = []
        hiddens = []
        for ntype, feature in features.items():
            hiddens.append(feature)
            targets.append(self.node_labels[ntype])
        hiddens = torch.cat(hiddens)
        targets = torch.cat(targets)
        output = self.classify(hiddens)
        return output, targets



if __name__ == '__main__':
    from dataloader import EthNodeDataset, EthIdsDataset
    from dgl.dataloading import GraphDataLoader
    dataset = './dataset/node_classification/source_code'
    compressed_graph = './dataset/node_classification/compressed_graphs/compressed_graphs.gpickle'
    labels = './dataset/aggregate/labels.json'
    # ethdataset = EthNodeDataset(dataset, compressed_graph)
    ethdataset = EthIdsDataset(dataset, compressed_graph, labels)
    dataloader = GraphDataLoader(ethdataset, batch_size=256)
    # Get feature extractor
    print('Getting features')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = HANVulNodeClassifier(compressed_graph, node_feature='nodetype', device=device)
    model.to(device)
    # print(model.meta_paths)
    logits, targets = model()
    print(logits.shape)
    print(targets.shape)
    print(model.symmetrical_global_graph.number_of_nodes())
