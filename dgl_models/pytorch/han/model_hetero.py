"""This model shows an example of using dgl.metapath_reachable_graph on the original heterogeneous
graph.

Because the original HAN implementation only gives the preprocessed homogeneous graph, this model
could not reproduce the result in HAN as they did not provide the preprocessing code, and we
constructed another dataset from ACM with a different set of papers, connections, features and
labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import dgl
from dgl.nn.pytorch import GATConv

from graph_utils import add_hetero_ids, generate_hetero_graph_data, get_number_of_nodes


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


class HAN(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(meta_paths, hidden_size * num_heads[l-1],
                                        hidden_size, num_heads[l], dropout))
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)

        return self.predict(h)


class HANVulClassifier(nn.Module):
    def __init__(self, compressed_global_graph_path, filename_mapping, in_size, hidden_size, out_size, num_heads, dropout, device):
        super(HANVulClassifier, self).__init__()
        self.filename_mapping = filename_mapping
        self.device = device
        # Get Global graph
        nx_graph = nx.read_gpickle(compressed_global_graph_path)
        nx_graph = nx.convert_node_labels_to_integers(nx_graph)
        nx_graph = add_hetero_ids(nx_graph)
        nx_g_data, _node_tracker = generate_hetero_graph_data(nx_graph, filename_mapping)

        # Reflect graph data
        self.symmetrical_global_graph_data = self.reflect_graph(nx_g_data)
        self.number_of_nodes = get_number_of_nodes(nx_graph)
        self.symmetrical_global_graph = dgl.heterograph(self.symmetrical_global_graph_data, num_nodes_dict=self.number_of_nodes)
        self.symmetrical_global_graph.ndata['filename'] = _node_tracker
        self.meta_paths = self.get_symmatrical_metapaths()
        self.node_types = set([meta_path[0][0] for meta_path in self.meta_paths])
        self.ntypes_dict = {k: v for v, k in enumerate(self.node_types)}
        features = {}
        for ntype in self.symmetrical_global_graph.ntypes:
            features[ntype] = self.nodetype2onehot(ntype).repeat(self.symmetrical_global_graph.num_nodes(ntype), 1)
        self.symmetrical_global_graph.ndata['feat'] = features

        # Init Model
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer([self.meta_paths[0]], in_size, hidden_size, num_heads, dropout))
        for meta_path in self.meta_paths[1:]:
            self.layers.append(HANLayer([meta_path], in_size, hidden_size, num_heads, dropout))
        self.features = {}
        for han in self.layers:
            ntype = han.meta_paths[0][0][0]
            self.features[ntype] = han(self.symmetrical_global_graph, self.symmetrical_global_graph.ndata['feat'][ntype])

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

    def nodetype2onehot(self, ntype):
        feature = torch.zeros(len(self.ntypes_dict), dtype=torch.float)
        feature[self.ntypes_dict[ntype]] = 1
        return feature

    def forward(self, batched_g_name):
        batched_graph_embedded = []
        for g_name in batched_g_name:
            file_ids = self.filename_mapping[g_name]
            graph_embedded = 0
            for node_type in self.node_types:
                file_mask = self.symmetrical_global_graph.ndata['filename'][node_type] == file_ids
                if file_mask.sum().item() != 0:
                    graph_embedded += self.features[node_type][file_mask].mean(0)
            batched_graph_embedded.append(graph_embedded.tolist())
        batched_graph_embedded = torch.tensor(batched_graph_embedded).to(self.device)
        output = self.classify(batched_graph_embedded)
        return output
