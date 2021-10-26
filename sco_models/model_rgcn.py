import os

import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch_geometric.nn import MetaPath2Vec
import dgl
import dgl.nn.pytorch as dglnn

from .graph_utils import load_hetero_nx_graph, generate_hetero_graph_data, reflect_graph, \
                         get_number_of_nodes, add_cfg_mapping, get_symmatrical_metapaths, get_node_tracker


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
    def __init__(self,compressed_global_graph_path, source_path, feature_extractor=None, node_feature='nodetype', hidden_dim=32, out_size=2, device='cpu'):
        super(RGCNVulClassifier, self).__init__()
        self.compressed_global_graph_path = compressed_global_graph_path
        self.hidden_dim = hidden_dim
        self.source_path = source_path
        self.extracted_graph = [f for f in os.listdir(self.source_path) if f.endswith('.sol')]
        self.filename_mapping = {file: idx for idx, file in enumerate(self.extracted_graph)}
        self.device = device
        # Get Global graph
        nx_graph = load_hetero_nx_graph(compressed_global_graph_path)
        nx_g_data = generate_hetero_graph_data(nx_graph)
        _node_tracker = get_node_tracker(nx_graph, self.filename_mapping)
        self.symmetrical_global_graph_data = reflect_graph(nx_g_data)
        self.number_of_nodes = get_number_of_nodes(nx_graph)
        self.symmetrical_global_graph = dgl.heterograph(self.symmetrical_global_graph_data, num_nodes_dict=self.number_of_nodes)
        self.symmetrical_global_graph.ndata['filename'] = _node_tracker
        self.meta_paths = get_symmatrical_metapaths(self.symmetrical_global_graph)
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
                        sparse=True)
                ntype = metapath[0][0]
                if ntype not in features.keys():
                    features[ntype] = _metapath_embedding(ntype).unsqueeze(0)
                else:
                    features[ntype] = torch.cat((features[ntype], _metapath_embedding(ntype).unsqueeze(0)))
            features = {k: torch.mean(v, dim=0).to(self.device) for k, v in features.items()}
        
        self.symmetrical_global_graph = self.symmetrical_global_graph.to(self.device)
        self.symmetrical_global_graph.ndata['feat'] = features

        self.rgcn = RGCN(self.in_size, self.hidden_dim, self.hidden_dim, self.edge_types)
        self.classify = nn.Linear(self.hidden_dim, out_size)

    def _nodetype2onehot(self, ntype):
        feature = torch.zeros(len(self.ntypes_dict), dtype=torch.float)
        feature[self.ntypes_dict[ntype]] = 1
        return feature

    def forward(self, batched_graph):
        batched_graph_embedded = []
        for g_name in batched_graph:
            file_ids = self.filename_mapping[g_name]
            node_mask = {}
            for node_type in self.node_types:
                node_mask[node_type] = self.symmetrical_global_graph.ndata['filename'][node_type] == file_ids
            sub_graph = dgl.node_subgraph(self.symmetrical_global_graph, node_mask)
            h = sub_graph.ndata['feat']
            h = self.rgcn(sub_graph, h)
            graph_embedded = 0
            for node_type, feature in h.items():
                graph_embedded += feature.mean(0)
            if isinstance(graph_embedded, int):
                batched_graph_embedded.append([0] * self.hidden_dim)
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
