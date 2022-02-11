from torch_geometric.nn import MetaPath2Vec

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import dgl

from .graph_utils import load_hetero_nx_graph, generate_hetero_graph_data, get_number_of_nodes, reflect_graph, get_symmatrical_metapaths

class VulMetaPath2Vec(nn.Module):
    def __init__(self, compressed_global_graph_path, embedding_dim=128, walk_length=50,
                 context_size=7, walks_per_node=5, num_negative_samples=5,
                 num_nodes_dict=None, sparse=False, device='cpu'):
        super(VulMetaPath2Vec, self).__init__()
        self.compressed_global_graph_path = compressed_global_graph_path
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.num_negative_samples = num_negative_samples
        self.device = device

        # Get Global graph
        nx_graph = load_hetero_nx_graph(compressed_global_graph_path)
        nx_g_data = generate_hetero_graph_data(nx_graph)

        # Reflect graph data
        self.symmetrical_global_graph_data = reflect_graph(nx_g_data)
        self.number_of_nodes = get_number_of_nodes(nx_graph)
        self.symmetrical_global_graph = dgl.heterograph(self.symmetrical_global_graph_data, num_nodes_dict=self.number_of_nodes)
        self.meta_paths = get_symmatrical_metapaths(self.symmetrical_global_graph)
        self.node_types = sorted(list(set([meta_path[0][0] for meta_path in self.meta_paths])))
        self.ntypes_dict = {k: v for v, k in enumerate(self.node_types)}
        max_node_id = {}
        count_node = {}
        for k, v in self.symmetrical_global_graph_data.items():
            if k[0] not in count_node.keys():
                count_node[k[0]] = torch.unique(v[0])
                max_node_id[k[0]] = torch.max(v[0]).item()
            else:
                count_node[k[0]] = torch.unique(torch.cat((count_node[k[0]], torch.unique(v[0]))))
                max_node_id[k[0]] = max(max_node_id[k[0]], torch.max(v[0]).item())
            if k[2] not in count_node.keys():
                count_node[k[2]] = torch.unique(v[1])
                max_node_id[k[2]] = torch.max(v[1])
            else:
                count_node[k[2]] = torch.unique(torch.cat((count_node[k[2]], torch.unique(v[1]))))
                max_node_id[k[2]] = max(max_node_id[k[2]], torch.max(v[1]).item())
        
        self.symmetrical_global_graph = self.symmetrical_global_graph.to(self.device)
        self.extended_metapaths = {}
        for ntype in self.node_types:
            _metapath_list = []
            for metapath in self.meta_paths:
                if metapath[0][0] == ntype:
                    _metapath_list += metapath
            self.extended_metapaths[ntype] = _metapath_list
        self.metapath2vec_dict = {}
        for ntype in list(self.node_types):
            self.metapath2vec_dict[ntype] = MetaPath2Vec(self.symmetrical_global_graph_data ,embedding_dim=128,
                                                                 metapath=self.extended_metapaths[ntype], walk_length=5, context_size=2,
                                                                 walks_per_node=2, num_negative_samples=5,
                                                                 num_nodes_dict=self.number_of_nodes,
                                                                 sparse=False)

        self.layers = nn.ModuleDict(self.metapath2vec_dict)


    def train(self, nodetype, optimizer, log_steps=10):
        embedding = self.layers[nodetype]
        loader = embedding.loader(batch_size=64, shuffle=False)
        embedding.train()
        print(embedding.num_nodes_dict)
        print(embedding.metapath)
        # print(embedding.adj_dict)
        # for k, v in embedding.adj_dict.items():
        #     print(f'{k} - {v.size(1)}')
        total_loss = 0
        for i, (pos_rw, neg_rw) in enumerate(loader):
            print(i)
            optimizer.zero_grad()
            loss = embedding.loss(pos_rw.to(self.device), neg_rw.to(self.device))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (i + 1) % log_steps == 0:
                print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                    f'Loss: {total_loss / log_steps:.4f}'))
                total_loss = 0

if __name__ == '__main__':
    compressed_global_graph_path = './dataset/aggregate/compressed_graph/compressed_graphs.gpickle'
    model = VulMetaPath2Vec(compressed_global_graph_path)
    print(model.layers)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.01)
    for epoch in range(1, 6):
        model.train('BEGIN_LOOP', optimizer)
