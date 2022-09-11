import os

import pickle
import dgl
import math
from matplotlib.pyplot import get
import torch
import networkx as nx
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
from dgl.nn.functional import edge_softmax
from torch_geometric.nn import MetaPath2Vec
from .utils import load_meta_paths
from .graph_utils import add_hetero_ids, \
                         load_hetero_nx_graph, \
                         generate_hetero_graph_data, \
                         get_number_of_nodes, add_cfg_mapping, \
                         get_node_label, get_node_ids_dict, \
                         map_node_embedding, get_symmatrical_metapaths, \
                         reflect_graph, get_node_tracker, get_node_ids_by_filename, \
                         generate_random_node_features, generate_zeros_node_features, \
                         get_length_3_metapath, get_length_2_metapath, \
                         generate_lstm_node_features


class HGTLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 node_dict,
                 edge_dict,
                 n_heads,
                 dropout = 0.2,
                 use_norm = False):
        super(HGTLayer, self).__init__()

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.node_dict     = node_dict
        self.edge_dict     = edge_dict
        self.num_types     = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel     = self.num_types * self.num_relations * self.num_types
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.att           = None

        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()
        self.use_norm    = use_norm

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri   = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.skip           = nn.Parameter(torch.ones(self.num_types))
        self.drop           = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h ):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            for srctype, etype, dsttype in G.canonical_etypes:
                sub_graph = G[srctype, etype, dsttype]
                canonical_etype = (srctype, etype, dsttype)
                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]]
                q_linear = self.q_linears[node_dict[dsttype]]

                k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                q = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k)

                e_id = self.edge_dict[canonical_etype]

                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                sub_graph.srcdata['k'] = k
                sub_graph.dstdata['q'] = q
                sub_graph.srcdata['v_%d' % e_id] = v

                sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
                attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
                attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')

                sub_graph.edata['t'] = attn_score.unsqueeze(-1)

            G.multi_update_all({etype : (fn.u_mul_e('v_%d' % e_id, 't', 'm'), fn.sum('m', 't')) \
                                for etype, e_id in edge_dict.items()}, cross_reducer = 'mean')

            new_h = {}
            for ntype in G.ntypes:
                '''
                    Step 3: Target-specific Aggregation
                    x = norm( W[node_type] * gelu( Agg(x) ) + x )
                '''
                n_id = node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])
                t = G.nodes[ntype].data['t'].view(-1, self.out_dim)
                trans_out = self.drop(self.a_linears[n_id](t))
                trans_out = trans_out * alpha + h[ntype] * (1-alpha)
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out
            return new_h


class HGT(nn.Module):
    def __init__(self, G, node_dict, edge_dict, n_inp, n_hid, n_out, n_layers, n_heads, use_norm = True):
        super(HGT, self).__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.gcs = nn.ModuleList()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_layers = n_layers
        self.adapt_ws  = nn.ModuleList()
        for t in range(len(node_dict)):
            self.adapt_ws.append(nn.Linear(n_inp,   n_hid))
        for _ in range(n_layers):
            self.gcs.append(HGTLayer(n_hid, n_hid, node_dict, edge_dict, n_heads, use_norm = use_norm))
        self.out = nn.Linear(n_hid, n_out)

    def forward(self, G, out_key):
        h = {}
        for ntype in G.ntypes:
            n_id = self.node_dict[ntype]
            h[ntype] = F.gelu(self.adapt_ws[n_id](G.nodes[ntype].data['inp']))
        for i in range(self.n_layers):
            h = self.gcs[i](G, h)
        return self.out(h[out_key])


class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        # W_r for each relation
        self.weight = nn.ModuleDict({
                name : nn.Linear(in_size, out_size) for name in etypes
            })

    def forward(self, G, feat_dict):
        # The input is a dictionary of node features for each type
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            # Compute W_r * h
            Wh = self.weight[etype](feat_dict[srctype])
            # Save it in graph for message passing
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            # Specify per-relation message passing functions: (message_func, reduce_func).
            # Note that the results are saved to the same destination feature 'h', which
            # hints the type wise reducer for aggregation.
            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
        # Trigger message passing of multiple types.
        # The first argument is the message passing functions for each relation.
        # The second one is the type wise reducer, could be "sum", "max",
        # "min", "mean", "stack"
        G.multi_update_all(funcs, 'sum')
        # return the updated node feature dictionary
        return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}


class HeteroRGCN(nn.Module):
    def __init__(self, G, in_size, hidden_size, out_size):
        super(HeteroRGCN, self).__init__()
        # create layers
        self.layer1 = HeteroRGCNLayer(in_size, hidden_size, G.etypes)
        self.layer2 = HeteroRGCNLayer(hidden_size, out_size, G.etypes)

    def forward(self, G, out_key):
        input_dict = {ntype : G.nodes[ntype].data['inp'] for ntype in G.ntypes}
        h_dict = self.layer1(G, input_dict)
        h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
        h_dict = self.layer2(G, h_dict)
        # get appropriate logits
        return h_dict[out_key]


class HGTVulNodeClassifier(nn.Module):
    def __init__(self, compressed_global_graph_path, feature_extractor=None, node_feature='han', hidden_size=128, num_layers=2,num_heads=8, use_norm=True, device='cpu'):
        super(HGTVulNodeClassifier, self).__init__()
        self.compressed_global_graph_path = compressed_global_graph_path
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        # self.source_path = source_path
        # self.extracted_graph = [f for f in os.listdir(self.source_path) if f.endswith('.sol')]
        self.device = device
        # Get Global graph
        nx_graph = load_hetero_nx_graph(compressed_global_graph_path)
        self.nx_graph = nx_graph
        # self.nx_graph = nx_graph.to_undirected()
        nx_g_data = generate_hetero_graph_data(nx_graph)
        self.total_nodes = len(nx_graph)

        # Get Node Labels
        self.node_labels, self.labeled_node_ids, self.label_ids = get_node_label(nx_graph)
        self.node_ids_dict = get_node_ids_dict(nx_graph)

        # Reflect graph data
        self.symmetrical_global_graph_data = reflect_graph(nx_g_data)
        # self.symmetrical_global_graph_data = nx_g_data
        self.number_of_nodes = get_number_of_nodes(nx_graph)
        self.symmetrical_global_graph = dgl.heterograph(self.symmetrical_global_graph_data, num_nodes_dict=self.number_of_nodes, device=device)
        # self.meta_paths = get_symmatrical_metapaths(self.symmetrical_global_graph)
        self.meta_paths = get_length_2_metapath(self.symmetrical_global_graph)
        # Concat the metapaths have the same begin nodetype
        # self.length_3_meta_paths = get_length_3_metapath(self.symmetrical_global_graph)
        self.full_metapath = {}
        for metapath in self.meta_paths:
            ntype = metapath[0][0]
            if ntype not in self.full_metapath:
                self.full_metapath[ntype] = [metapath]
            else:
                self.full_metapath[ntype].append(metapath)
        self.node_types = set([meta_path[0][0] for meta_path in self.meta_paths])
        self.node_types = list(self.symmetrical_global_graph.ntypes)
        # node/edge dictionaries
        self.ntypes_dict = {k: v for v, k in enumerate(self.node_types)}
        self.etypes_dict = {}
        for etype in self.symmetrical_global_graph.canonical_etypes:
            self.etypes_dict[etype] = len(self.etypes_dict)
            self.symmetrical_global_graph.edges[etype].data['id'] = \
                torch.ones(self.symmetrical_global_graph.number_of_edges(etype), 
                        dtype=torch.long, device=device) * self.etypes_dict[etype]

        # Create input node features
        self.node_feature = node_feature
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
            han_features = feature_extractor.get_assemble_node_features()
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
        elif node_feature in ['gae', 'node2vec', 'line']:
            embedding_dim = 128
            self.in_size = embedding_dim
            with open(feature_extractor, 'rb') as f:
                embedding = pickle.load(f, encoding="utf8")
            embedding = torch.tensor(embedding, device=device)
            features = map_node_embedding(nx_graph, embedding)

        self.symmetrical_global_graph = self.symmetrical_global_graph.to(self.device)
        # self.symmetrical_global_graph.ndata['feat'] = features
        for ntype in self.symmetrical_global_graph.ntypes:
            emb = nn.Parameter(features[ntype], requires_grad = False)
            self.symmetrical_global_graph.nodes[ntype].data['inp'] = emb.to(device)

        # Init Model
        self.gcs = nn.ModuleList()
        self.out_size = 2
        self.num_layers = num_layers
        self.adapt_ws  = nn.ModuleList()
        for t in range(len(self.ntypes_dict)):
            self.adapt_ws.append(nn.Linear(self.in_size, self.hidden_size))
        for _ in range(self.num_layers):
            self.gcs.append(HGTLayer(self.hidden_size, self.hidden_size, self.ntypes_dict, self.etypes_dict, self.num_heads, use_norm=use_norm))
        self.classify = nn.Linear(self.hidden_size, self.out_size)

    def extend_forward(self, new_graph):
        nx_graph = new_graph
        nx_graph = nx.convert_node_labels_to_integers(nx_graph)
        nx_graph = add_hetero_ids(nx_graph)
        nx_g_data = generate_hetero_graph_data(nx_graph)

        # Get Node Labels
        node_labels, labeled_node_ids, label_ids = get_node_label(nx_graph)
        print('Bug dict: ', label_ids)
        node_ids_dict = get_node_ids_dict(nx_graph)

        # Reflect graph data
        symmetrical_global_graph_data = reflect_graph(nx_g_data)
        number_of_nodes = get_number_of_nodes(nx_graph)
        symmetrical_global_graph = dgl.heterograph(symmetrical_global_graph_data, num_nodes_dict=number_of_nodes, device=self.device)
        # Create input node features
        features = {}
        if self.node_feature == 'nodetype':
            for ntype in self.symmetrical_global_graph.ntypes:
                features[ntype] = self._nodetype2onehot(ntype).repeat(symmetrical_global_graph.num_nodes(ntype), 1).to(self.device)
            self.in_size = len(self.node_types)
        for ntype in self.symmetrical_global_graph.ntypes:
            emb = nn.Parameter(features[ntype], requires_grad = False)
            symmetrical_global_graph.nodes[ntype].data['inp'] = emb.to(self.device)

        h = {}
        hiddens = torch.zeros((symmetrical_global_graph.number_of_nodes(), self.hidden_size), device=self.device)
        for ntype in symmetrical_global_graph.ntypes:
            n_id = self.ntypes_dict[ntype]
            h[ntype] = F.gelu(self.adapt_ws[n_id](symmetrical_global_graph.nodes[ntype].data['inp']))
        for i in range(self.num_layers):
            h = self.gcs[i](symmetrical_global_graph, h)
        for ntype, feature in h.items():
            assert len(node_ids_dict[ntype]) == feature.shape[0]
            hiddens[node_ids_dict[ntype]] = feature
        output = self.classify(hiddens)
        return output, node_labels

    def _nodetype2onehot(self, ntype):
        feature = torch.zeros(len(self.ntypes_dict), dtype=torch.float)
        feature[self.ntypes_dict[ntype]] = 1
        return feature

    def reset_parameters(self):
        for model in self.adapt_ws:
            for layer in model.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        for model in self.gcs:
            for layer in model.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        for layer in self.classify.children():
            if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

    def forward(self):
        h = {}
        hiddens = torch.zeros((self.symmetrical_global_graph.number_of_nodes(), self.hidden_size), device=self.device)
        for ntype in self.symmetrical_global_graph.ntypes:
            n_id = self.ntypes_dict[ntype]
            h[ntype] = F.gelu(self.adapt_ws[n_id](self.symmetrical_global_graph.nodes[ntype].data['inp']))
        for i in range(self.num_layers):
            h = self.gcs[i](self.symmetrical_global_graph, h)
        for ntype, feature in h.items():
            assert len(self.node_ids_dict[ntype]) == feature.shape[0]
            hiddens[self.node_ids_dict[ntype]] = feature
        output = self.classify(hiddens)
        return output


class HGTVulGraphClassifier(nn.Module):
    def __init__(self, compressed_global_graph_path, feature_extractor=None, node_feature='nodetype', hidden_size=128, num_layers=2,num_heads=8, use_norm=True, device='cpu'):
        super(HGTVulGraphClassifier, self).__init__()
        self.compressed_global_graph_path = compressed_global_graph_path
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        # self.source_path = source_path
        # self.extracted_graph = [f for f in os.listdir(self.source_path) if f.endswith('.sol')]
        # self.filename_mapping = {file: idx for idx, file in enumerate(self.extracted_graph)}
        self.device = device
        # Get Global graph
        nx_graph = load_hetero_nx_graph(compressed_global_graph_path)
        self.nx_graph = nx_graph
        nx_g_data = generate_hetero_graph_data(nx_graph)
        self.total_nodes = len(nx_graph)

        # Get Node Labels
        self.node_ids_dict = get_node_ids_dict(nx_graph)
        # _node_tracker = get_node_tracker(nx_graph, self.filename_mapping)
        self.node_ids_by_filename = get_node_ids_by_filename(nx_graph)
        # Reflect graph data
        self.symmetrical_global_graph_data = reflect_graph(nx_g_data)
        # self.symmetrical_global_graph_data = nx_g_data
        self.number_of_nodes = get_number_of_nodes(nx_graph)
        self.symmetrical_global_graph = dgl.heterograph(self.symmetrical_global_graph_data, num_nodes_dict=self.number_of_nodes)
        # self.symmetrical_global_graph.ndata['filename'] = _node_tracker
        self.symmetrical_global_graph = self.symmetrical_global_graph.to(device)
        self.meta_paths = get_symmatrical_metapaths(self.symmetrical_global_graph)
        # print(self.meta_paths)
        # self.length_3_meta_paths = get_length_3_metapath(self.symmetrical_global_graph)
        # self.length_2_meta_paths = get_length_2_metapath(self.symmetrical_global_graph)
        # self.meta_paths = self.length_3_meta_paths
        # self.meta_paths = self.length_2_meta_paths + self.length_3_meta_paths
        # Concat the metapaths have the same begin nodetype
        self.full_metapath = {}
        for metapath in self.meta_paths:
            ntype = metapath[0][0]
            if ntype not in self.full_metapath:
                self.full_metapath[ntype] = [metapath]
            else:
                self.full_metapath[ntype].append(metapath)
        self.node_types = set([meta_path[0][0] for meta_path in self.meta_paths])
        self.node_types = list(self.symmetrical_global_graph.ntypes)
        # node/edge dictionaries
        self.ntypes_dict = {k: v for v, k in enumerate(self.node_types)}
        self.etypes_dict = {}
        for etype in self.symmetrical_global_graph.canonical_etypes:
            self.etypes_dict[etype] = len(self.etypes_dict)
            self.symmetrical_global_graph.edges[etype].data['id'] = \
                torch.ones(self.symmetrical_global_graph.number_of_edges(etype), 
                        dtype=torch.long, device=device) * self.etypes_dict[etype]

        # Create input node features
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
        elif node_feature in ['gae', 'node2vec', 'line']:
            embedding_dim = 128
            self.in_size = embedding_dim
            with open(feature_extractor, 'rb') as f:
                embedding = pickle.load(f, encoding="utf8")
            embedding = torch.tensor(embedding, device=device)
            features = map_node_embedding(nx_graph, embedding)
        elif node_feature == 'random':
            embedding_dim = int(feature_extractor)
            self.in_size = embedding_dim
            features = generate_random_node_features(nx_graph, self.in_size)
            features = {k: v.to(self.device) for k, v in features.items()}
        elif node_feature == 'zeros':
            embedding_dim = int(feature_extractor)
            self.in_size = embedding_dim
            features = generate_zeros_node_features(nx_graph, self.in_size)
            features = {k: v.to(self.device) for k, v in features.items()}
        elif node_feature == 'lstm':
            embedding_dim = int(feature_extractor)
            self.in_size = embedding_dim
            features = generate_lstm_node_features(nx_graph)
            features = {k: v for k, v in features.items()}

        self.symmetrical_global_graph = self.symmetrical_global_graph.to(self.device)
        self.symmetrical_global_graph.ndata['feat'] = features
        for ntype in self.symmetrical_global_graph.ntypes:
            emb = nn.Parameter(features[ntype], requires_grad = False)
            self.symmetrical_global_graph.nodes[ntype].data['inp'] = emb.to(device)

        # Init Model
        self.gcs = nn.ModuleList()
        self.out_size = 2
        self.num_layers = num_layers
        self.adapt_ws  = nn.ModuleList()
        for t in range(len(self.ntypes_dict)):
            self.adapt_ws.append(nn.Linear(self.in_size, self.hidden_size))
        for _ in range(self.num_layers):
            self.gcs.append(HGTLayer(self.hidden_size, self.hidden_size, self.ntypes_dict, self.etypes_dict, self.num_heads, use_norm=use_norm))
        self.classify = nn.Linear(self.hidden_size, self.out_size)

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

    def reset_parameters(self):
        for model in self.adapt_ws:
            for layer in model.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        for model in self.gcs:
            for layer in model.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        for layer in self.classify.children():
            if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

    def forward(self, batched_g_name, save_featrues=None):
        h = {}
        hiddens = torch.zeros((self.symmetrical_global_graph.number_of_nodes(), self.hidden_size), device=self.device)
        for ntype in self.symmetrical_global_graph.ntypes:
            n_id = self.ntypes_dict[ntype]
            h[ntype] = F.gelu(self.adapt_ws[n_id](self.symmetrical_global_graph.nodes[ntype].data['inp']))
        for i in range(self.num_layers):
            h = self.gcs[i](self.symmetrical_global_graph, h)
        for ntype, feature in h.items():
            assert len(self.node_ids_dict[ntype]) == feature.shape[0]
            hiddens[self.node_ids_dict[ntype]] = feature
        batched_graph_embedded = []
        for g_name in batched_g_name:
            node_list = self.node_ids_by_filename[g_name]
            batched_graph_embedded.append(hiddens[node_list].mean(0).tolist())
        batched_graph_embedded = torch.tensor(batched_graph_embedded).to(self.device)
        if save_featrues:
            torch.save(batched_graph_embedded, save_featrues)
        output = self.classify(batched_graph_embedded)
        return output, batched_graph_embedded


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    compressed_graph = './experiments/ge-sc-data/source_code/reentrancy/buggy_curated/cfg_cg_compressed_graphs.gpickle'
    dataset = './experiments/ge-sc-data/source_code/access_control/clean_57_buggy_curated_0'
    node_feature = 'nodetype'
    feature_extractor = None
    model = HGTVulNodeClassifier(compressed_graph, feature_extractor=None, node_feature=node_feature, device=device).to(device)
    # model.train()
    # logits = model()
    # print(model.meta_paths)
    # print(len(model.length_3_meta_paths))
    load_meta_paths('./metapath_length_3.txt')
