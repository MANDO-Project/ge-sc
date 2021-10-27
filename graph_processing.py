# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Import block

# +
import os
from os.path import join
import shutil

import json
import torch
import dgl
import networkx as nx
from torch import nn
from torch.nn.functional import cross_entropy, relu, softmax, log_softmax, one_hot

from torch.utils.tensorboard import SummaryWriter

import dgl.nn.pytorch as dglnn
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from dgl import graph
from tqdm import tqdm

from copy import deepcopy
from slither.slither import Slither
from slither.core.cfg.node import NodeType


# -

# !which solc && solc --version

# + tags=[]
# from solc import install_solc
# install_solc('v0.4.24')
# -

# # Define functions

# +
def add_node_type_feature(nx_graph):
    nx_g = nx_graph
    list_node_type = []
    node_type_feat_attrs = dict()
    for node, data in nx_graph.nodes(data=True):
        if data.get('node_type') is not None:
            if data['node_type'] not in list_node_type:
                list_node_type.append(data['node_type'])
            node_type_feat = torch.tensor(list_node_type.index(data['node_type']), dtype=torch.int64)
            node_type_feat_attrs[node] = node_type_feat
            # print(node_type_feat)

    nx.set_node_attributes(nx_g, node_type_feat_attrs, '_TYPE')

    return nx_g, list_node_type

def add_edge_type_feature(nx_graph):
    nx_g = nx_graph
    list_edge_type = []

    for source, target, data in nx_graph.edges(data=True):
        if data.get('edge_type') is not None:
            if data['edge_type'] not in list_edge_type:
                list_edge_type.append(data['edge_type'])
            edge_type_feat = torch.tensor(list_edge_type.index(data['edge_type']), dtype=torch.int64)
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
        dict_three_cannonical_egdes[key] = (torch.tensor(list_source, dtype=torch.int64), torch.tensor(list_target, dtype=torch.int64))

    return dict_three_cannonical_egdes

def generate_hetero_graph_data(nx_graph):
    nx_g = nx_graph
    dict_three_cannonical_egdes = dict()
    for source, target, data in nx_g.edges(data=True):
        edge_type = data['edge_type']
        source_node_type = nx_g.nodes[source]['node_type']
        target_node_type = nx_g.nodes[target]['node_type']
        three_cannonical_egde = (source_node_type, edge_type, target_node_type)
        # print(dict_three_cannonical_egdes)
        # print(three_cannonical_egde, source, target)
        if three_cannonical_egde not in dict_three_cannonical_egdes.keys():
            dict_three_cannonical_egdes[three_cannonical_egde] = [(source, target)]
        else:
            current_val = dict_three_cannonical_egdes[three_cannonical_egde]
            temp_edge = (source, target)
            current_val.append(temp_edge)
            dict_three_cannonical_egdes[three_cannonical_egde] = current_val
    
    dict_three_cannonical_egdes = convert_edge_data_to_tensor(dict_three_cannonical_egdes)

    return dict_three_cannonical_egdes



# -

def add_full_metapath(hete_graph_data, metapaths):
    for metapath in metapaths:
        if metapath not in hete_graph_data.keys():
            hete_graph_data[metapath] = (torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64))
    return hete_graph_data


def get_full_graph(contract_path):
#     print(contract_path)
    slither = Slither(contract_path, solc="/home/minhnn/.py-solc/solc-v0.4.24/bin/solc")
    merge_contract_graph = None
    for contract in slither.contracts:
        merged_graph = None
        for function in contract.functions + contract.modifiers:
            if len(function.nodes) == 0:
                continue
            nx_g = nx.MultiDiGraph()
            for node in function.nodes:
#                 print('Node:', node, 'NodeType:', node.type, 'NodeExpression:', node.expression)
                node_label = "Node Type: {}\n".format(str(node.type))
                node_type = str(node.type)
                if node.expression:
                    node_label += "\nEXPRESSION:\n{}\n".format(node.expression)
                    node_expression = str(node.expression)
                else:
                    node_expression = None
                if node.irs:
                    node_label += "\nIRs:\n" + "\n".join([str(ir) for ir in node.irs])
                    node_irs = "\n".join([str(ir) for ir in node.irs])
                else:
                    node_irs = None
                nx_g.add_node(node.node_id, label=node_label,
                              node_type=node_type, node_expression=node_expression, node_irs=node_irs,
                              function_fullname=function.full_name, contract_name=contract.name)

                if node.type in [NodeType.IF, NodeType.IFLOOP]:
                    true_node = node.son_true
                    if true_node:
                        nx_g.add_edge(node.node_id, true_node.node_id, edge_type='if_true', label='True')
                    false_node = node.son_false
                    if false_node:
                        nx_g.add_edge(node.node_id, false_node.node_id, edge_type='if_false', label='False')
                else:
                    for son in node.sons:
                        nx_g.add_edge(node.node_id, son.node_id, edge_type='next', label='Next')
            nx_graph = nx_g
#             print(nx.info(nx_graph))
            # add FUNCTION_NAME node
            nx_graph.add_node('function.name', label=contract.name + '_' + function.full_name,
                              node_type='FUNCTION_NAME', node_expression=None, node_irs=None,
                              function_fullname=function.full_name, contract_name=contract.name)
            nx_graph.add_edge('function.name', 0, edge_type='next', label='Next')
            
            if merged_graph is None:
                nx_graph = nx.relabel_nodes(nx_graph, lambda x: contract.name + '_' + function.name + '_' + str(x), copy=False)
                merged_graph = deepcopy(nx_graph)
            else:
                nx_graph = nx.relabel_nodes(nx_graph, lambda x: contract.name + '_' + function.name + '_' + str(x), copy=False)
                merged_graph = nx.disjoint_union(merged_graph, nx_graph)
#             print('merged_graph: ', nx.info(merged_graph))
        if merge_contract_graph is None:
            merge_contract_graph = deepcopy(merged_graph)
        elif merged_graph is not None:
            merge_contract_graph = nx.disjoint_union(merge_contract_graph, merged_graph)
#     print(nx.infor(merge_contract_graph))
    return merge_contract_graph


# # Retrieve graph structure

# ## Get meta-path

smart_contract_path = '/home/minhnn/minhnn/ICSE/datasets/Etherscan_Contract/source_code'
smart_contracts = sorted(sorted([f for f in os.listdir(smart_contract_path) if f.endswith('.sol')]), key=len)
len(smart_contracts)

meta_path_types  = []
extracted_contracts = []
excepted_contracts = []

# + tags=[]
for sc in tqdm(smart_contracts):
    sc_path = join(smart_contract_path, sc)
    try:
        full_graph = get_full_graph(sc_path)
        nx.write_gpickle(full_graph, join(smart_contract_path, '../extracted_graph', sc.replace('.sol', '.gpickle')))
        full_graph, list_node_type = add_node_type_feature(full_graph)
        full_graph, list_edge_type = add_edge_type_feature(full_graph)
        full_graph = nx.convert_node_labels_to_integers(full_graph)
    #     print("graph info: ", nx.info(full_graph))
    #     for graph in full_graph.nodes(data=True):
    #         print(graph)
        nx_g_data = generate_hetero_graph_data(full_graph)
        for meta_path in nx_g_data.keys():
            if meta_path not in meta_path_types:
                meta_path_types.append(meta_path)
        extracted_contracts.append(sc)
    except:
        excepted_contracts.append(sc)
len(meta_path_types)
# -

print("Extracted/Excepted contracts: {}/{}".format(len(extracted_contracts), len(excepted_contracts) + len(extracted_contracts)))

# + tags=[]
metapath_path = '/home/minhnn/minhnn/ICSE/ge-sc/metapaths.txt'
meta_path_str = [str(mt) for mt in meta_path_types]
with open(metapath_path, 'w') as f:
    f.write('\n'.join([str(meta_path) for meta_path in meta_path_types]))
# + tags=[]
compressed_graph_path = '/home/minhnn/minhnn/ICSE/ge-sc/outputs/compress_graphs.gpickle'
nx_graph = nx.read_gpickle(compressed_graph_path)
# nx_graph, list_node_type = add_node_type_feature(nx_graph)
# nx_graph, list_edge_type = add_edge_type_feature(nx_graph)
nx_graph = nx.convert_node_labels_to_integers(nx_graph)
nx_g_data = generate_hetero_graph_data(nx_graph)
dgl_hete_graph = dgl.heterograph(nx_g_data)
print(dgl_hete_graph)
print(dgl_hete_graph.ntypes, dgl_hete_graph.num_nodes())
print(dgl_hete_graph.etypes, dgl_hete_graph.num_edges())
# -


# ## Get node types

# + tags=[]
ntypes = list(set([e[0] for e in meta_path_types] + [e[2] for e in meta_path_types]))
len(ntypes), ntypes

# + tags=[]
ntypes_dict = {k: v for v, k in enumerate(ntypes)}
ntypes_dict, len(ntypes_dict)

# + tags=[]
ntypes_dest_dict = {k: ... for k in ntypes}
ntypes_dest_dict, len(ntypes_dest_dict)


# -

def nodetype2onehot(ntype, ntypes_dicts):
    feature = torch.zeros(len(ntypes_dicts), dtype=torch.float)
    feature[ntypes_dicts[ntype]] = 1
    return feature
nodetype2onehot('FUNCTION_NAME', ntypes_dict)

# ## GET edge types

etypes = list(set([e[1] for e in meta_path_types]))
len(etypes), etypes

# # Data Loader

pickle_path = '/home/minhnn/minhnn/ICSE/datasets/Etherscan_Contract/extracted_graph'
pickle_files = sorted(sorted([f for f in os.listdir(pickle_path) if f.endswith('.gpickle')]), key=len)
len(pickle_files)

nx_g_data = add_full_metapath(nx_g_data, meta_path_types)

# + tags=[]
label_path = '/home/minhnn/minhnn/ICSE/datasets/Etherscan_Contract/Reentrancy_AutoExtract_corenodes.json'
with open(label_path, 'r') as f:
    content = f.readlines()
label_dict = {}
for l in content:
    sc = json.loads(l.strip('\n').strip(','))
    label_dict[sc['contract_name']] = sc['targets']
label_dict['No_Reentrance.sol'] = '0'
# -

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

# + tags=[]
"""QM7b dataset for graph property prediction (regression)."""
import numpy as np
import os
import json

from torch_geometric.nn import MetaPath2Vec

class EtherumSmartContract(DGLDataset):
    _url = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/' \
           'datasets/qm7b.mat'
    _sha1_str = '4102c744bb9d6fd7b40ac67a300e49cd87e28392'
    _label = '/home/minhnn/minhnn/ICSE/datasets/Etherscan_Contract/Reentrancy_AutoExtract_corenodes.json'
    _data_path = '/home/minhnn/minhnn/ICSE/datasets/Etherscan_Contract/extracted_graph'

    def __init__(self, raw_dir=None, force_reload=False, verbose=False):
        super(EtherumSmartContract, self).__init__(name='ethsc',
                                          url=self._url,
                                          raw_dir=raw_dir,
                                          force_reload=force_reload,
                                          verbose=verbose)

    def process(self):
        self.graphs, self.label = self._load_graph()

    def _load_graph(self):
        extracted_graph = [f for f in os.listdir(self._data_path) if f.endswith('.gpickle')]
        num_graphs = len(extracted_graph)
        graphs = []
        labels = []
        for i in range(num_graphs):
            nx_graph = nx.read_gpickle(join(self._data_path, extracted_graph[i]))
            nx_graph, list_node_type = add_node_type_feature(nx_graph)
            nx_graph, list_edge_type = add_edge_type_feature(nx_graph)
            nx_graph = nx.convert_node_labels_to_integers(nx_graph)
            nx_g_data = generate_hetero_graph_data(nx_graph)
            geo_g_data = {}
            for k, v in nx_g_data.items():
                geo_g_data[k] = torch.stack(list(v), dim=0)
            
            for k, v in geo_g_data.items():
                if len(v[0]) == 0:
                    print(k)
            
            geo_meta_path_types = list(geo_g_data.keys())
            bidirect_geo_meta_path_types = geo_meta_path_types + [t[::-1] for t in geo_meta_path_types[::-1]]
            
            metapath_embedding = MetaPath2Vec(geo_g_data, embedding_dim=128,
                     metapath=bidirect_geo_meta_path_types, walk_length=2, context_size=2,
                     walks_per_node=1, num_negative_samples=1, num_nodes_dict=None,
                     sparse=True).to(device).eval()
            
            nx_g_data = add_full_metapath(nx_g_data, meta_path_types)
            dgl_hete_graph = dgl.heterograph(nx_g_data).to(device)
            feature_data = {}
            h_data = {}
            
            for ntype in dgl_hete_graph.ntypes:
                feature_data[ntype] = nodetype2onehot(ntype, ntypes_dict).repeat(dgl_hete_graph.num_nodes(ntype), 1)
#                 if ntype in list(metapath_embedding.num_nodes_dict.keys()):
#                     feature_data[ntype] = metapath_embedding(ntype)
#                 else:
#                     feature_data[ntype] = torch.zeros((dgl_hete_graph.num_nodes(ntype), 128), device='cuda')
#                 h_data[ntype] = torch.tensor([], dtype=torch.int64).repeat(dgl_hete_graph.num_nodes(ntype), 1)
                
            dgl_hete_graph.ndata['feat'] = feature_data
#             dgl_hete_graph.ndata['h'] = h_data
            graphs.append(dgl_hete_graph)
            labels.append(int(label_dict[extracted_graph[i].replace('.gpickle', '.sol')]))
        labels = torch.tensor(labels, dtype=torch.int64).to(device)
#         print(graphs[0].ndata)
        return graphs, labels


    @property
    def num_labels(self):
        return 2

    def __getitem__(self, idx):
        return self.graphs[idx], self.label[idx]

    def __len__(self):
        return len(self.graphs)

Ethdataset = EtherumSmartContract()

# +
# import dgl.data
# dataset = dgl.data.GINDataset('MUTAG', False)

dataloader = GraphDataLoader(
    Ethdataset,
    batch_size=8,
    drop_last=False,
    shuffle=True)

# + tags=[]
for batched_graph, labels in dataloader:
    for k, v in batched_graph.ndata['feat'].items():
        print(k, v.get_device())
    print(len(batched_graph.ndata['feat'].items()))
    for k, v in batched_graph.ndata['feat'].items():
        print(k, v.shape)


# + tags=[]
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
#         print(inputs.get_device())
        h = self.conv1(graph, inputs)
        h = {k: relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h

class HeteroClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
        super().__init__()
        self.rgcn = RGCN(in_dim, hidden_dim, hidden_dim, rel_names)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        h = g.ndata['feat']
        h = self.rgcn(g, h)
        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by average readout.
            hg = 0
            for ntype in h.keys():
                hg = hg + dgl.mean_nodes(g, 'h', ntype=ntype)
            return self.classify(hg)


# -

def accuracy(preds, labels):
    return (preds == labels).sum().item() / labels.shape[0]


etypes

tensorboard_path = '/home/minhnn/minhnn/ICSE/ge-sc/logs/MetaPath2Vec_ConvHete'
writer = SummaryWriter(tensorboard_path)

# + tags=[]
# etypes is the list of edge types as strings.
model = HeteroClassifier(128, 32, 2, etypes).to(device)
opt = torch.optim.Adam(model.parameters(),  lr=0.0005)
for epoch in range(100):
    total_loss = 0
    train_acc = 0
    steps = 0
    for idx, (batched_graph, labels) in enumerate(dataloader):
        logits = model(batched_graph)
        preds = logits.argmax(dim=1)
        train_acc += accuracy(preds, labels)
        loss = cross_entropy(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
        steps += 1
    print('train_loss: {:4f} - train_acc: {:4f}'.format(total_loss/steps, train_acc/steps))
#     writer.add_scalar('Loss/train', total_loss/steps, epoch)
#     writer.add_scalar('Accuracy/train', train_acc/steps, epoch)
# writer.close()

# + [markdown] tags=[]
# ## Cross validate
# -

from sklearn.model_selection import KFold

k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True)

dataloader = GraphDataLoader(
    Ethdataset,
    batch_size=8,
    drop_last=False,
    shuffle=True,
    sampler=test_subsampler)

num_graphs

epochs = 80

# + tags=[] jupyter={"outputs_hidden": true}
train_results = {}
test_results = {}
for fold, (train_ids, test_ids) in enumerate(kfold.split(range(num_graphs))):
    train_results[fold] = {'loss': [], 'acc': []}
    test_results[fold] = {'loss': [], 'acc': []}
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    train_dataloader = GraphDataLoader(
    Ethdataset,
    batch_size=128,
    drop_last=False,
    sampler=train_subsampler)
    test_dataloader = GraphDataLoader(
    Ethdataset,
    batch_size=128,
    drop_last=False,
    sampler=test_subsampler)
    print('Start training fold {} with {}/{} train/test smart contracts'.format(fold, len(train_dataloader), len(test_dataloader)))
    total_steps = len(train_dataloader) * epochs
    model = HeteroClassifier(128, 32, 2, etypes).to(device)
    opt = torch.optim.Adam(model.parameters(),  lr=0.0005)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.01, total_steps=total_steps)
    lrs = []
    for epoch in range(epochs):
        print('Fold {} - Epochs {}'.format(fold, epoch))
        total_loss = 0
        train_acc = 0
        steps = 0
        for idx, (batched_graph, labels) in enumerate(train_dataloader):
            logits = model(batched_graph)
            preds = logits.argmax(dim=1)
            train_acc += accuracy(preds, labels)
            loss = cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
            total_loss += loss.item()
            steps += 1
            lrs.append(opt.param_groups[0]["lr"])
        print('train_loss: {:4f} - train_acc: {:4f}'.format(total_loss/steps, train_acc/steps))
        train_results[fold]['loss'].append(total_loss/steps)
        train_results[fold]['acc'].append(train_acc/steps)

        with torch.no_grad():
            total_loss = 0
            test_acc = 0
            steps = 0
            for idx, (batched_graph, labels) in enumerate(test_dataloader):
                logits = model(batched_graph)
                preds = logits.argmax(dim=1)
                test_acc += accuracy(preds, labels)
                loss = cross_entropy(logits, labels)
                total_loss += loss.item()
                steps += 1
            print('valid_loss: {:4f} - valid_acc: {:4f}'.format(total_loss/steps, test_acc/steps))
            test_results[fold]['loss'].append(total_loss/steps)
            test_results[fold]['acc'].append(test_acc/steps)
    print('Saving model fold {}'.format(fold))
    save_path = f'./models/model_conv_fold_{fold}.pth'
    torch.save(model.state_dict(), save_path)
# -

print('Start training fold {} with {}/{} train/test smart contracts'.format(fold, len(train_ids), len(test_ids)))

print(len(lrs))

# +
tensorboard_path = '/home/minhnn/minhnn/ICSE/ge-sc/logs/MetaPath2Vec_ConvHete_CrossVal'
writer = SummaryWriter(tensorboard_path)
tensorboard_acc = {'train': train_results[0]['acc'], 'valid': test_results[0]['acc']}
tensorboard_loss = {'train': train_results[0]['loss'], 'valid': test_results[0]['loss']}
# for key, results in train_results[0].items():
#     tensorboard_acc[] = 
#     writer.add_scalars('Loss', train_res, epoch)
# for idx, lr in enumerate(lrs):
#     writer.add_scalar('Learning rate', lr, idx)
for idx, lr in enumerate(lrs):
    writer.add_scalar('Learning rate', lr, idx)

for fold in range(k_folds):
    for idx in range(epochs):
        writer.add_scalars('Accuracy', {f'train_{fold+1}': train_results[fold]['acc'][idx],
                                        f'valid_{fold+1}': test_results[fold]['acc'][idx]}, idx)
        writer.add_scalars('Loss', {f'train_{fold+1}': train_results[fold]['loss'][idx],
                                    f'valid_{fold+1}': test_results[fold]['loss'][idx]}, idx)
writer.close()


# -

# # Embedding

# + tags=[]
def get_num_node_dict(g_data):
    num_node_dict = {}
    for k, v in g_data.items():
        if not num_node_dict.get(k[0]):
            num_node_dict[k[0]] = v[0].shape[0]
        else:
            num_node_dict[k[0]] += v[0].shape[0]
        if not num_node_dict.get(k[2]):
            num_node_dict[k[2]] = v[1].shape[0]
        else:
            num_node_dict[k[2]] += v[1].shape[0]
    return num_node_dict
get_num_node_dict(nx_g_data)

# + tags=[]
# convert dgl to geomatric graph format
nx_g_data
geo_g_data = {}
for k, v in nx.items():
    geo_g_data[k] = torch.stack(list(v), dim=0)

print(geo_g_data)
# -

# get num node dict of graph sample
num_nodes_dict = {}
for n in list_node_type:
    num_nodes_dict[n] = dgl_hete_graph.number_of_nodes(n)
num_nodes_dict

# +
import os.path as osp

import torch
from torch_geometric.datasets import AMiner
from torch_geometric.nn import MetaPath2Vec

path = osp.join(osp.dirname('/home/minhnn/minhnn/ICSE/pytorch_geometric/data/AMiner/processed'))
dataset = AMiner(path)
data = dataset[0]
# -

data.num_nodes_dict

pickle_path = '/home/minhnn/minhnn/ICSE/datasets/Etherscan_Contract/extracted_graph'
pickle_files = sorted(sorted([f for f in os.listdir(pickle_path) if f.endswith('.gpickle')]), key=len)
len(pickle_files)
extracted_graph = [f for f in os.listdir(pickle_path) if f.endswith('.gpickle')]
num_graphs = len(extracted_graph)
print('num graphs: {}'.format(num_graphs))

# ## Get Geometric graph

# + tags=[]
geo_graph_data = {}
dgl_graph_data = {}
for i in range(num_graphs):
    nx_graph = nx.read_gpickle(join(pickle_path, extracted_graph[i]))
    nx_graph, list_node_type = add_node_type_feature(nx_graph)
    nx_graph, list_edge_type = add_edge_type_feature(nx_graph)
    nx_graph = nx.convert_node_labels_to_integers(nx_graph)
    nx_g_data = generate_hetero_graph_data(nx_graph)
#     nx_g_data = add_full_metapath(nx_g_data, meta_path_types)
#     dgl_hete_graph = dgl.heterograph(nx_g_data)
    for k, v in nx_g_data.items():
        v_tensor = torch.stack(list(v), dim=0)
        if k in geo_graph_data.keys():
            geo_graph_data[k] = torch.cat((geo_graph_data[k], v_tensor), 1)
            dgl_graph_data[k] = (torch.cat((dgl_graph_data[k][0], v[0])), torch.cat((dgl_graph_data[k][1], v[1])))
        else:
            geo_graph_data[k] = v_tensor
            dgl_graph_data[k] = v
print(len(geo_graph_data.keys()))
num_nodes_dict = get_num_node_dict(geo_graph_data)
print(num_nodes_dict)

# + tags=[]
geo_graph_data = dict(sorted(geo_graph_data.items(), key=lambda item: max(item[1][0].max().item(), item[1][1].max().item()), reverse=True))
geo_graph_data[list(geo_graph_data.keys())[0]][0].max()
# -

single_graph_meta_path = list(nx_g_data.keys())
bi_single_graph_meta_path= single_graph_meta_path + [t[::-1] for t in single_graph_meta_path[::-1]]
single_meta_path_embedding =  MetaPath2Vec(nx_g_data, embedding_dim=128,
                     metapath=bi_single_graph_meta_path, walk_length=2, context_size=2,
                     walks_per_node=1, num_negative_samples=5, num_nodes_dict=None,
                     sparse=True).to(device)
single_meta_path_embedding
z = single_meta_path_embedding('EXPRESSION')
z, z.shape

# ## Get DGL graph data

# + tags=[]
dgl_graph_data = {}
for k, v in geo_graph_data.items():
    dgl_graph_data[k] = (v[0], v[1])
# -

geo_meta_path = list(geo_graph_data.keys())
bidirect_geo_meta_path_types = geo_meta_path + [t[::-1] for t in geo_meta_path[::-1]]
print(len(bidirect_geo_meta_path_types))

num_nodes_dict[bidirect_geo_meta_path_types[0][0]]

# + tags=[]
dgl_hete_graph = dgl.heterograph(dgl_graph_data)
num_nodes_dict = {}
for n in dgl_hete_graph.ntypes:
    num_nodes_dict[n] = dgl_hete_graph.number_of_nodes(n)
# num_nodes_dict = dict(sorted(num_nodes_dict.items(), key=lambda item: item[1], reverse=True))
num_nodes_dict
# -

from dgl.data.utils import save_graphs
save_graphs('./outputs/graph.bin', [dgl_hete_graph])

len(dgl_hete_graph.canonical_etypes)

# +
explicated_dgl_graph_data = {}
for k, v in dgl_graph_data.items():
    explicated_dgl_graph_data[(k[0], '_'.join(k), k[-1])] = v

explicated_dgl_hete_graph = dgl.heterograph(explicated_dgl_graph_data)

# +
bi_dgl_graph_data = {}
for k, v in dgl_graph_data.items():
    bi_dgl_graph_data[k] = v
    if k[::-1] in dgl_graph_data.keys():
#         print(k)
#         bi_dgl_graph_data[k[::-1]] = (torch.cat((v[0], v[1])), torch.cat((v[1], v[0])))
        continue
    else:
        bi_dgl_graph_data[k[::-1]] = v[::-1]

bi_dgl_hete_graph = dgl.heterograph(bi_dgl_graph_data)
print(len(bi_dgl_hete_graph.canonical_etypes))
# -

total = 0
for n in bi_dgl_hete_graph.ntypes:
    total += bi_dgl_hete_graph.number_of_nodes(n)
print(total)

bi_dgl_hete_graph.num_nodes()

save_graphs('./outputs/symmetric_graph.bin', [bi_dgl_hete_graph])

total = 0
for v in num_nodes_dict.values():
    total += v
print(total)

row, col = geo_graph_data[('ENTRY_POINT', 'next', 'EXPRESSION')]
print(row, col)

# + jupyter={"outputs_hidden": true} tags=[]
for keys, edge_index in geo_graph_data.items():
    sizes = (num_nodes_dict[keys[0]], num_nodes_dict[keys[-1]])
    row, col = edge_index
    print(keys)
#     adj = SparseTensor(row=row, col=col, sparse_sizes=sizes)
#     adj = adj.to('cpu')
#     adj_dict[keys] = adj
# -

model = MetaPath2Vec(geo_graph_data, embedding_dim=128,
                     metapath=bidirect_geo_meta_path_types, walk_length=2, context_size=2,
                     walks_per_node=1, num_negative_samples=5, num_nodes_dict=num_nodes_dict,
                     sparse=True).to(device)
model

loader = model.loader(batch_size=1, shuffle=False, num_workers=0, drop_last=True)
print(loader.__dict__)
print(list(model.parameters()))

# + tags=[]
loader = model.loader(batch_size=1, shuffle=False, num_workers=0, drop_last=True)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.001)

def train(epoch, log_steps=100, eval_steps=2000):
    model.train()
    total_loss = 0
    for i, (pos_rw, neg_rw) in enumerate(loader):
#         print(i)
#         print((pos_rw.shape, neg_rw.shape))
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % log_steps == 0:
            print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                   f'Loss: {total_loss / log_steps:.4f}'))
            total_loss = 0

        if (i + 1) % eval_steps == 0:
            acc = test()
            print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                   f'Acc: {acc:.4f}'))


# + jupyter={"outputs_hidden": true} tags=[]
for i, (pos_rw, neg_rw) in enumerate(loader):
    print(i, pos_rw.shape, neg_rw.shape)

# + jupyter={"outputs_hidden": true} tags=[]
train(1)
            
# for epoch in range(1, 6):
#     train(epoch)
#     acc = test()
#     print(f'Epoch: {epoch}, Accuracy: {acc:.4f}')


# + [markdown] tags=[]
# # Metapath2vec
# -

from torch_geometric.nn import MetaPath2Vec

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

geo_graph_data.keys()

# +
metapath_embedding = MetaPath2Vec(geo_graph_data, embedding_dim=128,
                     metapath=bidirect_geo_meta_path_types, walk_length=20, context_size=15,
                     walks_per_node=1, num_negative_samples=5, num_nodes_dict=None,
                     sparse=True).to(device)

# metapath_embedding.eval()
metapath_embedding.embedding.weight.shape
# -

features = None
for node in bi_dgl_hete_graph.ntypes:
    if features is None:
        features = metapath_embedding(n)
    else:
        features = torch.cat((features, metapath_embedding(n)))
print(features.shape)

# + jupyter={"outputs_hidden": true} tags=[]
for i in geo_graph_data.items():
    print(i)
# -

# # Load global graph

global_graph_path = '/home/minhnn/minhnn/ICSE/ge-sc/outputs/compress_graphs.gpickle'

# + tags=[]
nx_graph = nx.read_gpickle(global_graph_path)
nx_graph = nx.convert_node_labels_to_integers(nx_graph)
nx_g_data = generate_hetero_graph_data(nx_graph)
explicated_dgl_graph_data = {}
for k, v in nx_g_data.items():
    explicated_dgl_graph_data[(k[0], '_'.join(k), k[-1])] = v

global_graph = dgl.heterograph(explicated_dgl_graph_data)
    
# dgl_hete_graph = dgl.heterograph(nx_g_data)
# print(dgl_hete_graph.ntypes, dgl_hete_graph.num_nodes())
# print(dgl_hete_graph.etypes, dgl_hete_graph.num_edges())
# global_graph = dgl_hete_graph

# + tags=[] jupyter={"outputs_hidden": true}
global_graph.canonical_etypes

# + tags=[] jupyter={"outputs_hidden": true}
for k in nx_g_data.keys():
    print(k)
# -

feature_data = {}
for ntype in global_graph.ntypes:
    feature_data[ntype] = nodetype2onehot(ntype, ntypes_dict).repeat(dgl_hete_graph.num_nodes(ntype), 1)
global_graph.ndata['feat'] = feature_data

# + tags=[] jupyter={"outputs_hidden": true}
global_graph.ndata['feat']
# -



# # Load global graph

from dgl_graph_generator import generate_hetero_graph_data, add_hetero_ids, get_number_of_nodes

extracted_graph = '/home/minhnn/minhnn/ICSE/datasets/Etherscan_Contract/extracted_source_code'
filename_mapping = {file: idx for idx, file in enumerate(os.listdir(extracted_graph))}

# + tags=[]
nx_graph = nx.read_gpickle('/home/minhnn/minhnn/ICSE/datasets/Etherscan_Contract/compressed_graphs/compress_graphs.gpickle')
nx_graph = nx.convert_node_labels_to_integers(nx_graph)
nx_graph = add_hetero_ids(nx_graph)
nx_g_data, node_tracker = generate_hetero_graph_data(nx_graph, filename_mapping)
number_of_nodes = get_number_of_nodes(nx_graph)
global_graph = dgl.heterograph(nx_g_data, num_nodes_dict=number_of_nodes)
global_graph.ndata['filename'] = node_tracker
# -

global_graph.number_of_nodes(), global_graph.number_of_edges()


def reflect_graph(nx_g_data):
    reflected_data = {}
    for metapath, value in nx_g_data.items():
        if metapath[0] == metapath[-1]:
            reflected_data[metapath] = (torch.cat((value[0], value[1])), torch.cat((value[1], value[0])))
        else:
            if metapath not in reflected_data.keys():
                reflected_data[metapath] = value
            else:
                reflected_data[metapath] = (torch.cat((reflected_data[metapath][0], value[0])), torch.cat((reflected_data[metapath][1], value[1])))
            if metapath[::-1] not in reflected_data.keys():
                reflected_data[metapath[::-1]] = (value[1], value[0])
            else:
                reflected_data[metapath[::-1]] = (torch.cat((reflected_data[metapath[::-1]][0], value[1])), torch.cat((reflected_data[metapath[::-1]][1], value[0])))
    return reflected_data


# + tags=[]
reflected_global_graph_data = reflect_graph(nx_g_data)
reflected_global_graph = dgl.heterograph(reflected_global_graph_data)
reflected_global_graph.ndata['filename'] = global_graph.ndata['filename']
reflected_global_graph.number_of_nodes(), reflected_global_graph.number_of_edges()
# -

features = {}
for ntype in reflected_global_graph.ntypes:
    features[ntype] = nodetype2onehot(ntype, ntypes_dict).repeat(global_graph.num_nodes(ntype), 1)
reflected_global_graph.ndata['feat'] = features

# + [markdown] toc-hr-collapsed=true
# # HAN
# -

import sys
sys.path.append('../dgl')
from examples.pytorch.han.model_hetero import SemanticAttention, HANLayer
from examples.pytorch.han.utils import EarlyStopping

# + tags=[]
"""QM7b dataset for graph property prediction (regression)."""
import numpy as np
import os
import json


class HANDataset(DGLDataset):
    _url = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/' \
           'datasets/qm7b.mat'
    _sha1_str = '4102c744bb9d6fd7b40ac67a300e49cd87e28392'
    _label = '/home/minhnn/minhnn/ICSE/datasets/Etherscan_Contract/Reentrancy_AutoExtract_corenodes.json'
    _data_path = '/home/minhnn/minhnn/ICSE/datasets/Etherscan_Contract/extracted_graph'

    def __init__(self, raw_dir=None, force_reload=False, verbose=False):
        super(HANDataset, self).__init__(name='ethsc',
                                          url=self._url,
                                          raw_dir=raw_dir,
                                          force_reload=force_reload,
                                          verbose=verbose)

    def process(self):
        self.graphs, self.label = self._load_graph()

    def _load_graph(self):
        extracted_graph = [f for f in os.listdir(self._data_path) if f.endswith('.gpickle')]
        num_graphs = len(extracted_graph)
        graphs = []
        labels = []
        for i in range(num_graphs):
            nx_graph = nx.read_gpickle(join(self._data_path, extracted_graph[i]))
            nx_graph = nx.convert_node_labels_to_integers(nx_graph)
            nx_g_data = generate_hetero_graph_data(nx_graph)
            dgl_hete_graph = dgl.heterograph(nx_g_data)
            feature_data = {}
            h_data = {}
            for ntype in dgl_hete_graph.ntypes:
                feature_data[ntype] = nodetype2onehot(ntype, ntypes_dict).repeat(dgl_hete_graph.num_nodes(ntype), 1)
#                 h_data[ntype] = torch.tensor([], dtype=torch.int64).repeat(dgl_hete_graph.num_nodes(ntype), 1)
                
            dgl_hete_graph.ndata['feat'] = feature_data
#             dgl_hete_graph.ndata['h'] = h_data
            graphs.append(dgl_hete_graph)
            labels.append(int(label_dict[extracted_graph[i].replace('.gpickle', '.sol')]))
        labels = torch.tensor(labels, dtype=torch.int64)
#         print(graphs[0].ndata)
        return graphs, labels


    @property
    def num_labels(self):
        return 2

    def __getitem__(self, idx):
        return self.graphs[idx], self.label[idx]

    def __len__(self):
        return len(self.graphs)

Ethdataset = HANDataset()


# +
class ETHidsDataset(DGLDataset):
    _label = '/home/minhnn/minhnn/ICSE/datasets/Etherscan_Contract/Reentrancy_AutoExtract_corenodes.json'
    _data_path = '/home/minhnn/minhnn/ICSE/datasets/Etherscan_Contract/extracted_source_code'

    def __init__(self, raw_dir=None, force_reload=False, verbose=False):
        super(ETHidsDataset, self).__init__(name='ethscids',
                                          raw_dir=raw_dir,
                                          force_reload=force_reload,
                                          verbose=verbose)

    def process(self):
        self.graphs, self.label = self._load_graph()

    def _load_graph(self):
        extracted_graph = [f for f in os.listdir(self._data_path) if f.endswith('.sol')]
        num_graphs = len(extracted_graph)
        graphs = []
        labels = []
        with open(self._label, 'r') as f:
            content = f.readlines()
        label_dict = {}
        for l in content:
            sc = json.loads(l.strip('\n').strip(','))
            label_dict[sc['contract_name']] = sc['targets']
        label_dict['No_Reentrance.sol'] = '0'
        for i in range(num_graphs):
            graphs.append(extracted_graph[i])
            labels.append(int(label_dict[extracted_graph[i].replace('.gpickle', '.sol')]))
        labels = torch.tensor(labels, dtype=torch.int64)
#         onehot_label = None
#         for label in labels:
#             one_hot = torch.zeros(2)
#             one_hot[label] = 1
#             if onehot_label is None:
#                 one_hot_label = one_hot
#             else:
#                 onehot_label = torch.cat((onehot_label, one_hot), dim=0)
#         labels = onehot_label
#         print(graphs[0].ndata)
        return graphs, labels


    @property
    def num_labels(self):
        return 2

    def __getitem__(self, idx):
        return self.graphs[idx], self.label[idx]

    def __len__(self):
        return len(self.graphs)

EthIdsdataset = ETHidsDataset()
# -

dataloader = GraphDataLoader(
    EthIdsdataset,
    batch_size=8,
    drop_last=False,
    shuffle=True)

# + tags=[]
for graph, label in dataloader:
    print(*graph)
#     print(graph, label)

# + tags=[]
loss_fcn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005,
                             weight_decay=0.001)

# + tags=[] jupyter={"outputs_hidden": true}
meta_paths = []
for mt in reflected_global_graph.canonical_etypes:
    if mt[0] == mt[1]:
        ref_mt = [mt]
    else:
        ref_mt = [mt, mt[::-1]]
    if ref_mt not in meta_paths:
        meta_paths.append(ref_mt)
print(len(meta_paths))
meta_paths
# -

print(len(reflected_global_graph.canonical_etypes))

# meta_paths = [[('BEGIN_LOOP', 'BEGIN_LOOP_next_EXPRESSION', 'EXPRESSION'), ('EXPRESSION', 'EXPRESSION_next_BEGIN_LOOP', 'BEGIN_LOOP')],
#               [('EXPRESSION', 'EXPRESSION_next_IF', 'IF'), ('IF', 'IF_if_false_EXPRESSION', 'EXPRESSION')]]
meta_paths = [[('BEGIN_LOOP', 'next', 'EXPRESSION'), ('EXPRESSION', 'next', 'BEGIN_LOOP')],
              [('EXPRESSION', 'next', 'EXPRESSION'), ('IF', 'if_false', 'EXPRESSION')]]
# meta_paths = [['BEGIN_LOOP_next_EXPRESSION', 'EXPRESSION_next_BEGIN_LOOP'],
#               ['EXPRESSION_next_IF', 'IF_if_false_EXPRESSION']]

features = {}
for ntype in reflected_global_graph.ntypes:
    features[ntype] = nodetype2onehot(ntype, ntypes_dict).repeat(reflected_global_graph.num_nodes(ntype), 1)
reflected_global_graph.ndata['feat'] = features

# + tags=[] jupyter={"outputs_hidden": true}
for k, v in reflected_global_graph.ndata['filename'].items():
    print(k)
    file_mapping = v == 0


# -

class HANVulClassifier(nn.Module):
    def __init__(self, reflected_global_graph, meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super(HANVulClassifier, self).__init__()
        self.reflected_global_graph = reflected_global_graph
        self.meta_paths = meta_paths
        self.node_types = set([meta_path[0][0] for meta_path in meta_paths])
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer([meta_paths[0]], in_size, hidden_size, num_heads, dropout))
        for meta_path in meta_paths[1:]:
            self.layers.append(HANLayer([meta_path], in_size, hidden_size, num_heads, dropout))
        self.features = {}
        for han in self.layers:
            ntype = han.meta_paths[0][0][0]
            self.features[ntype] = han(self.reflected_global_graph, self.reflected_global_graph.ndata['feat'][ntype])
        self.classify = nn.Linear(hidden_size * num_heads , out_size)
        

    def forward(self, batched_g_name):
        batched_graph_embedded = []
        for g_name in batched_g_name:
            file_ids = filename_mapping[g_name]
            graph_embedded = 0
            for node_type in self.node_types:
                file_mask = self.reflected_global_graph.ndata['filename'][node_type] == file_ids
                if file_mask.sum().item() != 0:
                    graph_embedded += self.features[node_type][file_mask].mean(0)
            batched_graph_embedded.append(graph_embedded.tolist())
        batched_graph_embedded = torch.tensor(batched_graph_embedded).to(device)
        output = self.classify(batched_graph_embedded)
        return output

extracted_graph_path = '/home/minhnn/minhnn/ICSE/datasets/Etherscan_Contract/extracted_source_code'
extracted_graph = [f for f in os.listdir(extracted_graph_path) if f.endswith('.sol')]
num_graphs = len(extracted_graph)

type(one_hot(torch.tensor([1]), 2))
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
print(target, target.shape, target.dtype)
# output = loss(input, target)output.backward()
a = torch.tensor([float('nan'), float(1)])
torch.isnan(a).any()
a = torch.tensor([list(torch.tensor([1,2])), list(torch.tensor([3,4]))])

device = torch.device('cuda:0')
device

# + tags=[]
model = HANVulClassifier(reflected_global_graph, meta_paths, in_size=16, hidden_size=16, out_size=2, num_heads=8, dropout=0.6)
# opt = torch.optim.Adam(model.parameters(),  lr=0.0005)
model.to(device)
loss = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.001)
model.train()
for epoch in range(10):
    total_loss = 0
    train_acc = 0
    steps = 0
    logists = []
    target = []
    for idx, (batched_graph_name, labels) in enumerate(dataloader):
#         print(labels)
        torch.set_grad_enabled(True)
        logist = model(batched_graph_name)
        preds = logits.argmax(dim=1)
#         label = int(label_dict[graph_name])
#         logists.append(logist.tolist())
#         target.append(label)
        loss = cross_entropy(logist.to(device), labels.to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
        train_acc += accuracy(preds, labels)
        
#     preds = torch.tensor(logists, requires_grad=True)
#     target = torch.tensor(target, dtype=torch.int64)
#     loss = cross_entropy(preds, target)
#     opt.zero_grad()
#     loss.backward()
#     opt.step()
    print('train_loss: {:4f} - train_acc: {:4f}'.format(total_loss/(idx+1), train_acc/(idx+1)))
# -

num_graphs

# + tags=[] jupyter={"outputs_hidden": true}
epochs = 100
k_folds = 2
kfold = KFold(n_splits=k_folds, shuffle=True)
train_results = {}
test_results = {}
for fold, (train_ids, test_ids) in enumerate(kfold.split(range(num_graphs))):
    train_results[fold] = {'loss': [], 'acc': []}
    test_results[fold] = {'loss': [], 'acc': []}
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    train_dataloader = GraphDataLoader(
    EthIdsdataset,
    batch_size=128,
    drop_last=False,
    sampler=train_subsampler)
    test_dataloader = GraphDataLoader(
    EthIdsdataset,
    batch_size=128,
    drop_last=False,
    sampler=test_subsampler)
    print('Start training fold {} with {}/{} train/test smart contracts'.format(fold, len(train_dataloader), len(test_dataloader)))
    total_steps = len(train_dataloader) * epochs
    model = HANVulClassifier(reflected_global_graph, meta_paths, in_size=16, hidden_size=16, out_size=2, num_heads=8, dropout=0.6)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(),  lr=0.0005)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.01, total_steps=total_steps)
    lrs = []
    for epoch in range(epochs):
        print('Fold {} - Epochs {}'.format(fold, epoch))
        total_loss = 0
        train_acc = 0
        steps = 0
        for idx, (batched_graph, labels) in enumerate(train_dataloader):
            labels = labels.to(device)
            logits = model(batched_graph)
            preds = logits.argmax(dim=1)
            train_acc += accuracy(preds, labels)
            loss = cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
            total_loss += loss.item()
            steps += 1
            lrs.append(opt.param_groups[0]["lr"])
        print('train_loss: {:4f} - train_acc: {:4f}'.format(total_loss/steps, train_acc/steps))
        train_results[fold]['loss'].append(total_loss/steps)
        train_results[fold]['acc'].append(train_acc/steps)

        with torch.no_grad():
            total_loss = 0
            test_acc = 0
            steps = 0
            for idx, (batched_graph, labels) in enumerate(test_dataloader):
                labels = labels.to(device)
                logits = model(batched_graph)
                preds = logits.argmax(dim=1)
                test_acc += accuracy(preds, labels)
                loss = cross_entropy(logits, labels)
                total_loss += loss.item()
                steps += 1
            print('valid_loss: {:4f} - valid_acc: {:4f}'.format(total_loss/steps, test_acc/steps))
            test_results[fold]['loss'].append(total_loss/steps)
            test_results[fold]['acc'].append(test_acc/steps)
    print('Saving model fold {}'.format(fold))
    save_path = f'./models/model_han_fold_{fold}.pth'
    torch.save(model.state_dict(), save_path)

# +
tensorboard_path = '/home/minhnn/minhnn/ICSE/ge-sc/logs/HAN_CrossVal'
writer = SummaryWriter(tensorboard_path)
tensorboard_acc = {'train': train_results[0]['acc'], 'valid': test_results[0]['acc']}
tensorboard_loss = {'train': train_results[0]['loss'], 'valid': test_results[0]['loss']}
# for key, results in train_results[0].items():
#     tensorboard_acc[] = 
#     writer.add_scalars('Loss', train_res, epoch)
# for idx, lr in enumerate(lrs):
#     writer.add_scalar('Learning rate', lr, idx)
for idx, lr in enumerate(lrs):
    writer.add_scalar('Learning rate', lr, idx)

for fold in range(k_folds):
    for idx in range(epochs):
        writer.add_scalars('Accuracy', {f'train_{fold+1}': train_results[fold]['acc'][idx],
                                        f'valid_{fold+1}': test_results[fold]['acc'][idx]}, idx)
        writer.add_scalars('Loss', {f'train_{fold+1}': train_results[fold]['loss'][idx],
                                    f'valid_{fold+1}': test_results[fold]['loss'][idx]}, idx)
writer.close()
# -

epochs = 5
# etypes is the list of edge types as strings.
opt = torch.optim.Adam(model.parameters())
for epoch in range(epochs):
    for batched_graph, labels in dataloader:
        logits = model(batched_graph)
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad()
        opt.step()







edge_metapath = list(dgl_hete_graph.metagraph().edges())

bi_dgl_hete_graph.get_etype_id(('RETURN', 'next', 'END_IF'))

# + tags=[]
num_node = 0
for node in explicated_dgl_hete_graph.ntypes:
    print('node {} has {}'.format(node, explicated_dgl_hete_graph.number_of_nodes(node)))
    num_node += explicated_dgl_hete_graph.number_of_nodes(node)
print(len(explicated_dgl_hete_graph.ntypes))
print(num_node)
# -

features = []
for node in explicated_dgl_hete_graph.ntypes:
    if features is None:
        features = metapath_embedding(n)
    else:
        features.append(metapath_embedding(node))
print(len(features))

# + tags=[]
dgl.metapath_reachable_graph(dgl_hete_graph, dgl_hete_graph.canonical_etypes)
# -

dgl_hete_graph.adj(etype=('BEGIN_LOOP', 'next', 'EXPRESSION'))

# + jupyter={"outputs_hidden": true} tags=[]
adj = 1
for etype in dgl_hete_graph.canonical_etypes:
    adj_tmp = dgl_hete_graph.adj(etype=etype, scipy_fmt='csr', transpose=False)
    
    
    adj = adj * dgl_hete_graph.adj(etype=etype, scipy_fmt='csr', transpose=False)

# + tags=[] jupyter={"outputs_hidden": true}
explicated_dgl_hete_graph.canonical_etypes

# + tags=[]
edge_metapah = [[emt for emt in explicated_dgl_hete_graph.etypes]]
len(edge_metapah[0])
# -

single_edge_metapah = [['BEGIN_LOOP_next_EXPRESSION', 'EXPRESSION_next_BEGIN_LOOP'],
                       ['IF_LOOP_if_true_BEGIN_LOOP', 'BEGIN_LOOP_next_IF_LOOP'],
#                       ['EXPRESSION_next_NEW VARIABLE', 'NEW VARIABLE_next_EXPRESSION'],
                      ]

explicated_dgl_hete_graph = dgl.remove_self_loop(explicated_dgl_hete_graph, etype=('END_IF', 'END_IF_next_END_IF', 'END_IF'))

print(explicated_dgl_hete_graph.number_of_nodes('BEGIN_LOOP'), explicated_dgl_hete_graph.number_of_nodes('EXPRESSION'))

model = HAN(meta_paths=single_edge_metapah,
            in_size=128,
            hidden_size=8,
            out_size=2,
            num_heads=[8],
            dropout=0.6).to(device)

explicated_dgl_hete_graph.number_of_nodes('BEGIN_LOOP')

feature_test = []
current = 0
for idx, node in enumerate(explicated_dgl_hete_graph.ntypes):
    feature_test.append(metapath_embedding.embedding.weight[current:current+explicated_dgl_hete_graph.number_of_nodes(node)])
print(len(feature_test))

# + tags=[]
logit = model(explicated_dgl_hete_graph.to(device), metapath_embedding.embedding.weight.data[:1559])
# -

logit.shape

bi_dgl_hete_graph.number_of_edges(('_', 'next', 'IF'))

features = torch.tensor([]).to(device)
for node in num_nodes_dict:
    features = torch.cat((features, metapath_embedding(node)))
print(features.shape)


def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1


# + tags=[]
for epoch in range(100):
    model.train()
    logits = model(bi_dgl_hete_graph, bi_features)
    loss = loss_fcn(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_acc, train_micro_f1, train_macro_f1 = score(logits, labels)
# -



# # Visualization

from torch.utils.tensorboard import SummaryWriter

logs_path = '/home/minhnn/minhnn/ICSE/ge-sc/logs/2convs.log'
tensorboard_path = '/home/minhnn/minhnn/ICSE/ge-sc/logs/ConvHete'

writer = SummaryWriter(tensorboard_path)
with open(logs_path, 'r') as f:
    content = f.readlines()
for idx, l in enumerate(content):
    loss = float(l.split(' - ')[0].split()[-1])
    acc = float(l.split(' - ')[1].split()[-1])
    writer.add_scalar('Loss/train', loss, idx)
    writer.add_scalar('Accuracy/train', acc, idx)
writer.close()



label_0 = '/home/minhnn/minhnn/ICSE/ge-sc/dgl_models/pytorch/han/dataset/aggregate/labels.json'
smartbugs_path = '/home/minhnn/minhnn/ICSE/ge-sc/dgl_models/pytorch/han/dataset/ijcai2020/source_code'
output_path = '/home/minhnn/minhnn/ICSE/ge-sc/dgl_models/pytorch/han/dataset/ijcai2020/non_vul_source_code'
smartbugs = [f for f in os.listdir(smartbugs_path)]

from shutil import copy
with open(label_0, 'r') as f:
    content = f.readlines()
print(len(content))
non_vul_sc = []
for sc in content:
    line = sc.strip('\n').strip(',')
    line = json.loads(line)
    if line['targets'] == '0':
        non_vul_sc.append(line['contract_name'])
        try:
            copy(join(smartbugs_path, line['contract_name']), join(output_path, line['contract_name']))
        except:
            print(line['contract_name'])
print(len(non_vul_sc))
# for sc in smartbugs:
#     item = {"target": "1", "contract_name": sc}
#     content.append(json.dumps(item) + ',\n')
# with open(output, 'w') as f:
#     f.writelines(content)

a = torch.rand((3,4)).tolist()

# +
from torch_geometric.datasets import AMiner
import os.path as osp

path = '/home/minhnn/minhnn/ICSE/pytorch_geometric/data/AMiner'
dataset = AMiner(path)
data = dataset[0]
# -

data

# +
metapath = [
        ('author', 'writes', 'paper'),
        ('paper', 'published in', 'venue'),
        ('venue', 'published', 'paper'),
        ('paper', 'written by', 'author'),
    ]
model = MetaPath2Vec(data.edge_index_dict, embedding_dim=128,
                         metapath=metapath, walk_length=50, context_size=7,
                         walks_per_node=5, num_negative_samples=5,
                         sparse=True).to(device)

loader = model.loader(batch_size=256, shuffle=True, num_workers=12)
# -

data.edge_index_dict

# + tags=[]
for i, (pos_rw, neg_rw) in enumerate(loader[10]):
        print(pos_rw)
        print(neg_rw)
# -

data.y_index_dict['venue']

[1741, 2245,  111,  837, 2588, 2116, 2696, 3648, 3784,  313, 3414,  598,
        2995, 2716, 1423,  783, 1902, 3132, 1753, 2748, 2660, 3182,  775, 3339,
        1601, 3589,  156, 1145,  692, 3048,  925, 1587,  820, 1374, 3719,  819,
         492, 3830, 2777, 3001, 3693,  517, 1808, 2353, 3499, 1763, 2372, 1030,
         721, 2680, 3355, 1217, 3400, 1271, 1970, 1127,  407,  353, 1471, 1095,
         477, 3701,   65, 1009, 1899, 1442, 2073, 3143, 2466,  289, 1996, 1070,
        3871, 3695,  281, 3633,   50, 2642, 1925, 1285, 2587, 3814, 3582, 1873,
        1339, 3450,  271, 2966,  453, 2638, 1354, 3211,  391, 1588, 3875, 2216,
        2146, 3765, 2486,  661, 3367,  426,  750, 2158,  519,  230, 1677,  839,
        2945, 1313, 1037, 2879, 2225, 3523, 1247,  448,  227, 3385,  529, 2849,
        1584, 1229,  373, 2235, 1819, 1764, 3155, 2852, 2789, 3474, 1571, 2088,
         208,  462])types = set([x[0] for x in data.edge_index_dict.keys()]) | set([x[-1] for x in data.edge_index_dict.keys()])
print(types)
types = sorted(list(types))

data[0]['author']

buggy_file = './data/smartbugs_wild/'
import re
pattern = re.compile(r'\d.\d.\d+')
with open(buggy_file, 'r') as f:
    line = f.readline()
    print(line)
    while line:
        if 'pragma solidity' in line:
            result = pattern.findall(line)
            print(result)
            parts = line.split()[2].split('.')
            version = '.'.join([parts[0][-1], parts[1], parts[-1]])
            print(version)
        line = f.readline()

import solc
from solc import install_solc

# +
import sys
import re
import subprocess

pattern =  re.compile(r'\d.\d.\d+')
def get_solc_version(source):
    with open(source, 'r') as f:
        line = f.readline()
        while line:
            if 'pragma solidity' in line:
                if len(pattern.findall(line)) > 0:
                    return pattern.findall(line)[0]
                else:
                    return '0.4.25'
            line = f.readline()
    return '0.4.25'

smart_contract_path = '/home/minhnn/minhnn/ICSE/ge-sc/data/solidifi_buggy_contracts/Re-entrancy'
smart_contracts = [join(smart_contract_path, f) for f in os.listdir(smart_contract_path) if f.endswith('.sol')]
count = 0
for sc in smart_contracts:
    sc_version = get_solc_version(sc)
    try:
        subprocess.run(['solc-select', 'install', sc_version])
        count += 1
    except:
        print(sc_version)
print(f'Extract {count}/{len(smart_contracts)} sources')
# -

path = '/home/minhnn/minhnn/ICSE/ge-sc/data/solidifi_buggy_contracts/Overflow-Underflow/vulnerabilities.json'
out = '/home/minhnn/minhnn/ICSE/ge-sc/data/solidifi_buggy_contracts/aggregate/vulnerabilities.json'
buggy_path = '/home/minhnn/minhnn/ICSE/ge-sc/data/solidifi_buggy_contracts'
buggys = [join(buggy_path, f) for f in os.listdir(buggy_path)]
print()
buggys.remove(join(buggy_path, 'aggregate'))
total = []
for bug in buggys:
    bugtype = bug.split('/')[-1]
    with open(path, 'r') as f:
        content = json.load(f)
    for i in range(len(content)):
        content[i]['name'] = bugtype + '_' + content[i]['name'] 
    total += content
with open(out, 'w') as fout:
    json.dump(total, fout)

import json
with open(out, 'w') as fout:
    json.dump(content, fout)






