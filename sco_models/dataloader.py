import os

import json
import torch
import dgl
import numpy as np
import networkx as nx
from dgl.data import DGLDataset

from .opcodes import int2op


class EthIdsDataset(DGLDataset):
    def __init__(self, label, raw_dir=None, force_reload=True, verbose=False):
        # self._data_path = data_path
        self._label = label
        super(EthIdsDataset, self).__init__(name='ethscids',
                                            raw_dir=raw_dir,
                                            force_reload=force_reload,
                                            verbose=verbose)

    def process(self):
        # Get labels
        with open(self._label, 'r') as f:
            self._annotations = json.load(f)
        # self.label_dict = {}
        # for sc in annotations:
        #     self.label_dict[sc['contract_name']] = sc['targets']
         # Get source names
        # self.extracted_graph = [f for f in os.listdir(self._data_path) if f.endswith('.sol')]
        self.num_graphs = len(self._annotations)
        self.graphs, self.label = self._load_graph()
        # Get filename ids
        # self.filename_mapping = {file: idx for idx, file in enumerate(self.extracted_graph)}

    def _load_graph(self):
        graphs = []
        labels = []
        for contract in self._annotations:
            graphs.append(contract['contract_name'])
            labels.append(int(contract['targets']))
        labels = torch.tensor(labels, dtype=torch.int64)
        return graphs, labels

    @property
    def num_labels(self):
        return 2

    def __getitem__(self, idx):
        return self.graphs[idx], self.label[idx]

    def __len__(self):
        return len(self.graphs)


class EthNodeDataset(DGLDataset):
    def __init__(self, data_path, compressed_graph, raw_dir=None, force_reload=True, verbose=False):
        self._data_path = data_path
        self._compressed_graph = compressed_graph
        super(EthNodeDataset, self).__init__(name='ethscids',
                                            raw_dir=raw_dir,
                                            force_reload=force_reload,
                                            verbose=verbose)

    def process(self):
         # Get source names
        self.extracted_graph = [f for f in os.listdir(self._data_path) if f.endswith('.sol')]
        self.num_graphs = len(self.extracted_graph)
        self.graphs, self.label = self._load_graph()
        # Get filename ids
        self.filename_mapping = {file: idx for idx, file in enumerate(self.extracted_graph)}

    def _load_graph(self):
        graphs = []
        for i in range(self.num_graphs):
            graphs.append(self.extracted_graph[i])
        return graphs

    @property
    def num_labels(self):
        return 2

    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self):
        return len(self.graphs)


class dataGenerator:
    def __init__(self, file):
        s = '0123456789abcdef'
        self.file = file
        self.str2int = {s[i]: i for i in range(len(s))}
        #self.str2bit = {s[i]: [int(x) for x in (4 - len(bin(i)[2:])) * '0' + bin(i)[2:]] for i in range(len(s))}
        self.data = []
    # op is a operation code like 'PUSH1' , "RETURN" returns a 256 dim onehot vec for this operation
    def op2onehot(self, op):
        onehot_line = [0 for _ in range(256)]
        if op != 'EXIT BLOCK':
            op2int = {int2op[key]:key for key in int2op}
            hexstr = op2int[op]
            number = self.str2int[hexstr[0]] * 16 + self.str2int[hexstr[1]]
            onehot_line[number] = 1
        return onehot_line

    def encoder(self, nx_g):
        nodes = nx_g.nodes()
        graph_seq_feature = []
        vec_u, vec_v = [], []
        mapping = {n: i for i, n in enumerate(nodes)}
        for idx, node_idx in enumerate(nodes):
            node_seq_feature = []
            obj = nx_g._node[node_idx]
            seq = obj.get('label', '').split(':')[1:]
            for unit in seq:
                op = unit[1:unit.find('\\')]
                if 'PUSH' in op:
                    op = op.split(' ')[0]
                node_seq_feature.append(self.op2onehot(op))
            graph_seq_feature.append(torch.tensor([node_seq_feature]).float())
        for u, v in nx_g.edges():
            vec_u.append(mapping[u])
            vec_v.append(mapping[v])
        dg = dgl.graph((vec_u,vec_v))
        dg = dgl.add_self_loop(dg)
        dg.ndata['f'] = torch.zeros(dg.num_nodes(), 64)
        return dg, graph_seq_feature


    def get_data(self):
        counter = 0
        for obj in self.file:
            counter += 1
            nx_g = nx.drawing.nx_pydot.read_dot(obj['path'])
            dg, graph_seq_feature = self.encoder(nx_g)
            l = torch.LongTensor([obj['target']])
            self.data.append(([dg, graph_seq_feature], l))
            # if counter % 10 == 0:
            #     print(l, counter)
        return np.array(self.data)