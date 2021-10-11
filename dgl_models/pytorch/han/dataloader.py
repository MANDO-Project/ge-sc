import os

import json
import torch

from dgl.data import DGLDataset


class EthIdsDataset(DGLDataset):
    def __init__(self, data_path, compressed_graph, label, raw_dir=None, force_reload=True, verbose=False):
        self._data_path = data_path
        self._compressed_graph = compressed_graph
        self._label = label
        super(EthIdsDataset, self).__init__(name='ethscids',
                                            raw_dir=raw_dir,
                                            force_reload=force_reload,
                                            verbose=verbose)

    def process(self):
        # Get labels
        with open(self._label, 'r') as f:
            content = f.readlines()
        self.label_dict = {}
        for l in content:
            sc = json.loads(l.strip('\n').strip(','))
            self.label_dict[sc['contract_name']] = sc['targets']
         # Get source names
        self.extracted_graph = [f for f in os.listdir(self._data_path) if f.endswith('.sol')]
        self.num_graphs = len(self.extracted_graph)
        self.graphs, self.label = self._load_graph()
        # Get filename ids
        self.filename_mapping = {file: idx for idx, file in enumerate(self.extracted_graph)}

    def _load_graph(self):
        graphs = []
        labels = []
        for i in range(self.num_graphs):
            graphs.append(self.extracted_graph[i])
            labels.append(int(self.label_dict[self.extracted_graph[i]]))
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
