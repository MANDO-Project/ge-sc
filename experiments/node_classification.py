from calendar import EPOCH
import os
import sys
from os.path import join

import pickle
import json
import numpy as np
import networkx as nx
import torch
from sklearn.model_selection import KFold, train_test_split
from statistics import mean

from sco_models.model_hetero import HANVulClassifier
from sco_models.utils import get_classification_report
from sco_models.graph_utils import reveert_map_node_embedding, load_hetero_nx_graph


ROOT = '.'
DATA_ID = 0
REPEAT = 20
EPOCHS = 100
LR = 0.001
TASK = "node_classification"
STRUCTURE = 'han'
COMPRESSED_GRAPH = 'cfg_cg'
DATASET = 'clean_50_buggy_curated'
TRAIN_RATE = 0.7
VAL_RATE = 0.3
ratio = 1

models = ['nodetype', 'metapath2vec', 'gae', 'line', 'node2vec']
base_models = ['base_metapath2vec', 'base_gae', 'base_line', 'base_node2vec']
bug_list = ['access_control', 'arithmetic', 'denial_of_service',
            'front_running', 'reentrancy', 'time_manipulation', 
            'unchecked_low_level_calls']
file_counter = {'access_control': 57, 'arithmetic': 60, 'denial_of_service': 46,
              'front_running': 44, 'reentrancy': 71, 'time_manipulation': 50, 
              'unchecked_low_level_calls': 95}


def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size, dtype=torch.bool)
    mask[indices] = 1
    return mask.byte()


def get_node_label(nx_graph):
    nx_g = nx_graph
    node_labels = []
    label_ids = {'valid': 0}
    labeled_node_ids = {'buggy': [], 'valid': []}
    for node_id, node_data in nx_g.nodes(data=True):
        node_type = node_data['node_type']
        node_label = node_data['node_info_vulnerabilities']
        target = 0
        if node_label is None:
            target = 0
            labeled_node_ids['valid'].append(node_id)
        else:
            bug_type = node_label[0]['category']
            if bug_type not in label_ids:
                label_ids[bug_type] = len(label_ids)
            target = 1
            labeled_node_ids['buggy'].append(node_id)
        node_labels.append(target)
    return node_labels, labeled_node_ids, label_ids


def base_metapath2vec(compressed_graph, source_path, bugtype, device):
    logs = f'{ROOT}/logs/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/base_metapath2vec/{bugtype}/buggy_curated/'
    if not os.path.exists(logs):
        os.makedirs(logs)
    model = HANVulClassifier(compressed_graph, source_path, feature_extractor=None, node_feature='metapath2vec', device=device)
    features = model.symmetrical_global_graph.ndata['feat']
    nx_graph = load_hetero_nx_graph(compressed_graph)
    embedding = reveert_map_node_embedding(nx_graph, features)
    embedding = torch.tensor(embedding, device=device)
    assert len(nx_graph.nodes) == embedding.shape[0]
    number_of_nodes = embedding.shape[0]
    train, test = train_test_split(range(number_of_nodes), test_size=VAL_RATE)
    train_mask = get_binary_mask(number_of_nodes, train)
    test_mask = get_binary_mask(number_of_nodes, test)
    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        test_mask = test_mask.bool()
    classifier = torch.nn.Linear(128, 2)
    classifier.to(device)
    classifier.train()
    targets, _, _ = get_node_label(nx_graph)
    targets = torch.tensor(targets, device=device)
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        logits = classifier(embedding)
        logits = logits.to(device)
        train_loss = loss_fcn(logits[train_mask], targets[train_mask])
        train_loss.backward()
        optimizer.step()
    test_results = get_classification_report(targets[test_mask], logits[test_mask], output_dict=True)
    if os.path.isfile(join(logs, 'test_report.json')):
        with open(join(logs, 'test_report.json'), 'r') as f:
            report = json.load(f)
        report.append(test_results)
    else:
        report = [test_results]
    with open(join(logs, 'test_report.json'), 'w') as f:
        json.dump(report, f, indent=2)


def base_gae(nx_graph, embedded, bugtype, device):
    logs = f'{ROOT}/logs/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/base_gae/{bugtype}/buggy_curated/'
    if not os.path.exists(logs):
        os.makedirs(logs)
    with open(embedded, 'rb') as f:
        embedding = pickle.load(f, encoding="utf8")
        embedding = torch.tensor(embedding, device=device)
    assert len(nx_graph.nodes) == embedding.shape[0]
    number_of_nodes = embedding.shape[0]
    train, test = train_test_split(range(number_of_nodes), test_size=VAL_RATE)
    train_mask = get_binary_mask(number_of_nodes, train)
    test_mask = get_binary_mask(number_of_nodes, test)
    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        test_mask = test_mask.bool()
    classifier = torch.nn.Linear(128, 2)
    classifier.to(device)
    classifier.train()
    targets, _, _ = get_node_label(nx_graph)
    targets = torch.tensor(targets, device=device)
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        logits = classifier(embedding)
        logits = logits.to(device)
        train_loss = loss_fcn(logits[train_mask], targets[train_mask])
        train_loss.backward()
        optimizer.step()
    test_results = get_classification_report(targets[test_mask], logits[test_mask], output_dict=True)
    if os.path.isfile(join(logs, 'test_report.json')):
        with open(join(logs, 'test_report.json'), 'r') as f:
            report = json.load(f)
        report.append(test_results)
    else:
        report = [test_results]
    with open(join(logs, 'test_report.json'), 'w') as f:
        json.dump(report, f, indent=2)


def base_line(nx_graph, embedded, bugtype, device):
    logs = f'{ROOT}/logs/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/base_line/{bugtype}/buggy_curated/'
    if not os.path.exists(logs):
        os.makedirs(logs)
    with open(embedded, 'rb') as f:
        embedding = pickle.load(f, encoding="utf8")
        embedding = torch.tensor(embedding, device=device)
    assert len(nx_graph.nodes) == embedding.shape[0]
    number_of_nodes = embedding.shape[0]
    train, test = train_test_split(range(number_of_nodes), test_size=VAL_RATE)
    train_mask = get_binary_mask(number_of_nodes, train)
    test_mask = get_binary_mask(number_of_nodes, test)
    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        test_mask = test_mask.bool()
    classifier = torch.nn.Linear(128, 2)
    classifier.to(device)
    classifier.train()
    targets, _, _ = get_node_label(nx_graph)
    targets = torch.tensor(targets, device=device)
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        logits = classifier(embedding)
        logits = logits.to(device)
        train_loss = loss_fcn(logits[train_mask], targets[train_mask])
        train_loss.backward()
        optimizer.step()
    test_results = get_classification_report(targets[test_mask], logits[test_mask], output_dict=True)
    if os.path.isfile(join(logs, 'test_report.json')):
        with open(join(logs, 'test_report.json'), 'r') as f:
            report = json.load(f)
        report.append(test_results)
    else:
        report = [test_results]
    with open(join(logs, 'test_report.json'), 'w') as f:
        json.dump(report, f, indent=2)


def base_node2vec(nx_graph, embedded, bugtype, device):
    logs = f'{ROOT}/logs/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/base_node2vec/{bugtype}/buggy_curated/'
    if not os.path.exists(logs):
        os.makedirs(logs)
    with open(embedded, 'rb') as f:
        embedding = pickle.load(f, encoding="utf8")
        embedding = torch.tensor(embedding, device=device)
    assert len(nx_graph.nodes) == embedding.shape[0]
    number_of_nodes = embedding.shape[0]
    train, test = train_test_split(range(number_of_nodes), test_size=VAL_RATE)
    train_mask = get_binary_mask(number_of_nodes, train)
    test_mask = get_binary_mask(number_of_nodes, test)
    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        test_mask = test_mask.bool()
    classifier = torch.nn.Linear(128, 2)
    classifier.to(device)
    classifier.train()
    targets, _, _ = get_node_label(nx_graph)
    targets = torch.tensor(targets, device=device)
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        logits = classifier(embedding)
        logits = logits.to(device)
        train_loss = loss_fcn(logits[train_mask], targets[train_mask])
        train_loss.backward()
        optimizer.step()
    test_results = get_classification_report(targets[test_mask], logits[test_mask], output_dict=True)
    if os.path.isfile(join(logs, 'test_report.json')):
        with open(join(logs, 'test_report.json'), 'r') as f:
            report = json.load(f)
        report.append(test_results)
    else:
        report = [test_results]
    with open(join(logs, 'test_report.json'), 'w') as f:
        json.dump(report, f, indent=2)


def main(device):
    for bugtype in bug_list:
        print('Bugtype {}'.format(bugtype))
        for i in range(REPEAT):
            print(f'Train bugtype {bugtype} {i}-th')
            compressed_graph = f'{ROOT}/ge-sc-data/{TASK}/{COMPRESSED_GRAPH}/{bugtype}/buggy_curated/compressed_graphs.gpickle'
            nx_graph = nx.read_gpickle(compressed_graph)
            source_path = f'{ROOT}/ge-sc-data/{TASK}/cg/{bugtype}/buggy_curated'
            ## Base lines
            gae_embedded = f'{ROOT}/ge-sc-data/{TASK}/{COMPRESSED_GRAPH}/gesc_matrices_node_embedding/matrix_gae_dim128_of_core_graph_of_{bugtype}_{COMPRESSED_GRAPH}_buggy_curated_{DATA_ID}.pkl'
            line_embedded = f'{ROOT}/ge-sc-data/{TASK}/{COMPRESSED_GRAPH}/gesc_matrices_node_embedding/matrix_line_dim128_of_core_graph_of_{bugtype}_{COMPRESSED_GRAPH}_buggy_curated_{DATA_ID}.pkl'
            node2vec_embedded = f'{ROOT}/ge-sc-data/{TASK}/{COMPRESSED_GRAPH}/gesc_matrices_node_embedding/matrix_node2vec_dim128_of_core_graph_of_{bugtype}_{COMPRESSED_GRAPH}_buggy_curated_{DATA_ID}.pkl'

            # Base line
            base_metapath2vec(compressed_graph, source_path, bugtype, device)
            base_gae(nx_graph, gae_embedded, bugtype, device)
            base_line(nx_graph, line_embedded, bugtype, device)
            base_node2vec(nx_graph, gae_embedded, bugtype, device)


def get_avg_results(report_path, top_rate=0.5):
    with open(report_path, 'r') as f:
        results = json.load(f)
    buggy_f1 = []
    macro_f1 = []
    for i in range(len(results)):
        buggy_f1.append(results[i]['1']['f1-score'])
        macro_f1.append(results[i]['macro avg']['f1-score'])
    return round(mean(sorted(buggy_f1, reverse=True)[:int(top_rate*len(results))]) * 100, 2), round(mean(sorted(macro_f1, reverse=True)[:int(top_rate*len(results))]) * 100, 2)


def get_results():
    buggy_f1_report = {}
    macro_f1_report = {}
    for bugtype in bug_list:
        for model in base_models:
            report_path = f'{ROOT}/logs/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/{model}/{bugtype}/buggy_curated/test_report.json'
            buggy_f1, macro_f1 = get_avg_results(report_path, top_rate=0.5)
            # buggy_f1, macro_f1 = get_max_results(report_path)
            if model not in buggy_f1_report:
                buggy_f1_report[model] = [buggy_f1]
                macro_f1_report[model] = [macro_f1]
            else:
                buggy_f1_report[model].append(buggy_f1)
                macro_f1_report[model].append(macro_f1)
    for model in base_models:
        print(' ', end=' ')
        print(' \t'.join(['%.2f'%n for n in buggy_f1_report[model]]), end=r'')
        print()
        print(' ', end=' ')
        print(' \t'.join(['%.2f'%n for n in macro_f1_report[model]]), end=r'')
        print()


if __name__ == '__main__':
    torch.manual_seed(1)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if len(sys.argv) > 1:
        if sys.argv[1] == '--result':
            get_results()
    else:
        main(device)
