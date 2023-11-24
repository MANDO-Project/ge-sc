from math import ceil
import os
import argparse
import multiprocessing as mp
import threading
import time
from os.path import join
from threading import Thread

import pickle
import json
import networkx as nx
from numpy import zeros_like
import torch
from torch import nn
from tabulate import tabulate
from timebudget import timebudget
from sklearn.model_selection import train_test_split
from statistics import mean

from sco_models.utils import get_classification_report
from sco_models.graph_utils import reveert_map_node_embedding, load_hetero_nx_graph
from sco_models.model_hetero import MANDOGraphClassifier as GraphClassifier

# Arguments
parser = argparse.ArgumentParser('MANDO Experiments')
parser.add_argument('-s', '--seed', type=int, default=1,
                    help='Random seed')
parser.add_argument('-e', '--epochs', type=int, default=2,
                    help='Number of ephocs')
parser.add_argument('-rep', '--repeat', type=int, default=2,
                    help='Number of repetitions')
parser.add_argument('-b', '--bytecode', type=str, default='runtime',
                    help='Kind of bytecode')
                    
parser.add_argument('-r', '--result', action='store_true')
args = parser.parse_args().__dict__
    

torch.manual_seed(args['seed'])
ROOT = './experiments'
DATA_ID = 0
REPEAT = args['repeat']
EPOCHS = args['epochs']
TASK = "graph_classification"
COMPRESSED_GRAPH = 'cfg_cg'
DATASET = 'smartbugs'
STRUCTURE = 'han'
BYTECODE = args['bytecode']
TRAIN_RATE = 0.7
VAL_RATE = 0.3
ratio = 1


models = ['base_metapath2vec', 'base_line', 'base_node2vec', 'nodetype', 'metapath2vec', 'line', 'node2vec']
# bug_list = ['access_control', 'arithmetic', 'denial_of_service',
#             'front_running', 'reentrancy', 'time_manipulation', 
#             'unchecked_low_level_calls']
# models = ['base_metapath2vec', 'base_line', 'base_node2vec', 'nodetype', 'metapath2vec', 'line', 'node2vec', 'random_2', 'random_8', 'random_16', 'random_32', 'random_64', 'random_128', 'zeros_2', 'zeros_8', 'zeros_16', 'zeros_32', 'zeros_64', 'zeros_128']
# models = ['base_metapath2vec', 'base_line', 'base_node2vec', 'nodetype', 'metapath2vec', 'line', 'node2vec', 'random_32', 'random_64', 'random_128', 'zeros_32', 'zeros_64', 'zeros_128']
# models = ['base_lstm', 'lstm']
# feature_dim_list = [2, 8, 16, 32, 64, 128]
feature_dim_list = [32, 64]
# bug_list = ['ethor']
bug_list = [
            'access_control', 
            'arithmetic',
            'denial_of_service',
            'front_running', 
            'reentrancy', 
            'time_manipulation',
            'unchecked_low_level_calls'
            ]
file_counter = {'access_control': 57, 'arithmetic': 60, 'denial_of_service': 46,
                'front_running': 44, 'reentrancy': 71, 'time_manipulation': 50, 
                'unchecked_low_level_calls': 95}
print(f'Run experiments on {len(bug_list)} kinds of bug for {len(models)} kinds of model.')
print(f'Repeat {REPEAT} times and {EPOCHS} epochs for each experiment.')


def train(model, train_loader, labels, device):
    total_steps = EPOCHS
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.005, total_steps=total_steps)
    for _ in range(total_steps):
        optimizer.zero_grad()
        logits, _ = model(train_loader)
        logits = logits.to(device)
        train_loss = loss_fcn(logits, labels) 
        train_loss.backward()
        optimizer.step()
        scheduler.step()
    return model


def get_node_id_by_file_name(nx_graph):
    file_name_dict = {}
    for idx, data in nx_graph.nodes(data=True):
        source_name = data['source_file']
        if source_name not in file_name_dict:
            file_name_dict[source_name] = [idx]
        else:
            file_name_dict[source_name].append(idx)
    return file_name_dict


def save_last_hidden(hiddens, targets, contract_name, output):
    assert hiddens.shape[0] == targets.shape[0] == len(contract_name)
    logger = []
    for i in range(hiddens.shape[0]):
        logger.append({'contract_name': contract_name[i],
                       'hiddens': hiddens[i].tolist()  ,
                       'targets': targets[i].tolist()})
    with open(output, 'w') as f:
        json.dump(logger, f, indent=4)


def base_metapath2vec(compressed_graph, file_name_dict, dataset, bugtype, device, repeat):
    logs = f'{ROOT}/logs/{TASK}/byte_code/{DATASET}/{BYTECODE}/{STRUCTURE}/{COMPRESSED_GRAPH}/base_metapath2vec/{bugtype}/'
    if not os.path.exists(logs):
        os.makedirs(logs)
    output_models = f'{ROOT}/models/{TASK}/byte_code/{DATASET}/{BYTECODE}/{STRUCTURE}/{COMPRESSED_GRAPH}/base_metapath2vec/{bugtype}/'
    if not os.path.exists(output_models):
        os.makedirs(output_models)
    model = GraphClassifier(compressed_graph, feature_extractor=None, 
                                 node_feature='metapath2vec', device=device)
    features = model.symmetrical_global_graph.ndata['feat']
    nx_graph = load_hetero_nx_graph(compressed_graph)
    embedding = reveert_map_node_embedding(nx_graph, features)
    assert len(nx_graph.nodes()) == embedding.shape[0]
    X_train, X_val, y_train, y_val = dataset
    X_embedded_train = []
    X_embedded_val = []
    for file in X_train:
        X_embedded_train.append(torch.mean(embedding[file_name_dict[file]], 0).tolist())
    for file in X_val:
        X_embedded_val.append(torch.mean(embedding[file_name_dict[file]], 0).tolist())
    X_embedded_train = torch.tensor(X_embedded_train, device=device)
    X_embedded_val = torch.tensor(X_embedded_val, device=device)
    print('Training phase')
    classifier = torch.nn.Linear(128, 2)
    classifier.to(device)
    classifier.train()
    targets = torch.tensor(y_train, device=device)
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0005)
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        logits = classifier(X_embedded_train)
        logits = logits.to(device)
        train_loss = loss_fcn(logits, targets)
        train_loss.backward()
        optimizer.step()
    save_path = os.path.join(output_models, f'{bugtype}_{STRUCTURE}_{repeat}.pth')
    torch.save(classifier.state_dict(), save_path)
    classifier.eval()
    with torch.no_grad():
        logits = classifier(X_embedded_val)
        logits = logits.to(device)
        test_results = get_classification_report(y_val, logits, output_dict=True)
    if os.path.isfile(join(logs, 'test_report.json')):
        with open(join(logs, 'test_report.json'), 'r') as f:
            report = json.load(f)
        report.append(test_results)
    else:
        report = [test_results]
    with open(join(logs, 'test_report.json'), 'w') as f:
        json.dump(report, f, indent=2)


def base_gae(dataset, bugtype, gae_embedded, file_name_dict, device, repeat):
    logs = f'{ROOT}/logs/{TASK}/byte_code/{DATASET}/{BYTECODE}/{STRUCTURE}/{COMPRESSED_GRAPH}/base_gae/{bugtype}/'
    if not os.path.exists(logs):
        os.makedirs(logs)
    output_models = f'{ROOT}/models/{TASK}/byte_code/{DATASET}/{BYTECODE}/{STRUCTURE}/{COMPRESSED_GRAPH}/base_gae/{bugtype}/'
    if not os.path.exists(output_models):
        os.makedirs(output_models)
    X_train, X_val, y_train, y_val = dataset
    with open(gae_embedded, 'rb') as f:
        embedding = pickle.load(f, encoding="utf8")
        embedding = torch.tensor(embedding, device=device)
    X_embedded_train = []
    X_embedded_val = []
    for file in X_train:
        X_embedded_train.append(torch.mean(embedding[file_name_dict[file]], 0).tolist())
    for file in X_val:
        X_embedded_val.append(torch.mean(embedding[file_name_dict[file]], 0).tolist())
    X_embedded_train = torch.tensor(X_embedded_train, device=device)
    X_embedded_val = torch.tensor(X_embedded_val, device=device)
    print('Training phase')
    classifier = torch.nn.Linear(128, 2)
    classifier.to(device)
    classifier.train()
    targets = torch.tensor(y_train, device=device)
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0005)
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        logits = classifier(X_embedded_train)
        logits = logits.to(device)
        train_loss = loss_fcn(logits, targets)
        train_loss.backward()
        optimizer.step()
    save_path = os.path.join(output_models, f'{bugtype}_{STRUCTURE}_{repeat}.pth')
    torch.save(classifier.state_dict(), save_path)
    classifier.eval()
    with torch.no_grad():
        logits = classifier(X_embedded_val)
        logits = logits.to(device)
        test_results = get_classification_report(y_val, logits, output_dict=True)
    if os.path.isfile(join(logs, 'test_report.json')):
        with open(join(logs, 'test_report.json'), 'r') as f:
            report = json.load(f)
        report.append(test_results)
    else:
        report = [test_results]
    with open(join(logs, 'test_report.json'), 'w') as f:
        json.dump(report, f, indent=2)


def base_line(dataset, bugtype, line_embedded, file_name_dict, device, repeat):
    logs = f'{ROOT}/logs/{TASK}/byte_code/{DATASET}/{BYTECODE}/{STRUCTURE}/{COMPRESSED_GRAPH}/base_line/{bugtype}/'
    if not os.path.exists(logs):
        os.makedirs(logs)
    output_models = f'{ROOT}/models/{TASK}/byte_code/{DATASET}/{BYTECODE}/{STRUCTURE}/{COMPRESSED_GRAPH}/base_line/{bugtype}/'
    if not os.path.exists(output_models):
        os.makedirs(output_models)
    X_train, X_val, y_train, y_val = dataset
    with open(line_embedded, 'rb') as f:
        embedding = pickle.load(f, encoding="utf8")
        embedding = torch.tensor(embedding, device=device)
    X_embedded_train = []
    X_embedded_val = []
    for file in X_train:
        X_embedded_train.append(torch.mean(embedding[file_name_dict[file]], 0).tolist())
    for file in X_val:
        X_embedded_val.append(torch.mean(embedding[file_name_dict[file]], 0).tolist())
    X_embedded_train = torch.tensor(X_embedded_train, device=device)
    X_embedded_val = torch.tensor(X_embedded_val, device=device)
    print('Training phase')
    classifier = torch.nn.Linear(128, 2)
    classifier.to(device)
    classifier.train()
    targets = torch.tensor(y_train, device=device)
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0005)
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        logits = classifier(X_embedded_train)
        logits = logits.to(device)
        train_loss = loss_fcn(logits, targets)
        train_loss.backward()
        optimizer.step()
    save_path = os.path.join(output_models, f'{bugtype}_{STRUCTURE}_{repeat}.pth')
    torch.save(classifier.state_dict(), save_path)
    classifier.eval()
    with torch.no_grad():
        logits = classifier(X_embedded_val)
        logits = logits.to(device)
        test_results = get_classification_report(y_val, logits, output_dict=True)
    if os.path.isfile(join(logs, 'test_report.json')):
        with open(join(logs, 'test_report.json'), 'r') as f:
            report = json.load(f)
        report.append(test_results)
    else:
        report = [test_results]
    with open(join(logs, 'test_report.json'), 'w') as f:
        json.dump(report, f, indent=2)


def base_node2vec(dataset, bugtype, node2vec_embedded, file_name_dict, device, repeat):
    logs = f'{ROOT}/logs/{TASK}/byte_code/{DATASET}/{BYTECODE}/{STRUCTURE}/{COMPRESSED_GRAPH}/base_node2vec/{bugtype}/'
    if not os.path.exists(logs):
        os.makedirs(logs)
    output_models = f'{ROOT}/models/{TASK}/byte_code/{DATASET}/{BYTECODE}/{STRUCTURE}/{COMPRESSED_GRAPH}/base_node2vec/{bugtype}/'
    if not os.path.exists(output_models):
        os.makedirs(output_models)
    X_train, X_val, y_train, y_val = dataset
    with open(node2vec_embedded, 'rb') as f:
        embedding = pickle.load(f, encoding="utf8")
        embedding = torch.tensor(embedding, device=device)
    X_embedded_train = []
    X_embedded_val = []
    for file in X_train:
        X_embedded_train.append(torch.mean(embedding[file_name_dict[file]], 0).tolist())
    for file in X_val:
        X_embedded_val.append(torch.mean(embedding[file_name_dict[file]], 0).tolist())
    X_embedded_train = torch.tensor(X_embedded_train, device=device)
    X_embedded_val = torch.tensor(X_embedded_val, device=device)
    print('Training phase')
    classifier = torch.nn.Linear(128, 2)
    classifier.to(device)
    classifier.train()
    targets = torch.tensor(y_train, device=device)
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0005)
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        logits = classifier(X_embedded_train)
        logits = logits.to(device)
        train_loss = loss_fcn(logits, targets)
        train_loss.backward()
        optimizer.step()
    save_path = os.path.join(output_models, f'{bugtype}_{STRUCTURE}_{repeat}.pth')
    torch.save(classifier.state_dict(), save_path)
    classifier.eval()
    with torch.no_grad():
        logits = classifier(X_embedded_val)
        logits = logits.to(device)
        test_results = get_classification_report(y_val, logits, output_dict=True)
    if os.path.isfile(join(logs, 'test_report.json')):
        with open(join(logs, 'test_report.json'), 'r') as f:
            report = json.load(f)
        report.append(test_results)
    else:
        report = [test_results]
    with open(join(logs, 'test_report.json'), 'w') as f:
        json.dump(report, f, indent=2)


def nodetype(compressed_graph, dataset, feature_extractor, bugtype, device, repeat):
    logs = f'{ROOT}/logs/{TASK}/byte_code/{DATASET}/{BYTECODE}/{STRUCTURE}/{COMPRESSED_GRAPH}/nodetype/{bugtype}/'
    if not os.path.exists(logs):
        os.makedirs(logs)
    output_models = f'{ROOT}/models/{TASK}/byte_code/{DATASET}/{BYTECODE}/{STRUCTURE}/{COMPRESSED_GRAPH}/nodetype/{bugtype}/'
    if not os.path.exists(output_models):
        os.makedirs(output_models)
    feature_extractor = feature_extractor
    node_feature = 'nodetype'
    model = GraphClassifier(compressed_graph, feature_extractor=feature_extractor, 
                                 node_feature=node_feature, device=device)
    model.reset_parameters()
    model.to(device)
    X_train, X_val, y_train, y_val = dataset
    model = train(model, X_train, y_train, device)
    save_path = os.path.join(output_models, f'{bugtype}_{STRUCTURE}_{repeat}.pth')
    torch.save(model.state_dict(), save_path)
    model.eval()
    with torch.no_grad():
        logits, hiddens = model(X_val)
        logits = logits.to(device)
        # test_acc, test_micro_f1, test_macro_f1 = score(y_val, logits)
        test_results = get_classification_report(y_val, logits, output_dict=True)
    save_last_hidden(hiddens, y_val, X_val, join(logs, 'last_hiddens.json'))
    if os.path.isfile(join(logs, 'test_report.json')):
        with open(join(logs, 'test_report.json'), 'r') as f:
            report = json.load(f)
        report.append(test_results)
    else:
        report = [test_results]
    with open(join(logs, 'test_report.json'), 'w') as f:
        json.dump(report, f, indent=2)


def metapath2vec(compressed_graph, dataset, feature_extractor, bugtype, device, repeat):
    logs = f'{ROOT}/logs/{TASK}/byte_code/{DATASET}/{BYTECODE}/{STRUCTURE}/{COMPRESSED_GRAPH}/metapath2vec/{bugtype}/'
    if not os.path.exists(logs):
        os.makedirs(logs)
    output_models = f'{ROOT}/models/{TASK}/byte_code/{DATASET}/{BYTECODE}/{STRUCTURE}/{COMPRESSED_GRAPH}/metapath2vec/{bugtype}/'
    if not os.path.exists(output_models):
        os.makedirs(output_models)
    feature_extractor = feature_extractor
    node_feature = 'metapath2vec'
    model = GraphClassifier(compressed_graph, feature_extractor=feature_extractor, 
                                 node_feature=node_feature, device=device)
    model.reset_parameters()
    model.to(device)
    X_train, X_val, y_train, y_val = dataset
    model = train(model, X_train, y_train, device)
    save_path = os.path.join(output_models, f'{bugtype}_{STRUCTURE}_{repeat}.pth')
    torch.save(model.state_dict(), save_path)
    model.eval()
    with torch.no_grad():
        logits, hiddens = model(X_val)
        logits = logits.to(device)
        # test_acc, test_micro_f1, test_macro_f1 = score(y_val, logits)
        test_results = get_classification_report(y_val, logits, output_dict=True)
    save_last_hidden(hiddens, y_val, X_val, join(logs, 'last_hiddens.json'))
    if os.path.isfile(join(logs, 'test_report.json')):
        with open(join(logs, 'test_report.json'), 'r') as f:
            report = json.load(f)
        report.append(test_results)
    else:
        report = [test_results]
    with open(join(logs, 'test_report.json'), 'w') as f:
        json.dump(report, f, indent=2)


def gae(compressed_graph, dataset, feature_extractor, bugtype, device, repeat):
    logs = f'{ROOT}/logs/{TASK}/byte_code/{DATASET}/{BYTECODE}/{STRUCTURE}/{COMPRESSED_GRAPH}/gae/{bugtype}/'
    if not os.path.exists(logs):
        os.makedirs(logs)
    output_models = f'{ROOT}/models/{TASK}/byte_code/{DATASET}/{BYTECODE}/{STRUCTURE}/{COMPRESSED_GRAPH}/gae/{bugtype}/'
    if not os.path.exists(output_models):
        os.makedirs(output_models)
    feature_extractor = feature_extractor
    node_feature = 'gae'
    model = GraphClassifier(compressed_graph, feature_extractor=feature_extractor, 
                                 node_feature=node_feature, device=device)
    model.reset_parameters()
    model.to(device)
    X_train, X_val, y_train, y_val = dataset
    model = train(model, X_train, y_train, device)
    save_path = os.path.join(output_models, f'{bugtype}_{STRUCTURE}_{repeat}.pth')
    torch.save(model.state_dict(), save_path)
    model.eval()
    with torch.no_grad():
        logits, hiddens = model(X_val)
        logits = logits.to(device)
        # test_acc, test_micro_f1, test_macro_f1 = score(y_val, logits)
        test_results = get_classification_report(y_val, logits, output_dict=True)
    save_last_hidden(hiddens, y_val, X_val, join(logs, 'last_hiddens.json'))
    if os.path.isfile(join(logs, 'test_report.json')):
        with open(join(logs, 'test_report.json'), 'r') as f:
            report = json.load(f)
        report.append(test_results)
    else:
        report = [test_results]
    with open(join(logs, 'test_report.json'), 'w') as f:
        json.dump(report, f, indent=2)


def line(compressed_graph, dataset, feature_extractor, bugtype, device, repeat):
    logs = f'{ROOT}/logs/{TASK}/byte_code/{DATASET}/{BYTECODE}/{STRUCTURE}/{COMPRESSED_GRAPH}/line/{bugtype}/'
    if not os.path.exists(logs):
        os.makedirs(logs)
    output_models = f'{ROOT}/models/{TASK}/byte_code/{DATASET}/{BYTECODE}/{STRUCTURE}/{COMPRESSED_GRAPH}/line/{bugtype}/'
    if not os.path.exists(output_models):
        os.makedirs(output_models)
    feature_extractor = feature_extractor
    node_feature = 'line'
    model = GraphClassifier(compressed_graph, feature_extractor=feature_extractor, 
                                 node_feature=node_feature, device=device)
    model.reset_parameters()
    model.to(device)
    X_train, X_val, y_train, y_val = dataset
    model = train(model, X_train, y_train, device)
    save_path = os.path.join(output_models, f'{bugtype}_{STRUCTURE}_{repeat}.pth')
    torch.save(model.state_dict(), save_path)
    model.eval()
    with torch.no_grad():
        logits, hiddens = model(X_val)
        logits = logits.to(device)
        # test_acc, test_micro_f1, test_macro_f1 = score(y_val, logits)
        test_results = get_classification_report(y_val, logits, output_dict=True)
    save_last_hidden(hiddens, y_val, X_val, join(logs, 'last_hiddens.json'))
    if os.path.isfile(join(logs, 'test_report.json')):
        with open(join(logs, 'test_report.json'), 'r') as f:
            report = json.load(f)
        report.append(test_results)
    else:
        report = [test_results]
    with open(join(logs, 'test_report.json'), 'w') as f:
        json.dump(report, f, indent=2)


def node2vec(compressed_graph, dataset, feature_extractor, bugtype, device, repeat):
    logs = f'{ROOT}/logs/{TASK}/byte_code/{DATASET}/{BYTECODE}/{STRUCTURE}/{COMPRESSED_GRAPH}/node2vec/{bugtype}/'
    if not os.path.exists(logs):
        os.makedirs(logs)
    output_models = f'{ROOT}/models/{TASK}/byte_code/{DATASET}/{BYTECODE}/{STRUCTURE}/{COMPRESSED_GRAPH}/node2vec/{bugtype}/'
    if not os.path.exists(output_models):
        os.makedirs(output_models)
    feature_extractor = feature_extractor
    node_feature = 'node2vec'
    model = GraphClassifier(compressed_graph, feature_extractor=feature_extractor, 
                                 node_feature=node_feature, device=device)
    model.reset_parameters()
    model.to(device)
    X_train, X_val, y_train, y_val = dataset
    model = train(model, X_train, y_train, device)
    save_path = os.path.join(output_models, f'{bugtype}_{STRUCTURE}_{repeat}.pth')
    torch.save(model.state_dict(), save_path)
    model.eval()
    with torch.no_grad():
        logits, hiddens = model(X_val)
        logits = logits.to(device)
        # test_acc, test_micro_f1, test_macro_f1 = score(y_val, logits)
        test_results = get_classification_report(y_val, logits, output_dict=True)
    save_last_hidden(hiddens, y_val, X_val, join(logs, 'last_hiddens.json'))
    if os.path.isfile(join(logs, 'test_report.json')):
        with open(join(logs, 'test_report.json'), 'r') as f:
            report = json.load(f)
        report.append(test_results)
    else:
        report = [test_results]
    with open(join(logs, 'test_report.json'), 'w') as f:
        json.dump(report, f, indent=2)


def lstm(compressed_graph, dataset, bugtype, device, repeat):
    logs = f'{ROOT}/logs/{TASK}/byte_code/{DATASET}/{BYTECODE}/{STRUCTURE}/{COMPRESSED_GRAPH}/lstm/{bugtype}/'
    if not os.path.exists(logs):
        os.makedirs(logs, exist_ok=True)
    output_models = f'{ROOT}/models/{TASK}/byte_code/{DATASET}/{BYTECODE}/{STRUCTURE}/{COMPRESSED_GRAPH}/lstm/{bugtype}/'
    if not os.path.exists(output_models):
        os.makedirs(output_models, exist_ok=True)
    feature_extractor = 256
    node_feature = 'lstm'
    model = GraphClassifier(compressed_graph, feature_extractor=feature_extractor, 
                                 node_feature=node_feature, device=device)
    model.reset_parameters()
    # model = nn.DataParallel(model)
    model.to(device)
    X_train, X_val, y_train, y_val = dataset
    model = train(model, X_train, y_train, device)
    save_path = os.path.join(output_models, f'{bugtype}_{STRUCTURE}_{repeat}.pth')
    torch.save(model.state_dict(), save_path)
    model.eval()
    with torch.no_grad():
        logits, hiddens = model(X_val)
        logits = logits.to(device)
        # test_acc, test_micro_f1, test_macro_f1 = score(y_val, logits)
        test_results = get_classification_report(y_val, logits, output_dict=True)
    save_last_hidden(hiddens, y_val, X_val, join(logs, 'last_hiddens.json'))
    if os.path.isfile(join(logs, 'test_report.json')):
        with open(join(logs, 'test_report.json'), 'r') as f:
            report = json.load(f)
        report.append(test_results)
    else:
        report = [test_results]
    with open(join(logs, 'test_report.json'), 'w') as f:
        json.dump(report, f, indent=2)


def random(compressed_graph, dataset, feature_dims, bugtype, device, repeat):
    logs = f'{ROOT}/logs/{TASK}/byte_code/{DATASET}/{BYTECODE}/{STRUCTURE}/{COMPRESSED_GRAPH}/random_{feature_dims}/{bugtype}/'
    if not os.path.exists(logs):
        os.makedirs(logs, exist_ok=True)
    output_models = f'{ROOT}/models/{TASK}/byte_code/{DATASET}/{BYTECODE}/{STRUCTURE}/{COMPRESSED_GRAPH}/random_{feature_dims}/{bugtype}/'
    if not os.path.exists(output_models):
        os.makedirs(output_models, exist_ok=True)
    feature_extractor = feature_dims
    node_feature = 'random'
    model = GraphClassifier(compressed_graph, feature_extractor=feature_extractor, 
                                 node_feature=node_feature, device=device)
    model.reset_parameters()
    # model = nn.DataParallel(model)
    model.to(device)
    X_train, X_val, y_train, y_val = dataset
    model = train(model, X_train, y_train, device)
    save_path = os.path.join(output_models, f'{bugtype}_{STRUCTURE}_{repeat}.pth')
    torch.save(model.state_dict(), save_path)
    model.eval()
    with torch.no_grad():
        logits, hiddens = model(X_val)
        logits = logits.to(device)
        # test_acc, test_micro_f1, test_macro_f1 = score(y_val, logits)
        test_results = get_classification_report(y_val, logits, output_dict=True)
    save_last_hidden(hiddens, y_val, X_val, join(logs, 'last_hiddens.json'))
    if os.path.isfile(join(logs, 'test_report.json')):
        with open(join(logs, 'test_report.json'), 'r') as f:
            report = json.load(f)
        report.append(test_results)
    else:
        report = [test_results]
    with open(join(logs, 'test_report.json'), 'w') as f:
        json.dump(report, f, indent=2)


def zeros(compressed_graph, dataset, feature_dims, bugtype, device, repeat):
    logs = f'{ROOT}/logs/{TASK}/byte_code/{DATASET}/{BYTECODE}/{STRUCTURE}/{COMPRESSED_GRAPH}/zeros_{feature_dims}/{bugtype}/'
    if not os.path.exists(logs):
        os.makedirs(logs, exist_ok=True)
    output_models = f'{ROOT}/models/{TASK}/byte_code/{DATASET}/{BYTECODE}/{STRUCTURE}/{COMPRESSED_GRAPH}/zeros_{feature_dims}/{bugtype}/'
    if not os.path.exists(output_models):
        os.makedirs(output_models, exist_ok=True)
    feature_extractor = feature_dims
    node_feature = 'zeros'
    model = GraphClassifier(compressed_graph, feature_extractor=feature_extractor, 
                                 node_feature=node_feature, device=device)
    model.reset_parameters()
    # model = nn.DataParallel(model)
    model.to(device)
    X_train, X_val, y_train, y_val = dataset
    model = train(model, X_train, y_train, device)
    save_path = os.path.join(output_models, f'{bugtype}_{STRUCTURE}_{repeat}.pth')
    torch.save(model.state_dict(), save_path)
    model.eval()
    with torch.no_grad():
        logits, hiddens = model(X_val)
        logits = logits.to(device)
        # test_acc, test_micro_f1, test_macro_f1 = score(y_val, logits)
        test_results = get_classification_report(y_val, logits, output_dict=True)
    save_last_hidden(hiddens, y_val, X_val, join(logs, 'last_hiddens.json'))
    if os.path.isfile(join(logs, 'test_report.json')):
        with open(join(logs, 'test_report.json'), 'r') as f:
            report = json.load(f)
        report.append(test_results)
    else:
        report = [test_results]
    with open(join(logs, 'test_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

@timebudget
def main(device):
    for bugtype in bug_list:
        print('Bugtype {}'.format(bugtype))
        for i in range(REPEAT):
            print(f'Train bugtype {bugtype} {i}-th')
            compressed_graph = f'{ROOT}/ge-sc-data/byte_code/{DATASET}/{BYTECODE}/gpickles/{bugtype}/clean_{file_counter[bugtype]}_buggy_curated_0/compressed_graphs/{BYTECODE}_balanced_compressed_graphs.gpickle'
            nx_graph = nx.read_gpickle(compressed_graph)
            file_name_dict = get_node_id_by_file_name(nx_graph)
            # label = f'{ROOT}/ge-sc-data/byte_code/{DATASET}/{BYTECODE}/gpickles/{bugtype}/clean_{file_counter[bugtype]}_buggy_curated_0/graph_labels.json'
            label = f'{ROOT}/ge-sc-data/byte_code/smartbugs/contract_labels/{bugtype}/{BYTECODE}_balanced_contract_labels.json'
            # source_path = f'{ROOT}/ge-sc-data/byte_code/{DATASET}/{BYTECODE}/gpickles/{bugtype}/clean_{file_counter[bugtype]}_buggy_curated_0/'
            with open(label, 'r') as f:
                annotations = json.load(f)
            # total_files = [f for f in os.listdir(source_path) if f.endswith('.gpickle')]
            total_files = [anno['contract_name'] for anno in annotations]
            assert len(total_files) <= len(annotations)
            # targets = []
            # for file in total_files:
            #     try:
            #         target = next(anno['targets'] for anno in annotations if anno['contract_name'] == file.replace('.gpickle', '.sol'))
            #     except StopIteration:
            #         raise f'{file} not found!'
            #     targets.append(target)
            targets = [anno['targets'] for anno in annotations]
            targets = torch.tensor(targets, device=device)
            assert len(total_files) == len(targets)
            print(len(total_files), len(targets))
            X_train, X_val, y_train, y_val = train_test_split(total_files, targets, train_size=TRAIN_RATE)
            dataset = (tuple(X_train), tuple(X_val), y_train, y_val)
            print('Start training with {}/{} train/val smart contracts'.format(len(X_train), len(X_val)))
            gae_embedded = f'{ROOT}/ge-sc-data/byte_code/{DATASET}/{BYTECODE}/gpickles/gesc_matrices_node_embedding/matrix_gae_dim128_of_core_graph_of_{bugtype}_{COMPRESSED_GRAPH}_compressed_graphs.pkl'
            line_embedded = f'{ROOT}/ge-sc-data/byte_code/{DATASET}/{BYTECODE}/gpickles/gesc_matrices_node_embedding/balanced/matrix_line_dim128_of_core_graph_of_{bugtype}_{BYTECODE}_balanced_{COMPRESSED_GRAPH}_compressed_graphs.pkl'
            node2vec_embedded = f'{ROOT}/ge-sc-data/byte_code/{DATASET}/{BYTECODE}/gpickles/gesc_matrices_node_embedding/balanced/matrix_node2vec_dim128_of_core_graph_of_{bugtype}_{BYTECODE}_balanced_{COMPRESSED_GRAPH}_compressed_graphs.pkl'
            # Run experiments
            # Base lines
            # base_metapath2vec(compressed_graph, file_name_dict, dataset, bugtype, device, i)
            # base_line(dataset, bugtype, line_embedded, file_name_dict, device, i)
            # base_node2vec(dataset, bugtype, node2vec_embedded, file_name_dict, device, i)

            ## Out models
            # nodetype(compressed_graph, dataset, None, bugtype, device, i)
            # metapath2vec(compressed_graph, dataset, None, bugtype, device, i)
            # gae(compressed_graph, dataset, line_embedded, bugtype, device, i)
            # line(compressed_graph, dataset, line_embedded, bugtype, device, i)
            # node2vec(compressed_graph, dataset, node2vec_embedded, bugtype, device, i)
            lstm(compressed_graph, dataset, bugtype, device, i)
            # random(compressed_graph, dataset, 2, bugtype, device, i)
            # random(compressed_graph, dataset, 8, bugtype, device, i)
            # random(compressed_graph, dataset, 16, bugtype, device, i)
            # random(compressed_graph, dataset, 32, bugtype, device, i)
            # random(compressed_graph, dataset, 64, bugtype, device, i)
            # random(compressed_graph, dataset, 128, bugtype, device, i)
            # zeros(compressed_graph, dataset, 2, bugtype, device, i)
            # zeros(compressed_graph, dataset, 8, bugtype, device, i)
            # zeros(compressed_graph, dataset, 16, bugtype, device, i)
            # zeros(compressed_graph, dataset, 32, bugtype, device, i)
            # zeros(compressed_graph, dataset, 64, bugtype, device, i)
            # zeros(compressed_graph, dataset, 128, bugtype, device, i)




def get_avg_results(report_path, top_rate=0.5):
    with open(report_path, 'r') as f:
        results = json.load(f)
    buggy_f1 = []
    macro_f1 = []
    for i in range(len(results)):
        buggy_f1.append(results[i]['1']['f1-score'])
        macro_f1.append(results[i]['macro avg']['f1-score'])
    return round(mean(sorted(buggy_f1, reverse=True)[:int(top_rate*len(results))]) * 100, 2), round(mean(sorted(macro_f1, reverse=True)[:int(top_rate*len(results))]) * 100, 2)


def get_max_results(report_path):
    with open(report_path, 'r') as f:
        results = json.load(f)
    buggy_f1 = []
    macro_f1 = []
    for i in range(len(results)):
        buggy_f1.append(results[i]['1']['f1-score'])
        macro_f1.append(results[i]['macro avg']['f1-score'])
    return round(max(buggy_f1) * 100, 2), round(max(macro_f1) * 100, 2)


def get_results():
    buggy_f1_report = {}
    macro_f1_report = {}
    for bugtype in bug_list:
        for model in models:
            if  model in ['gae', 'base_gae'] and bugtype in ['arithmetic', 'front_running', 'reentrancy', 'unchecked_low_level_calls']:
                buggy_f1, macro_f1 = '-', '-'
            else:
                report_path = f'{ROOT}/logs/{TASK}/byte_code/{DATASET}/{BYTECODE}/{STRUCTURE}/{COMPRESSED_GRAPH}/{model}/{bugtype}/test_report.json'
                buggy_f1, macro_f1 = get_avg_results(report_path, top_rate=0.5)
            # buggy_f1, macro_f1 = get_max_results(report_path)
            if model not in buggy_f1_report:
                buggy_f1_report[model] = [buggy_f1]
                macro_f1_report[model] = [macro_f1]
            else:
                buggy_f1_report[model].append(buggy_f1)
                macro_f1_report[model].append(macro_f1)
    data = []
    for model in models:
        buggy_f1_row = []
        macro_f1_row = []
        for i in range(len(buggy_f1_report[model])):
            buggy_f1 = buggy_f1_report[model][i]
            macro_f1 = macro_f1_report[model][i]
            buggy_f1_row.append('%.2f'%buggy_f1 + '%' if isinstance(buggy_f1, float) else buggy_f1)
            macro_f1_row.append('%.2f'%macro_f1 + '%' if isinstance(macro_f1, float) else macro_f1)
        data.append([model, 'Buggy-F1'] + buggy_f1_row)
        data.append([model, 'Macro-F1'] + macro_f1_row)
        print(' ', end=' ')
        print(' \t'.join(buggy_f1_row), end=r'')
        print()
        print(' ', end=' ')
        print(' \t'.join(macro_f1_row), end=r'')
        print()
    print()
    print(tabulate(data, headers=bug_list, tablefmt='orgtbl'))


def get_exp_time(report_path):
    with open(report_path, 'r') as f:
        results = json.load(f)
    train_time = []
    test_time = []
    for i in range(len(results)):
        train_time.append(float(results[i]['train_time']))
        test_time.append(float(results[i]['test_time']))
    return round(mean(train_time), 2), round(mean(test_time), 2)

def get_runtime_result():
    train_time_report = {}
    test_time_report = {}
    for bugtype in bug_list:
        for model in models:
            report_path = f'{ROOT}/logs/{TASK}/source_code/{STRUCTURE}/{COMPRESSED_GRAPH}/{model}/{bugtype}/clean_{file_counter[bugtype]}_buggy_curated_0/test_report.json'
            train_time, test_time = get_exp_time(report_path)
            # train_time, test_time = get_max_results(report_path)
            if model not in train_time_report:
                train_time_report[model] = [train_time]
                test_time_report[model] = [test_time]
            else:
                train_time_report[model].append(train_time)
                test_time_report[model].append(test_time)
    avg_train_time = []
    avg_test_time = []
    for i in range(len(bug_list)):
        bug_train_list = []
        bug_test_list = []
        for model in models[-5:]:
            bug_train_list.append(train_time_report[model][i])
            bug_test_list.append(test_time_report[model][i])
        avg_train_time.append(mean(bug_train_list))
        avg_test_time.append(mean(bug_test_list))
    print(avg_train_time)
    for i in range(len(bug_list)):
        print(f'{ceil(avg_train_time[i])}/{ceil(avg_test_time[i])}', end=' ')
        print('&', end = ' ')


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    mps_device = 'mps'
    if args['result']:
            get_runtime_result()
    else:
        main(device)
