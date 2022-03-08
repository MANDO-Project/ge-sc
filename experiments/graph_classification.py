import os
import argparse
from os.path import join

import pickle
import json
import networkx as nx
import torch
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from statistics import mean

from sco_models.model_hetero import MANDOGraphClassifier
from sco_models.model_hgt import HGTVulGraphClassifier
from sco_models.utils import get_classification_report
from sco_models.graph_utils import reveert_map_node_embedding, load_hetero_nx_graph


# Arguments
parser = argparse.ArgumentParser('MANDO Experiments')
parser.add_argument('-s', '--seed', type=int, default=1,
                    help='Random seed')
parser.add_argument('-e', '--epochs', type=int, default=2,
                    help='Random seed')
parser.add_argument('-rep', '--repeat', type=int, default=2,
                    help='Random seed')
parser.add_argument('-r', '--result', action='store_true')
args = parser.parse_args().__dict__

torch.manual_seed(args['seed'])
ROOT = './experiments'
DATA_ID = 0
REPEAT = args['repeat']
EPOCHS = args['epochs']
TASK = "graph_classification"
STRUCTURE = 'hgt'
COMPRESSED_GRAPH = 'cfg_cg'
DATASET = 'clean_50_buggy_curated'
TRAIN_RATE = 0.7
VAL_RATE = 0.3
ratio = 1


models = ['base_metapath2vec', 'base_gae', 'base_line', 'base_node2vec', 'nodetype', 'metapath2vec', 'gae', 'line', 'node2vec']
bug_list = ['access_control', 'arithmetic', 'denial_of_service',
            'front_running', 'reentrancy', 'time_manipulation', 
            'unchecked_low_level_calls']
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


def base_metapath2vec(compressed_graph, source_path, file_name_dict, dataset, bugtype, device):
    logs = f'{ROOT}/logs/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/base_metapath2vec/{bugtype}/clean_{ratio*file_counter[bugtype]}_buggy_curated_{DATA_ID}/'
    if not os.path.exists(logs):
        os.makedirs(logs)
    output_models = f'{ROOT}/models/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/base_metapath2vec/{bugtype}/clean_{ratio*file_counter[bugtype]}_buggy_curated_{DATA_ID}/'
    if not os.path.exists(output_models):
        os.makedirs(output_models)
    model = HGTVulGraphClassifier(compressed_graph, source_path, feature_extractor=None, 
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
    save_path = os.path.join(output_models, f'hgt.pth')
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


def base_gae(dataset, bugtype, gae_embedded, file_name_dict, device):
    logs = f'{ROOT}/logs/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/base_gae/{bugtype}/clean_{ratio*file_counter[bugtype]}_buggy_curated_{DATA_ID}/'
    if not os.path.exists(logs):
        os.makedirs(logs)
    output_models = f'{ROOT}/models/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/base_gae/{bugtype}/clean_{ratio*file_counter[bugtype]}_buggy_curated_{DATA_ID}/'
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
    save_path = os.path.join(output_models, f'hgt.pth')
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


def base_line(dataset, bugtype, gae_embedded, file_name_dict, device):
    logs = f'{ROOT}/logs/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/base_line/{bugtype}/clean_{ratio*file_counter[bugtype]}_buggy_curated_{DATA_ID}/'
    if not os.path.exists(logs):
        os.makedirs(logs)
    output_models = f'{ROOT}/models/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/base_line/{bugtype}/clean_{ratio*file_counter[bugtype]}_buggy_curated_{DATA_ID}/'
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
    save_path = os.path.join(output_models, f'hgt.pth')
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


def base_node2vec(dataset, bugtype, gae_embedded, file_name_dict, device):
    logs = f'{ROOT}/logs/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/base_node2vec/{bugtype}/clean_{ratio*file_counter[bugtype]}_buggy_curated_{DATA_ID}/'
    if not os.path.exists(logs):
        os.makedirs(logs)
    output_models = f'{ROOT}/models/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/base_node2vec/{bugtype}/clean_{ratio*file_counter[bugtype]}_buggy_curated_{DATA_ID}/'
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
    save_path = os.path.join(output_models, f'hgt.pth')
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


def nodetype(compressed_graph, source_path, dataset, bugtype, device):
    logs = f'{ROOT}/logs/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/nodetype/{bugtype}/clean_{ratio*file_counter[bugtype]}_buggy_curated_{DATA_ID}/'
    if not os.path.exists(logs):
        os.makedirs(logs)
    output_models = f'{ROOT}/models/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/nodetype/{bugtype}/clean_{ratio*file_counter[bugtype]}_buggy_curated_{DATA_ID}/'
    if not os.path.exists(output_models):
        os.makedirs(output_models)
    feature_extractor = None
    node_feature = 'nodetype'
    model = HGTVulGraphClassifier(compressed_graph, source_path, feature_extractor=feature_extractor, 
                                 node_feature=node_feature, device=device)
    model.reset_parameters()
    model.to(device)
    X_train, X_val, y_train, y_val = dataset
    model = train(model, X_train, y_train, device)
    save_path = os.path.join(output_models, f'hgt.pth')
    torch.save(model.state_dict(), save_path)
    model.eval()
    with torch.no_grad():
        logits, _ = model(X_val)
        logits = logits.to(device)
        # test_acc, test_micro_f1, test_macro_f1 = score(y_val, logits)
        test_results = get_classification_report(y_val, logits, output_dict=True)
    if os.path.isfile(join(logs, 'test_report.json')):
        with open(join(logs, 'test_report.json'), 'r') as f:
            report = json.load(f)
        report.append(test_results)
    else:
        report = [test_results]
    with open(join(logs, 'test_report.json'), 'w') as f:
        json.dump(report, f, indent=2)


def metapath2vec(compressed_graph, source_path, dataset, bugtype, device):
    logs = f'{ROOT}/logs/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/metapath2vec/{bugtype}/clean_{ratio*file_counter[bugtype]}_buggy_curated_{DATA_ID}/'
    if not os.path.exists(logs):
        os.makedirs(logs)
    output_models = f'{ROOT}/models/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/metapath2vec/{bugtype}/clean_{ratio*file_counter[bugtype]}_buggy_curated_{DATA_ID}/'
    if not os.path.exists(output_models):
        os.makedirs(output_models)
    feature_extractor = None
    node_feature = 'metapath2vec'
    model = HGTVulGraphClassifier(compressed_graph, source_path, feature_extractor=feature_extractor, 
                                 node_feature=node_feature, device=device)
    model.reset_parameters()
    model.to(device)
    X_train, X_val, y_train, y_val = dataset
    model = train(model, X_train, y_train, device)
    save_path = os.path.join(output_models, f'hgt.pth')
    torch.save(model.state_dict(), save_path)
    model.eval()
    with torch.no_grad():
        logits, _ = model(X_val)
        logits = logits.to(device)
        # test_acc, test_micro_f1, test_macro_f1 = score(y_val, logits)
        test_results = get_classification_report(y_val, logits, output_dict=True)
    if os.path.isfile(join(logs, 'test_report.json')):
        with open(join(logs, 'test_report.json'), 'r') as f:
            report = json.load(f)
        report.append(test_results)
    else:
        report = [test_results]
    with open(join(logs, 'test_report.json'), 'w') as f:
        json.dump(report, f, indent=2)


def gae(compressed_graph, source_path, dataset, bugtype, device):
    logs = f'{ROOT}/logs/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/gae/{bugtype}/clean_{ratio*file_counter[bugtype]}_buggy_curated_{DATA_ID}/'
    if not os.path.exists(logs):
        os.makedirs(logs)
    output_models = f'{ROOT}/models/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/gae/{bugtype}/clean_{ratio*file_counter[bugtype]}_buggy_curated_{DATA_ID}/'
    if not os.path.exists(output_models):
        os.makedirs(output_models)
    feature_extractor = f'{ROOT}/ge-sc-data/source_code/gesc_matrices_node_embedding/matrix_gae_dim128_of_core_graph_of_{bugtype}_{COMPRESSED_GRAPH}_clean_{file_counter[bugtype]}_{DATA_ID}.pkl'
    node_feature = 'gae'
    model = HGTVulGraphClassifier(compressed_graph, source_path, feature_extractor=feature_extractor, 
                                 node_feature=node_feature, device=device)
    model.reset_parameters()
    model.to(device)
    X_train, X_val, y_train, y_val = dataset
    model = train(model, X_train, y_train, device)
    save_path = os.path.join(output_models, f'hgt.pth')
    torch.save(model.state_dict(), save_path)
    model.eval()
    with torch.no_grad():
        logits, _ = model(X_val)
        logits = logits.to(device)
        # test_acc, test_micro_f1, test_macro_f1 = score(y_val, logits)
        test_results = get_classification_report(y_val, logits, output_dict=True)
    if os.path.isfile(join(logs, 'test_report.json')):
        with open(join(logs, 'test_report.json'), 'r') as f:
            report = json.load(f)
        report.append(test_results)
    else:
        report = [test_results]
    with open(join(logs, 'test_report.json'), 'w') as f:
        json.dump(report, f, indent=2)


def line(compressed_graph, source_path, dataset, bugtype, device):
    logs = f'{ROOT}/logs/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/line/{bugtype}/clean_{ratio*file_counter[bugtype]}_buggy_curated_{DATA_ID}/'
    if not os.path.exists(logs):
        os.makedirs(logs)
    output_models = f'{ROOT}/models/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/line/{bugtype}/clean_{ratio*file_counter[bugtype]}_buggy_curated_{DATA_ID}/'
    if not os.path.exists(output_models):
        os.makedirs(output_models)
    feature_extractor = f'{ROOT}/ge-sc-data/source_code/gesc_matrices_node_embedding/matrix_line_dim128_of_core_graph_of_{bugtype}_{COMPRESSED_GRAPH}_clean_{file_counter[bugtype]}_{DATA_ID}.pkl'
    node_feature = 'line'
    model = HGTVulGraphClassifier(compressed_graph, source_path, feature_extractor=feature_extractor, 
                                 node_feature=node_feature, device=device)
    model.reset_parameters()
    model.to(device)
    X_train, X_val, y_train, y_val = dataset
    model = train(model, X_train, y_train, device)
    save_path = os.path.join(output_models, f'hgt.pth')
    torch.save(model.state_dict(), save_path)
    model.eval()
    with torch.no_grad():
        logits, _ = model(X_val)
        logits = logits.to(device)
        # test_acc, test_micro_f1, test_macro_f1 = score(y_val, logits)
        test_results = get_classification_report(y_val, logits, output_dict=True)
    if os.path.isfile(join(logs, 'test_report.json')):
        with open(join(logs, 'test_report.json'), 'r') as f:
            report = json.load(f)
        report.append(test_results)
    else:
        report = [test_results]
    with open(join(logs, 'test_report.json'), 'w') as f:
        json.dump(report, f, indent=2)


def node2vec(compressed_graph, source_path, dataset, bugtype, device):
    logs = f'{ROOT}/logs/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/node2vec/{bugtype}/clean_{ratio*file_counter[bugtype]}_buggy_curated_{DATA_ID}/'
    if not os.path.exists(logs):
        os.makedirs(logs)
    output_models = f'{ROOT}/models/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/node2vec/{bugtype}/clean_{ratio*file_counter[bugtype]}_buggy_curated_{DATA_ID}/'
    if not os.path.exists(output_models):
        os.makedirs(output_models)
    feature_extractor = f'{ROOT}/ge-sc-data/source_code/gesc_matrices_node_embedding/matrix_node2vec_dim128_of_core_graph_of_{bugtype}_{COMPRESSED_GRAPH}_clean_{file_counter[bugtype]}_{DATA_ID}.pkl'
    node_feature = 'node2vec'
    model = HGTVulGraphClassifier(compressed_graph, source_path, feature_extractor=feature_extractor, 
                                 node_feature=node_feature, device=device)
    model.reset_parameters()
    model.to(device)
    X_train, X_val, y_train, y_val = dataset
    model = train(model, X_train, y_train, device)
    save_path = os.path.join(output_models, f'hgt.pth')
    torch.save(model.state_dict(), save_path)
    model.eval()
    with torch.no_grad():
        logits, _ = model(X_val)
        logits = logits.to(device)
        # test_acc, test_micro_f1, test_macro_f1 = score(y_val, logits)
        test_results = get_classification_report(y_val, logits, output_dict=True)
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
            compressed_graph = f'{ROOT}/ge-sc-data/source_code/{bugtype}/clean_{ratio*file_counter[bugtype]}_buggy_curated_{DATA_ID}/{COMPRESSED_GRAPH}_compressed_graphs.gpickle'
            nx_graph = nx.read_gpickle(compressed_graph)
            file_name_dict = get_node_id_by_file_name(nx_graph)
            label = f'{ROOT}/ge-sc-data/source_code/{bugtype}/clean_{ratio*file_counter[bugtype]}_buggy_curated_{DATA_ID}/graph_labels.json'
            source_path = f'{ROOT}/ge-sc-data/source_code/{bugtype}/clean_{ratio*file_counter[bugtype]}_buggy_curated_{DATA_ID}'
            with open(label, 'r') as f:
                annotations = json.load(f)
            total_files = [f for f in os.listdir(source_path) if f.endswith('.sol')]
            assert len(total_files) == len(annotations)
            targets = []
            for file in total_files:
                try:
                    target = next(anno['targets'] for anno in annotations if anno['contract_name'] == file)
                except StopIteration:
                    raise f'{file} not found!'
                targets.append(target)
            targets = torch.tensor(targets, device=device)
            assert len(total_files) == len(targets)
            X_train, X_val, y_train, y_val = train_test_split(total_files, targets, train_size=TRAIN_RATE)
            dataset = (tuple(X_train), tuple(X_val), y_train, y_val)
            print('Start training with {}/{} train/val smart contracts'.format(len(X_train), len(X_val)))
            gae_embedded = f'{ROOT}/ge-sc-data/source_code/gesc_matrices_node_embedding/matrix_gae_dim128_of_core_graph_of_{bugtype}_{COMPRESSED_GRAPH}_clean_{file_counter[bugtype]}_{DATA_ID}.pkl'
            line_embedded = f'{ROOT}/ge-sc-data/source_code/gesc_matrices_node_embedding/matrix_line_dim128_of_core_graph_of_{bugtype}_{COMPRESSED_GRAPH}_clean_{file_counter[bugtype]}_{DATA_ID}.pkl'
            node2vec_embedded = f'{ROOT}/ge-sc-data/source_code/gesc_matrices_node_embedding/matrix_node2vec_dim128_of_core_graph_of_{bugtype}_{COMPRESSED_GRAPH}_clean_{file_counter[bugtype]}_{DATA_ID}.pkl'

            # Run experiments
            # Base lines
            base_metapath2vec(compressed_graph, source_path, file_name_dict, dataset, bugtype, device)
            if bugtype not in ['arithmetic', 'front_running', 'reentrancy', 'unchecked_low_level_calls']:
                base_gae(dataset, bugtype, gae_embedded, file_name_dict, device)
            base_line(dataset, bugtype, line_embedded, file_name_dict, device)
            base_node2vec(dataset, bugtype, node2vec_embedded, file_name_dict, device)

            ## Out models
            nodetype(compressed_graph, source_path, dataset, bugtype, device)
            metapath2vec(compressed_graph, source_path, dataset, bugtype, device)
            if bugtype not in ['arithmetic', 'front_running', 'reentrancy', 'unchecked_low_level_calls']:
                gae(compressed_graph, source_path, dataset, bugtype, device)
            line(compressed_graph, source_path, dataset, bugtype, device)
            node2vec(compressed_graph, source_path, dataset, bugtype, device)


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
                report_path = f'{ROOT}/logs/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/{model}/{bugtype}/clean_{ratio*file_counter[bugtype]}_buggy_curated_{DATA_ID}/test_report.json'
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
    #     data.append([model, 'Buggy-F1'] + buggy_f1_row)
    #     data.append([model, 'Macro-F1'] + macro_f1_row)
    # print(tabulate(data, headers=bug_list, tablefmt='orgtbl'))
        print(' ', end=' ')
        print(' \t'.join(buggy_f1_row), end=r'')
        print()
        print(' ', end=' ')
        print(' \t'.join(macro_f1_row), end=r'')
        print()


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if args['result']:
            get_results()
    else:
        main(device)
