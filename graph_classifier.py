from asyncio.unix_events import BaseChildWatcher
import os
from sys import abiflags
import matplotlib

import pandas as pd
import numpy as np
import torch
import lightgbm as lgb
import shap
from tabulate import tabulate
from torch import nn
from torch.nn.functional import cross_entropy
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import KFold
from dgl.dataloading import GraphDataLoader
from torch.utils.tensorboard import SummaryWriter

from sco_models.dataloader import EthIdsDataset
from sco_models.model_hetero import MANDOGraphClassifier
from sco_models.model_hgt import HGTVulGraphClassifier
from sco_models.visualization import visualize_average_k_folds, visualize_k_folds
from sco_models.utils import score, get_classification_report, get_confusion_matrix


def train(args, model, train_loader, optimizer, loss_fcn, epoch):
    model.train()
    total_accucracy =  0
    total_macro_f1 = 0
    total_micro_f1 = 0
    total_loss = 0
    circle_lrs = []
    for idx, (batched_graph, labels) in enumerate(train_loader):
        labels = labels.to(args['device'])
        optimizer.zero_grad()
        logits, _ , _ = model(batched_graph)
        loss = loss_fcn(logits, labels)
        train_acc, train_micro_f1, train_macro_f1 = score(labels, logits)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3)
        optimizer.step()
        total_accucracy += train_acc
        total_micro_f1 += train_micro_f1
        total_macro_f1 += train_macro_f1
        total_loss += loss.item()
        circle_lrs.append(optimizer.param_groups[0]["lr"])
    steps = idx + 1
    return model, total_loss/steps, total_micro_f1/steps, train_macro_f1/steps, total_accucracy/steps, circle_lrs


def validate(args, model, val_loader, loss_fcn):
    model.eval()
    total_loss = 0
    total_macro_f1 = 0
    total_micro_f1 = 0
    total_accucracy =  0
    with torch.no_grad():
        for idx, (batched_graph, labels) in enumerate(val_loader):
            labels = labels.to(args['device'])
            logits, _, _ = model(batched_graph)
            loss = loss_fcn(logits, labels)
            total_loss += loss.item()
            val_acc, val_micro_f1, val_macro_f1 = score(labels, logits)
            total_accucracy += val_acc
            total_micro_f1 += val_micro_f1
            total_macro_f1 += val_macro_f1
    steps = idx + 1
    return total_loss/steps, total_micro_f1/steps, val_macro_f1/steps, total_accucracy/steps


def test(args, model, test_loader):
    model.eval()
    total_macro_f1 = 0
    total_micro_f1 = 0
    total_accucracy =  0
    total_logits = []
    total_target = []
    with torch.no_grad():
        for idx, (batched_graph, labels) in enumerate(test_loader):
            labels = labels.to(args['device'])
            logits, _, _ = model(batched_graph, './forensics/graph_hiddens/reentrancy/creation_last_attention.pt')
            total_logits += logits.tolist()
            total_target += labels.tolist()
            test_acc, test_micro_f1, test_macro_f1 = score(labels, logits)
            total_accucracy += test_acc
            total_micro_f1 += test_micro_f1
            total_macro_f1 += test_macro_f1
    steps = idx + 1
    total_logits = torch.tensor(total_logits)
    total_target = torch.tensor(total_target)
    classification_report = get_classification_report(total_target, total_logits, output_dict=True)
    confusion_report = get_confusion_matrix(total_target, total_logits)
    return total_micro_f1/steps, test_macro_f1/steps, total_accucracy/steps, classification_report, confusion_report


def main(args):
    epochs = args['num_epochs']
    k_folds = args['k_folds']
    device = args['device']
    ethdataset = EthIdsDataset(args['label'])
    kfold = KFold(n_splits=k_folds, shuffle=True)
    train_results = {}
    val_results = {}
    # Get feature extractor
    print('Getting features')
    if args['node_feature'] == 'han':
        feature_extractor = GraphClassifier(args['feature_compressed_graph'], node_feature='nodetype', hidden_size=16, device=args['device'])
        feature_extractor.load_state_dict(torch.load(args['feature_extractor']))
        feature_extractor.to(args['device'])
        feature_extractor.eval()
    else:
        feature_extractor = args['feature_extractor']

    # dataloader =  GraphDataLoader(ethdataset)
    # # for graphs, labels in dataloader:
    # for epoch in range(epochs):
    classification_total_report = {'0': {'precision': [], 'recall': [], 'f1-score': [], 'support': []}, '1': {'precision': [], 'recall': [], 'f1-score': [], 'support': []}, 'macro avg': {'precision': [], 'recall': [], 'f1-score': [], 'support': []}, 'weighted avg': {'precision': [], 'recall': [], 'f1-score': [], 'support': []}}
    confusion_matrix_total_report = []
    # test_ids = [ethdataset.filename_mapping[sc] for sc in os.listdir(args['testset']) if sc.endswith('.sol')]
    test_ids = []
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    test_dataloader = GraphDataLoader(ethdataset, batch_size=args['batch_size'], drop_last=False, sampler=test_subsampler)
    total_train_ids = list(set(list(range(ethdataset.num_graphs))).difference(set(test_ids)))
    assert len(set(test_ids) & set(total_train_ids)) == 0
    for fold, (train_ids, val_ids) in enumerate(kfold.split(total_train_ids)):
        train_ids = [total_train_ids[i] for i in train_ids]
        val_ids = [total_train_ids[i] for i in val_ids]
        assert len(test_ids) + len(train_ids) + len(val_ids) == len(ethdataset)
        assert len(set(test_ids) & set(train_ids)) == 0
        assert len(set(test_ids) & set(val_ids)) == 0
        train_results[fold] = {'loss': [], 'acc': [], 'micro_f1': [], 'macro_f1': [], 'lrs': []}
        val_results[fold] = {'loss': [], 'acc': [], 'micro_f1': [], 'macro_f1': []}
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        train_dataloader = GraphDataLoader(ethdataset,batch_size=args['batch_size'],drop_last=False,sampler=train_subsampler)
        val_dataloader = GraphDataLoader(ethdataset,batch_size=args['batch_size'],drop_last=False,sampler=val_subsampler)
        print('Start training fold {} with {}/{}/{} train/val/test smart contracts'.format(fold, len(train_subsampler), len(val_subsampler), len(test_ids)))
        total_steps = epochs
        model = GraphClassifier(args['compressed_graph'], feature_extractor=feature_extractor, node_feature=args['node_feature'], device=device)
        model.reset_parameters()
        model.to(device)
        loss_fcn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, total_steps=total_steps)
        lrs = []
        for epoch in range(epochs):
            print('Fold {} - Epochs {}'.format(fold, epoch))
            model, train_loss, train_micro_f1, train_macro_f1, train_acc, lrs = train(args, model, train_dataloader, optimizer, loss_fcn, epoch)
            print('Train Loss: {:.4f} | Train Micro f1: {:.4f} | Train Macro f1: {:.4f} | Train Accuracy: {:.4f}'.format(
                    train_loss, train_micro_f1, train_macro_f1, train_acc))
            val_loss, val_micro_f1, val_macro_f1, val_acc = validate(args, model, val_dataloader, loss_fcn)
            print('Val Loss:   {:.4f} | Val Micro f1:   {:.4f} | Val Macro f1:   {:.4f} | Val Accuracy:   {:.4f}'.format(
                    val_loss, val_micro_f1, val_macro_f1, val_acc))
            scheduler.step()
            train_results[fold]['loss'].append(train_loss)
            train_results[fold]['micro_f1'].append(train_micro_f1)
            train_results[fold]['macro_f1'].append(train_macro_f1)
            train_results[fold]['acc'].append(train_acc)
            train_results[fold]['lrs'] += lrs

            val_results[fold]['loss'].append(val_loss)
            val_results[fold]['micro_f1'].append(val_micro_f1)
            val_results[fold]['macro_f1'].append(val_macro_f1)
            val_results[fold]['acc'].append(val_acc)

        _, _, _, classification_report, confusion_report = test(args, model, val_dataloader)
        for category, metrics in classification_total_report.items():
            for metric in metrics.keys():
                classification_total_report[category][metric].append(classification_report[category][metric])

        confusion_matrix_total_report.append(confusion_report)
        print('Saving model fold {}'.format(fold))
        # save_path = os.path.join(args['output_models'], f'han_fold_{fold}.pth')
        bugtype = args['log_dir'].split('/')[-1]
        save_path = os.path.join(args['output_models'], f'{bugtype}_hgt.pth')
        torch.save(model.state_dict(), save_path)
        break
    
    headers = ['precision', 'recall', 'f1-score', 'avg_support']
    classification_tabular_report = []
    for category, metrics in classification_total_report.items():
        row = [category]
        for metric in metrics.keys():
            std = np.std(classification_total_report[category][metric])
            classification_total_report[category][metric] = np.mean(classification_total_report[category][metric])
            row.append(f'{classification_total_report[category][metric]}(#{classification_total_report[category][metric]*std:.2f})')
        classification_tabular_report.append(row)
    print(tabulate(classification_tabular_report, headers=headers))
    print(np.round(np.mean(confusion_matrix_total_report, axis=0)))
    return train_results, val_results


def load_model(model_path):
    model = GraphClassifier()
    model.load_state_dict(torch.load(model_path))
    return model.eval()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('MANDO Graph Classifier')
    parser.add_argument('-m', '--model', type=str, default='hgt',
                    help='Kind of model')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed')
    archive_params = parser.add_argument_group(title='Storage', description='Directories for util results')
    archive_params.add_argument('-ld', '--log-dir', type=str, default='./logs/graph_classification', help='Directory for saving training logs and visualization')
    archive_params.add_argument('--output_models', type=str, default='./models/call_graph', help='Where you want to save your models')
    dataset_params = parser.add_argument_group(title='Dataset', description='Dataset paths')
    dataset_params.add_argument('--compressed_graph', type=str, default='./dataset/call_graph/compressed_graph/compress_call_graphs_no_solidity_calls.gpickle', help='Compressed graphs of dataset which was extracted by graph helper tools')
    dataset_params.add_argument('--dataset', type=str, default='./dataset/aggregate/source_code', help='Dicrectory of all souce code files which were used to extract the compressed graph')
    dataset_params.add_argument('--testset', type=str, default='./dataset/smartbugs/source_code', help='Dicrectory of all souce code files which is a partition of the dataset for testing')
    dataset_params.add_argument('--label', type=str, default='./dataset/aggregate/labels.json', help='Label of sources in source code storage')
    dataset_params.add_argument('--checkpoint', type=str, default='./models/ijcai2020_smartbugs/han_fold_1.pth', help='Checkpoint of trained models')
    node_feature_params = parser.add_argument_group(title='Node feature', description='Define the way to get node features')
    node_feature_params.add_argument('--feature_extractor', type=str, default='./models/metapath2vec_cfg/han_fold_1.pth', help='If "node_feature" is "GAE" or "LINE" or "Node2vec", we need a extracted features from those models')
    node_feature_params.add_argument('--node_feature', type=str, default='metapath2vec', help='Kind of node features we want to use, here is one of "nodetype", "metapath2vec", "han", "gae", "line", "node2vec"')
    train_option_params = parser.add_argument_group(title='Optional configures', description='Advanced options')
    train_option_params.add_argument('--k_folds', type=int, default=5, help='Config for cross validate strategy')
    train_option_params.add_argument('--test', action='store_true', help='Set true if you only want to run test phase')
    train_option_params.add_argument('--non_visualize', action='store_true', help='Wheather you want to visualize the metrics')
    args = parser.parse_args().__dict__

    default_configure = {
    'lr': 0.0005,             # Learning rate
    'num_heads': 8,        # Number of attention heads for node-level attention
    'hidden_units': 8,
    'dropout': 0.6,
    'weight_decay': 0.001,
    'num_epochs': 25,
    'batch_size': 256,
    'patience': 100,
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'
    }
    args.update(default_configure)
    torch.manual_seed(args['seed'])

    if not os.path.exists(args['output_models']):
        os.makedirs(args['output_models'])

    if args['model'] == 'han':
        from sco_models.model_hetero import MANDOGraphClassifier as GraphClassifier
    elif args['model'] == 'hgt':
        print(args['model'])
        from sco_models.model_hgt import HGTVulGraphClassifier as GraphClassifier
    # Training
    if not args['test']:
        print('Training phase')
        train_results, val_results = main(args)
        if not args['non_visualize']:
            print('Visualizing')
            visualize_average_k_folds(args, train_results, val_results)
            # visualize_k_folds(args, train_results, val_results)
    # Testing
    else:
        print('Testing phase')
        ethdataset = EthIdsDataset(args['label'])
        test_dataset = GraphDataLoader(ethdataset)
        # smartbugs_ids = [ethdataset.filename_mapping[sc] for sc in os.listdir(args['testset'])]
        dataloader = GraphDataLoader(ethdataset, batch_size=256, drop_last=False)
        model = GraphClassifier(args['compressed_graph'], feature_extractor=args['feature_extractor'], node_feature=args['node_feature'], device=args['device'])
        model.load_state_dict(torch.load(args['checkpoint']))
        model.to(args['device'])
        model.eval()
        # batch_graph_name = list(model.node_ids_by_filename.keys())
        with torch.no_grad():
            for _, (batch_graph_name, labels) in enumerate(dataloader):
                # continue
                hiddens = model.get_batch_hidden_by_metapath(batch_graph_name)

            # for _, (batched_graph, labels) in enumerate(dataloader):
            #     logits, _, hiddens = model(batched_graph)
        data_frame_ = []
        for g, h in hiddens.items():
            data_frame_.append(h.mean(1).tolist())
        X_train = pd.DataFrame(np.array(data_frame_), columns=[f'{mt[0][0]}_{mt[0][1]}_{mt[0][2]}' for mt in model.meta_paths])
        X_train.insert(0, 'filename', [g for g in hiddens.keys()])
        X_train.to_csv('./graph_attention_test/hiddens_mean.csv', index=False)
        X_train = pd.read_csv('./graph_attention_test/hiddens_mean.csv')
        X_train = X_train.replace(0, -np.Inf)
        X_train.to_csv('./graph_attention_test/hiddens_nan.csv', index=False)
        X_train = X_train.loc[:, X_train.columns != 'filename']
        # print(X_train)
        assert len(X_train) == len(labels)
        d_train = lgb.Dataset(X_train, label=labels.tolist())
        params = {
            "max_bin": 512,
            "learning_rate": 0.05,
            "boosting_type": "gbdt",
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 10,
            "verbose": -1,
            "min_data": 100,
            "boost_from_average": True
        }
        from sklearn.model_selection import train_test_split
        def f(graphs):
            logits, _, _ = model(graphs)
            logits = nn.functional.softmax(logits)
            _, indices = torch.max(logits, dim=1)
            prediction = indices.long().cpu().numpy()
            return prediction
        model = lgb.train(params, d_train, 1000, verbose_eval=10)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], X_train.iloc[0,:], matplotlib=True)
        shap.summary_plot(shap_values, X_train)
        print(shap_values)
        # X_train, X_valid, y_train, y_valid = train_test_split(X_train, labels, test_size=0.2, random_state=7)
        # # build model
        # input_els = []
        # encoded_els = []
        # for k,dtype in dtypes:
        #     input_els.append(Input(shape=(1,)))
        #     if dtype == "int8":
        #         e = Flatten()(Embedding(X_train[k].max()+1, 1)(input_els[-1]))
        #     else:
        #         e = input_els[-1]
        #     encoded_els.append(e)
        # encoded_els = concatenate(encoded_els)
        # layer1 = Dropout(0.5)(Dense(100, activation="relu")(encoded_els))
        # out = Dense(1)(layer1)

        # # train model
        # regression = Model(inputs=input_els, outputs=[out])
        # regression.compile(optimizer="adam", loss='binary_crossentropy')
        # regression.fit(
        #     [X_train[k].values for k,t in dtypes],
        #     y_train,
        #     epochs=50,
        #     batch_size=512,
        #     shuffle=True,
        #     validation_data=([X_valid[k].values for k,t in dtypes], y_valid)
        # )
        # explainer = shap.KernelExplainer(f, X_train.iloc[:50,:])
        # shap_values = explainer.shap_values(X_train.iloc[12,:], nsamples=100)
        # shap.force_plot(explainer.expected_value, shap_values, X_train.iloc[12,:], matplotlib=True)