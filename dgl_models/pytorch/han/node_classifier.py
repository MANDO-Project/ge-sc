"""This model shows an example of using dgl.metapath_reachable_graph on the original heterogeneous
graph.

Because the original HAN implementation only gives the preprocessed homogeneous graph, this model
could not reproduce the result in HAN as they did not provide the preprocessing code, and we
constructed another dataset from ACM with a different set of papers, connections, features and
labels.
"""
import os

import torch
import networkx as nx
from sklearn.model_selection import KFold
from dgl.dataloading import GraphDataLoader

from dataloader import EthIdsDataset
from model_hetero import HANVulClassifier
from model_node_classification import HANVulNodeClassifier
from utils import score
from visualization import visualize_average_k_folds


def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()


def main(args):
    epochs = args['num_epochs']
    k_folds = args['k_folds']
    device = args['device']
    # ethdataset = EthIdsDataset(args['dataset'], args['compressed_graph'], args['label'])
    kfold = KFold(n_splits=k_folds, shuffle=True)
    train_results = {}
    val_results = {}
    # Get feature extractor
    print('Getting features')
    if args['node_feature'] == 'han':
        han_model = HANVulClassifier(args['feature_compressed_graph'], ethdataset.filename_mapping, node_feature='metapath2vec', hidden_size=16, device=args['device'])
        han_model.load_state_dict(torch.load(args['feature_extractor']))
        han_model.to(args['device'])
        han_model.eval()
    else:
        han_model = None

    nx_graph = nx.read_gpickle(args['compressed_graph'])
    number_of_nodes = len(nx_graph)
    for fold, (train_ids, val_ids) in enumerate(kfold.split(range(number_of_nodes))):
        # Init model
        model = HANVulNodeClassifier(args['compressed_graph'], args['dataset'], feature_extractor=han_model, node_feature=args['node_feature'], device=device)
        model.to(device)
        train_results[fold] = {'loss': [], 'acc': [], 'micro_f1': [], 'macro_f1': [], 'lrs': []}
        val_results[fold] = {'loss': [], 'acc': [], 'micro_f1': [], 'macro_f1': []}
        print('Start training fold {} with {}/{} train/val smart contracts'.format(fold, len(train_ids), len(val_ids)))
        total_steps = epochs
        loss_fcn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, total_steps=total_steps)
        train_mask = get_binary_mask(number_of_nodes, train_ids)
        val_mask = get_binary_mask(number_of_nodes, val_ids)
        if hasattr(torch, 'BoolTensor'):
            train_mask = train_mask.bool()
            val_mask = val_mask.bool()

        for epoch in range(epochs):
            print('Fold {} - Epochs {}'.format(fold, epoch))
            optimizer.zero_grad()
            logits, targets = model()
            logits = logits.to(args['device'])
            targets = targets.to(args['device'])
            train_loss = loss_fcn(logits[train_mask], targets[train_mask]) 
            train_acc, train_micro_f1, train_macro_f1 = score(logits[train_mask], targets[train_mask])
            train_loss.backward()
            optimizer.step()
            scheduler.step()
            train_acc, train_micro_f1, train_macro_f1 = score(logits[train_mask], targets[train_mask])
            print('Train Loss: {:.4f} | Train Micro f1: {:.4f} | Train Macro f1: {:.4f} | Train Accuracy: {:.4f}'.format(
                    train_loss.item(), train_micro_f1, train_macro_f1, train_acc))
            val_loss = loss_fcn(logits[val_mask], targets[val_mask]) 
            val_acc, val_micro_f1, val_macro_f1 = score(logits[val_mask], targets[val_mask])
            print('Val Loss:   {:.4f} | Val Micro f1:   {:.4f} | Val Macro f1:   {:.4f} | Val Accuracy:   {:.4f}'.format(
                    val_loss.item(), val_micro_f1, val_macro_f1, val_acc))

            train_results[fold]['loss'].append(train_loss)
            train_results[fold]['micro_f1'].append(train_micro_f1)
            train_results[fold]['macro_f1'].append(train_macro_f1)
            train_results[fold]['acc'].append(train_acc)
            train_results[fold]['lrs'] += scheduler.get_last_lr()

            val_results[fold]['loss'].append(val_loss)
            val_results[fold]['micro_f1'].append(val_micro_f1)
            val_results[fold]['macro_f1'].append(val_macro_f1)
            val_results[fold]['acc'].append(val_acc)

        print('Saving model fold {}'.format(fold))
        save_path = os.path.join(args['output_models'], f'han_fold_{fold}.pth')
        torch.save(model.state_dict(), save_path)
    return train_results, val_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('HAN')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('-ld', '--log-dir', type=str, default='./logs/ijcai2020_smartbugs', help='Dir for saving training results')
    parser.add_argument('--compressed_graph', type=str, default='./dataset/call_graph/compressed_graph/compress_call_graphs_no_solidity_calls.gpickle')
    parser.add_argument('--dataset', type=str, default='./dataset/aggregate/source_code')
    parser.add_argument('--test_compressed_graph', type=str, default='./dataset/smartbugs/compressed_graphs/compress_graphs.gpickle')
    parser.add_argument('--testset', type=str, default='./dataset/smartbugs/source_code')
    parser.add_argument('--label', type=str, default='./dataset/aggregate/labels.json')
    parser.add_argument('--output_models', type=str, default='./models/call_graph_rgcn')
    parser.add_argument('--checkpoint', type=str, default='./models/call_graph_rgcn/han_fold_1.pth')
    parser.add_argument('--feature_compressed_graph', type=str, default='./dataset/aggregate/compressed_graph/compressed_graphs.gpickle')
    parser.add_argument('--feature_extractor', type=str, default='./models/metapath2vec_cfg/han_fold_1.pth')
    parser.add_argument('--node_feature', type=str, default='metapath2vec')
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--non_visualize', action='store_true')
    args = parser.parse_args().__dict__

    default_configure = {
    'lr': 0.0005,             # Learning rate
    'num_heads': 8,        # Number of attention heads for node-level attention
    'hidden_units': 8,
    'dropout': 0.6,
    'weight_decay': 0.001,
    'num_epochs': 100,
    'batch_size': 256,
    'patience': 100,
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'
    }
    args.update(default_configure)
    torch.manual_seed(args['seed'])

    if not os.path.exists(args['output_models']):
        os.makedirs(args['output_models'])

    # Training
    if not args['test']:
        print('Training phase')
        train_results, val_results = main(args)
        if not args['non_visualize']:
            print('Visualizing')
            visualize_average_k_folds(args, train_results, val_results)
    # Testing
    else:
        print('Testing phase')
        ethdataset = EthIdsDataset(args['testset'], args['test_compressed_graph'], args['label'])
        smartbugs_ids = [ethdataset.filename_mapping[sc] for sc in os.listdir(args['testset'])]
        print(smartbugs_ids)
        test_dataloader = GraphDataLoader(ethdataset, batch_size=8, drop_last=False, sampler=smartbugs_ids)
        model = HANVulNodeClassifier(args['test_compressed_graph'], ethdataset.filename_mapping, node_feature=args['node_feature'], hidden_size=16, out_size=2,num_heads=8, dropout=0.6, device=args['device'])
        model.load_state_dict(torch.load(args['checkpoint']))
        model.to(args['device'])
        logits, targets = model()
        logits = logits.to(args['device'])
        targets = targets.to(args['device'])
        test_acc, test_micro_f1, test_macro_f1 = score(logits, targets)
        print('Test Micro f1:   {:.4f} | Test Macro f1:   {:.4f} | Test Accuracy:   {:.4f}'.format(test_micro_f1, test_macro_f1, test_acc))