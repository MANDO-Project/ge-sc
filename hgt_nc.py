import os
from tkinter import N

import torch
import numpy as np
import networkx as nx
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt

from sco_models.model_hgt import HGTVulNodeClassifier
from sco_models.utils import score, get_classification_report, get_confusion_matrix, dump_result
from sco_models.visualization import visualize_average_k_folds
from sco_models.graph_utils import load_hetero_nx_graph, generate_hetero_graph_data, get_node_label, \
                                   get_node_ids_dict, get_number_of_nodes
from process_graphs.byte_code_control_flow_graph_generator import get_solc_version
from process_graphs import control_flow_graph_generator as cfg_engine
from process_graphs import call_graph_generator as cg_engine
from process_graphs import combination_call_graph_and_control_flow_graph_helper as combine_engine


def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()


def get_node_ids(graph, source_files):
    file_ids = []
    for node_ids, node_data in graph.nodes(data=True):
        filename = node_data['source_file']
        if filename in source_files:
            file_ids.append(node_ids)
    return file_ids


def main(args):
    epochs = args['num_epochs']
    k_folds = args['k_folds']
    device = args['device']
    # kfold = KFold(n_splits=k_folds, shuffle=True)
    train_results = {}
    val_results = {}
    # Get feature extractor
    print('Getting features')
    if args['node_feature'] == 'han':
        feature_extractor = HGTVulNodeClassifier(args['feature_compressed_graph'], feature_extractor=args['cfg_feature_extractor'], node_feature='gae', device=args['device'])
        feature_extractor.load_state_dict(torch.load(args['feature_extractor']))
        feature_extractor.to(args['device'])
        feature_extractor.eval()
    else:
        feature_extractor = args['feature_extractor']

    nx_graph = nx.read_gpickle(args['compressed_graph'])
    number_of_nodes = len(nx_graph)
    model = HGTVulNodeClassifier(args['compressed_graph'], feature_extractor=feature_extractor, node_feature=args['node_feature'], device=device)
    total_train_files = [f for f in os.listdir(args['dataset']) if f.endswith('.sol')]
    total_test_files = [f for f in os.listdir(args['testset']) if f.endswith('.sol')]
    total_train_files = list(set(total_train_files).difference(set(total_test_files)))
    clean_smart_contract = './ge-sc-data/smartbugs_wild/clean_50'
    # total_clean_files = [f for f in os.listdir(clean_smart_contract) if f.endswith('.sol')]
    total_clean_files = []
    total_train_files = list(set(total_train_files).difference(set(total_clean_files)))

    # Train valid split data
    train_rate = 0.8
    val_rate = 0.1
    rand_train_ids = torch.randperm(len(total_train_files)).tolist()
    rand_test_ids = torch.randperm(len(total_test_files)).tolist()
    rand_clean_ids = torch.randperm(len(total_clean_files)).tolist()

    train_size_0 = int(train_rate * len(total_train_files))
    train_size_1 = int(train_rate * len(total_test_files))
    train_size_2 = int(train_rate * len(total_clean_files))
    train_files = [total_train_files[i] for i in rand_train_ids[:train_size_0]] + \
                  [total_test_files[i] for i in rand_test_ids[:train_size_1]] + \
                  [total_clean_files[i] for i in rand_clean_ids[:train_size_2]]
    print('Buggy train files: ', [total_train_files[i] for i in rand_train_ids[:train_size_0]])
    print('Curated train files: ', [total_test_files[i] for i in rand_test_ids[:train_size_1]])

    val_size_0 = int(val_rate * len(total_train_files))
    val_size_1 = int(val_rate * len(total_test_files))
    val_size_2 = int(val_rate * len(total_clean_files))
    val_files = [total_train_files[i] for i in rand_train_ids[train_size_0:train_size_0 + val_size_0]] + \
                [total_test_files[i] for i in rand_test_ids[train_size_1:train_size_1 + val_size_1]] + \
                [total_clean_files[i] for i in rand_clean_ids[train_size_2:train_size_2 + val_size_2]]
    print('Buggy valid files: ', [total_train_files[i] for i in rand_train_ids[train_size_0:train_size_0 + val_size_0]])
    print('Curated valid files: ', [total_test_files[i] for i in rand_test_ids[train_size_1:train_size_1 + val_size_1]])
    test_files = [total_train_files[i] for i in rand_train_ids[train_size_0 + val_size_0:]] + \
                 [total_test_files[i] for i in rand_test_ids[train_size_1 + val_size_1:]] + \
                 [total_clean_files[i] for i in rand_clean_ids[train_size_2 + val_size_2:]]
    print('Buggy test files: ', [total_train_files[i] for i in rand_train_ids[train_size_0 + val_size_0:]])
    print('Curated test files: ', [total_test_files[i] for i in rand_test_ids[train_size_1 + val_size_1:]])

    assert len(train_files) + len(val_files) + len(test_files) == len(total_train_files) + len(total_test_files) + len(total_clean_files)

    print('Label dict: ', model.label_ids)
    print(f'Number of source code for Buggy/Curated: {len(total_train_files)}/{len(total_test_files)}')
    total_train_ids = get_node_ids(nx_graph, total_train_files)
    train_ids = get_node_ids(nx_graph, train_files)
    val_ids = get_node_ids(nx_graph, val_files)
    test_ids = get_node_ids(nx_graph, test_files)
    targets = torch.tensor(model.node_labels, device=args['device'])
    assert len(set(train_ids) | set(val_ids) | set(test_ids)) == len(targets)
    buggy_node_ids = torch.nonzero(targets).squeeze().tolist()
    print('Buggy node {}/{} ({}%)'.format(len(set(buggy_node_ids)), len(targets), 100*len(set(buggy_node_ids))/len(targets)))
    # for fold, (train_ids, val_ids) in enumerate(kfold.split(total_train_ids)):
        # Init model 
    fold = 0
    model.reset_parameters()
    model.to(device)
    train_results[fold] = {'loss': [], 'acc': [], 'micro_f1': [], 'macro_f1': [], 'lrs': []}
    val_results[fold] = {'loss': [], 'acc': [], 'micro_f1': [], 'macro_f1': []}
    train_buggy_node_ids = set(buggy_node_ids).intersection(set(train_ids))
    print('Buggy nodes in train: {}/{} ({}%)'.format(len(train_buggy_node_ids), len(train_ids), 100*len(train_buggy_node_ids)/len(train_ids)))
    val_buggy_node_ids = set(buggy_node_ids).intersection(set(val_ids))
    print('Buggy nodes in valid: {}/{} ({}%)'.format(len(val_buggy_node_ids), len(val_ids), 100*len(val_buggy_node_ids)/len(val_ids)))
    test_buggy_node_ids =set(buggy_node_ids).intersection(set(test_ids))
    print('Buggy nodes in test: {}/{} ({}%)'.format(len(test_buggy_node_ids), len(test_ids), 100*len(test_buggy_node_ids)/len(test_ids)))
    print('Start training fold {} with {}/{} train/val smart contracts'.format(fold, len(train_ids), len(val_ids)))
    total_steps = epochs
    # class_counter = [len(labeled_node_ids['valid']), len(labeled_node_ids['buggy'])]
    # class_weight = torch.tensor([1 - sample/len(class_counter) for sample in class_counter], requires_grad=False).to(args['device'])
    # Don't record the following operation in autograd
    # with torch.no_grad():
    #     loss_weights.copy_(initial_weights)
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.005, total_steps=total_steps)
    train_mask = get_binary_mask(number_of_nodes, train_ids)
    val_mask = get_binary_mask(number_of_nodes, val_ids)
    test_mask = get_binary_mask(number_of_nodes, test_ids)
    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()
    retain_graph = True if args['node_feature'] == 'han' else False
    for epoch in range(epochs):
        print('Fold {} - Epochs {}'.format(fold, epoch))
        optimizer.zero_grad()
        logits = model()
        logits = logits.to(args['device'])
        train_loss = loss_fcn(logits[train_mask], targets[train_mask]) 
        train_loss.backward(retain_graph=retain_graph)
        optimizer.step()
        scheduler.step()
        train_acc, train_micro_f1, train_macro_f1 = score(targets[train_mask], logits[train_mask])
        print('Train Loss: {:.4f} | Train Micro f1: {:.4f} | Train Macro f1: {:.4f} | Train Accuracy: {:.4f}'.format(
                train_loss.item(), train_micro_f1, train_macro_f1, train_acc))
        val_loss = loss_fcn(logits[val_mask], targets[val_mask]) 
        val_acc, val_micro_f1, val_macro_f1 = score(targets[val_mask], logits[val_mask])
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
    # dump_result(targets[val_mask], logits[val_mask], os.path.join(args['output_models'], f'confusion_{fold}.csv'))
    bugtype = args['log_dir'].split('/')[-1]
    save_path = os.path.join(args['output_models'], f'{bugtype}_hgt.pth')
    torch.save(model.state_dict(), save_path)
    print('Testing phase')
    print(f'Testing on {len(test_ids)} nodes')
    model.eval()
    with torch.no_grad():
        logits = model()
        logits = logits.to(args['device'])
        test_acc, test_micro_f1, test_macro_f1 = score(targets[test_mask], logits[test_mask])
        print('Test Micro f1:   {:.4f} | Test Macro f1:   {:.4f} | Test Accuracy:   {:.4f}'.format(test_micro_f1, test_macro_f1, test_acc))
        print('Classification report', '\n', get_classification_report(targets[test_mask], logits[test_mask]))
        print('Confusion matrix', '\n', get_confusion_matrix(targets[test_mask], logits[test_mask]))
    return train_results, val_results


def draw_graph(graph, edge_attention, cm='Oranges', layout='shell', node_size=50, edgewidth_=0.5):
    """
    Draw a graph with node features and labels
    """
    edgecolors = [0.3, 0.3, 0.3, 0.1]
    labels = {}
    for idx, n_data in graph.nodes(data=True):
        labels[idx] = n_data['node_type'][:5]
    if layout == 'spring':
        pos = nx.spring_layout(graph)
    elif layout == 'shell':
        pos = nx.shell_layout(graph)
    else:
        raise NotImplementedError(layout)
    for i in range(1):
    # First ======================================================================================
        # plt.subplot(int(f'42{i+1}'))
        nx.draw_networkx_labels(graph, pos, labels, font_size=4, font_color="black", font_weight='bold')
        edge_att_ = [edge_attention[1][_idx][0] for _idx in range(len(graph.edges))]
        edge_att_ = np.arange(len(graph.edges)) / (float(np.sum(np.arange(len(graph.edges)))) + 1e-7)
        cmap = plt.cm.get_cmap(cm, len(edge_att_))
        vmin = edge_att_.min()
        vmax = edge_att_.max()
        nx.draw_networkx(graph, pos, edge_color=edge_att_, with_labels=False, node_size=node_size,
                        width=edgewidth_, edge_cmap=cmap, vmin=vmin, vmax=vmax)
        plt.title(f'Attention score header {i+1}th', fontsize=5)
        
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin, vmax=vmax))
    sm._A = []
    ticks = np.linspace(0, edge_att_.max(), 5)
    cbar = plt.colorbar(sm, ticks=ticks, format='%.2f', fraction=0.02, pad=0.04) 
    cbar.ax.tick_params(labelsize=14)
    plt.axis('off')
    plt.savefig('./graph_attention_test/att_header_1_layer_2.png', dpi=800)
    # plt.show()


def get_node_types(graph):
    node_types = []
    for i, data in graph.nodes(data=True):
        node_types.append(data['node_type'])
    return list(set(node_types))


def get_edge_types(graph):
    edge_types = []
    for source, target, data in graph.edges(data=True):
        edge_type = data['edge_type']
        source_node_type = graph.nodes[source]['node_type']
        target_node_type = graph.nodes[target]['node_type']
        three_cannonical_egde = (source_node_type, edge_type, target_node_type)
        edge_types.append(three_cannonical_egde)
    return list(set(edge_types))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('HAN')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed')
    archive_params = parser.add_argument_group(title='Storage', description='Directories for util results')
    archive_params.add_argument('-ld', '--log_dir', type=str, default='./logs', help='Directory for saving training logs and visualization')
    archive_params.add_argument('--output_models', type=str, default='./models/call_graph_rgcn',
                        help='Where you want to save your models')

    dataset_params = parser.add_argument_group(title='Dataset', description='Dataset paths')
    dataset_params.add_argument('--compressed_graph', type=str, default='./dataset/call_graph/compressed_graph/compress_call_graphs_no_solidity_calls.gpickle',
                        help='Compressed graphs of dataset which was extracted by graph helper tools')
    dataset_params.add_argument('--dataset', type=str, default='./dataset/aggregate/source_code',
                        help='Dicrectory of all souce code files which were used to extract the compressed graph')
    dataset_params.add_argument('--testset', type=str, default='./dataset/smartbugs/source_code',
                        help='Dicrectory of all souce code files which is a partition of the dataset for testing')
    dataset_params.add_argument('--label', type=str, default='./dataset/aggregate/labels.json')
    
    node_feature_params = parser.add_argument_group(title='Node feature', description='Define the way to get node features')
    node_feature_params.add_argument('--feature_compressed_graph', type=str, default='./dataset/aggregate/compressed_graph/compressed_graphs.gpickle',
                        help='If "node_feature" is han, you mean use 2 HAN layers. The first one is HAN of CFGs as feature node for the second HAN of call graph, This is the compressed graphs were trained for the first HAN')
    node_feature_params.add_argument('--cfg_feature_extractor', type=str, default='./models/metapath2vec_cfg/han_fold_1.pth',
                        help='If "node_feature" is han, feature_extractor is a checkpoint of the first HAN layer')
    node_feature_params.add_argument('--feature_extractor', type=str, default='./models/metapath2vec_cfg/han_fold_1.pth',
                        help='If "node_feature" is "GAE" or "LINE" or "Node2vec", we need a extracted features from those models')
    node_feature_params.add_argument('--node_feature', type=str, default='nodetype',
                        help='Kind of node features we want to use, here is one of "nodetype", "metapath2vec", "han", "gae", "line", "node2vec"')
    
    train_option_params = parser.add_argument_group(title='Optional configures', description='Advanced options')
    train_option_params.add_argument('--k_folds', type=int, default=1, help='Config for cross validate strategy')
    train_option_params.add_argument('--test', action='store_true', help='Set true if you only want to run test phase')
    train_option_params.add_argument('--non_visualize', action='store_true',
                        help='Wheather you want to visualize the metrics')

    extra_test_params = parser.add_argument_group(title='Testing optional', description='Add more comprehensive testing')
    extra_test_params.add_argument('--extra_test', action='store_true', help='Set true if want to extra test')
    args = parser.parse_args().__dict__

    default_configure = {
    'lr': 0.0005,             # Learning rate
    'num_heads': 8,        # Number of attention heads for node-level attention
    'hidden_units': 8,
    'dropout': 0.6,
    'weight_decay': 0.001,
    'num_epochs': 30,
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
        nx_graph = nx.read_gpickle(args['compressed_graph'])
        number_of_nodes = len(nx_graph)
        test_files = [f for f in os.listdir(args['testset']) if f.endswith('.sol')]
        model = HGTVulNodeClassifier(args['compressed_graph'], feature_extractor=None, node_feature=args['node_feature'], device=args['device'])
        bugtype = args['log_dir'].split('/')[-1]
        model.load_state_dict(torch.load(os.path.join(args['output_models'], f'{bugtype}_hgt.pth')))
        model.eval()
        model.to(args['device'])
        test_ids = get_node_ids(nx_graph, test_files)
        targets = torch.tensor(model.node_labels, device=args['device'])
        buggy_node_ids = torch.nonzero(targets).squeeze().tolist()
        test_buggy_node_ids = set(buggy_node_ids) & set(test_ids)
        # print('Buggy nodes in test: {}/{} ({}%)'.format(len(test_buggy_node_ids), len(test_ids), 100*len(test_buggy_node_ids)/len(test_ids)))
        test_mask = get_binary_mask(number_of_nodes, test_ids)
        if hasattr(torch, 'BoolTensor'):
            test_mask = test_mask.bool()
        # print(f"Testing on {len(test_ids)} nodes")
        # print('Node type: ', model.ntypes_dict)
        if args['extra_test']:
            list_vulnerabilities_json_files = ['data/solidifi_buggy_contracts/access_control/vulnerabilities.json',
                'data/smartbug-dataset/vulnerabilities.json']
            extra_graph_path = [f'/Users/minh/Documents/2022/smart_contract/mando/ge-sc/experiments/ge-sc-data/source_code/access_control/buggy/buggy_20.sol']
            extra_cfg_output = f'/Users/minh/Documents/2022/smart_contract/mando/ge-sc/experiments/ge-sc-data/source_code/access_control/buggy/cfg_buggy_20.gpickle'
            extra_cg_output = f'/Users/minh/Documents/2022/smart_contract/mando/ge-sc/experiments/ge-sc-data/source_code/access_control/buggy/cg_buggy_20.gpickle'
            annotation_vulnerabilities = cfg_engine.merge_data_from_vulnerabilities_json_files(list_vulnerabilities_json_files)
            cfg_engine.compress_full_smart_contracts(extra_graph_path, None, extra_cfg_output, vulnerabilities=annotation_vulnerabilities)
            cg_engine.compress_full_smart_contracts(extra_graph_path, extra_cg_output, vulnerabilities=annotation_vulnerabilities)

            input_cfg = nx.read_gpickle(extra_cfg_output)
            print('cfg graph: ', len(input_cfg))
            input_call_graph = nx.read_gpickle(extra_cg_output)
            print('cg graph: ', len(input_call_graph))
            dict_node_label_cfg_and_cg = combine_engine.mapping_cfg_and_cg_node_labels(input_cfg, input_call_graph)
            merged_graph = combine_engine.add_new_cfg_edges_from_call_graph(input_cfg, dict_node_label_cfg_and_cg, input_call_graph)
            extra_cfg_cg_output = f'/Users/minh/Documents/2022/smart_contract/mando/ge-sc/experiments/ge-sc-data/source_code/access_control/buggy/cfg_cg_buggy_20.gpickle'
            combine_engine.update_cfg_node_types_by_call_graph_node_types(merged_graph, dict_node_label_cfg_and_cg)
            nx.write_gpickle(merged_graph, extra_cfg_cg_output)
            print('cfg + cg graph: ', len(merged_graph))
            bug_count = 0
            for i, n in merged_graph.nodes(data=True):
                if n['node_info_vulnerabilities'] is not None:
                    bug_count += 1
            print('Bug percentage: ', 100 * bug_count/len(merged_graph))
            extra_graph = nx.disjoint_union(model.nx_graph, merged_graph)
            print('Length of new graph: ', len(extra_graph))

            number_of_nodes = len(extra_graph)
            test_ids = get_node_ids(extra_graph, ['0x23a91059fdc9579a9fbd0edc5f2ea0bfdb70deb4.sol'])
            test_mask = get_binary_mask(number_of_nodes, test_ids)
            if hasattr(torch, 'BoolTensor'):
                test_mask = test_mask.bool()
            print(f"Testing on {len(test_ids)} nodes")
            print('Node type: ', model.ntypes_dict)

        with torch.no_grad():
            logits, node_labels, attention_layer = model.extend_forward(extra_graph)
            edge_attr = set()
            node_attr = set()
            symmetrical_global_graph = model.symmetrical_global_graph
            # print(attention_layer)
            # print(graph.ndata.keys())
            node_types = get_node_types(merged_graph)
            edge_types = get_edge_types(merged_graph)
            print(node_types)
            print(edge_types)
            att_score = {0: [], 1: []}
            for idx, edge in enumerate(edge_types):
                print(edge)
                # print(attention_layer[0][edge], attention_layer[0][edge].shape)
                # print(symmetrical_global_graph.edges(etype=edge), symmetrical_global_graph.edges(etype=edge)[0].shape)
                source_ = symmetrical_global_graph.edges(etype=edge)[0]
                for n_id in test_ids:
                    if n_id in symmetrical_global_graph.edges(etype=edge)[0]:
                        att_idx = (source_ == n_id).nonzero(as_tuple=True)[0].tolist()
                        print('attent shape: ', attention_layer[0][edge][att_idx].shape)
                        att_score[0] += attention_layer[0][edge][att_idx].tolist()
                        att_score[1] += attention_layer[1][edge][att_idx].tolist()
            print(len(att_score[0]))
            print(len(att_score[1]))

            # for n_id in symmetrical_global_graph.nodes:
            #     if n_id in test_ids:
            #         print(symmetrical_global_graph.nodes[i])
            # print(merged_graph.edges)
            for source, dest, e_data in merged_graph.edges(data=True):
                # print(e_data)
                canonical_edge = (merged_graph.nodes[source]['node_type'], e_data['edge_type'], merged_graph.nodes[dest]['node_type'])
                # print('attention score ===: ', att_score[0][canonical_edge])


            # print(len(merged_graph.edges))
            # draw_graph(merged_graph, att_score)
            # for src, dst, data in graph.edges(data=True):
            #     edge_attr.update(data.keys())
            # for n, data in graph.nodes(data=True):
            #     node_attr.update(data.keys())
            # print('Edge attributes: ', edge_attr)
            # print('Node attributes: ', node_attr)
                # data['edge_type'] = model.edge_types_dict[data['edge_type']]
            # targets = torch.tensor(node_labels, device=args['device'])
            # logits = logits.to(args['device'])
            # # print(torch.nonzero(targets, as_tuple=True)[0].shape)
            # test_acc, test_micro_f1, test_macro_f1 = score(targets[test_mask], logits[test_mask])
            # print('Test Micro f1:   {:.4f} | Test Macro f1:   {:.4f} | Test Accuracy:   {:.4f}'.format(test_micro_f1, test_macro_f1, test_acc))
            # print('Classification report', '\n', get_classification_report(targets[test_mask], logits[test_mask]))
            # print('Confusion matrix', '\n', get_confusion_matrix(targets[test_mask], logits[test_mask]))
            