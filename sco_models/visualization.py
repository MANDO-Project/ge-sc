import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter


def visualize_average_k_folds(args, train_results, val_results):
    avg_train_result = {}
    avg_train_result['acc'] = torch.mean(torch.tensor([train_results[fold]['acc'] for fold in range(args['k_folds'])]), dim=0).tolist()
    avg_train_result['micro_f1'] = torch.mean(torch.tensor([train_results[fold]['micro_f1'] for fold in range(args['k_folds'])]), dim=0).tolist()
    avg_train_result['macro_f1'] = torch.mean(torch.tensor([train_results[fold]['macro_f1'] for fold in range(args['k_folds'])]), dim=0).tolist()
    avg_train_result['buggy_f1'] = torch.mean(torch.tensor([train_results[fold]['buggy_f1'] for fold in range(args['k_folds'])]), dim=0).tolist()
    avg_train_result['loss'] = torch.mean(torch.tensor([train_results[fold]['loss'] for fold in range(args['k_folds'])]), dim=0).tolist()
    avg_train_result['lrs'] = torch.mean(torch.tensor([train_results[fold]['lrs'] for fold in range(args['k_folds'])]), dim=0).tolist()
    avg_train_result['lrs'] = train_results[0]['lrs']
    
    avg_val_result = {}
    avg_val_result['acc'] = torch.mean(torch.tensor([val_results[fold]['acc'] for fold in range(args['k_folds'])]), dim=0).tolist()
    avg_val_result['micro_f1'] = torch.mean(torch.tensor([val_results[fold]['micro_f1'] for fold in range(args['k_folds'])]), dim=0).tolist()
    avg_val_result['macro_f1'] = torch.mean(torch.tensor([val_results[fold]['macro_f1'] for fold in range(args['k_folds'])]), dim=0).tolist()
    avg_val_result['buggy_f1'] = torch.mean(torch.tensor([val_results[fold]['buggy_f1'] for fold in range(args['k_folds'])]), dim=0).tolist()
    avg_val_result['loss'] = torch.mean(torch.tensor([val_results[fold]['loss'] for fold in range(args['k_folds'])]), dim=0).tolist()
    writer = SummaryWriter(args['log_dir'])
    for idx in range(args['num_epochs']):
        writer.add_scalars('Accuracy', {f'train_avg': avg_train_result['acc'][idx],
                                        f'valid_avg': avg_val_result['acc'][idx]}, idx)
        writer.add_scalars('Micro_f1', {f'train_avg': avg_train_result['micro_f1'][idx],
                                    f'valid_avg': avg_val_result['micro_f1'][idx]}, idx)
        writer.add_scalars('Macro_f1', {f'train_avg': avg_train_result['macro_f1'][idx],
                                    f'valid_avg': avg_val_result['macro_f1'][idx]}, idx)
        writer.add_scalars('Buggy_f1', {f'train_avg': avg_train_result['buggy_f1'][idx],
                                    f'valid_avg': avg_val_result['buggy_f1'][idx]}, idx)
        writer.add_scalars('Train_results', {f'buggy_f1': avg_train_result['buggy_f1'][idx],
                                    f'macro_f1': avg_train_result['macro_f1'][idx]}, idx)
        writer.add_scalars('Val_results', {f'buggy_f1': avg_val_result['buggy_f1'][idx],
                                    f'macro_f1': avg_val_result['macro_f1'][idx]}, idx)
        writer.add_scalars('Loss', {f'train_avg': avg_train_result['loss'][idx],
                                    f'valid_avg': avg_val_result['loss'][idx]}, idx)
    for idx, lr in enumerate(avg_train_result['lrs']):
        writer.add_scalar('Learning rate', lr, idx)
    writer.close()
    return avg_train_result, avg_val_result


def visualize_k_folds(args, train_results, val_results):
    writer = SummaryWriter(args['log_dir'])
    for fold in range(args['k_folds']):
        for idx in range(args['num_epochs']):
            writer.add_scalars('Accuracy', {f'train_{fold+1}': train_results[fold]['acc'][idx],
                                            f'valid_{fold+1}': val_results[fold]['acc'][idx]}, idx)
            writer.add_scalars('Micro_f1', {f'train_{fold+1}': train_results[fold]['micro_f1'][idx],
                                        f'valid_{fold+1}': val_results[fold]['micro_f1'][idx]}, idx)
            writer.add_scalars('Macro_f1', {f'train_{fold+1}': train_results[fold]['macro_f1'][idx],
                                        f'valid_{fold+1}': val_results[fold]['macro_f1'][idx]}, idx)
            writer.add_scalars('Train_results', {f'buggy_f1_{fold+1}': train_results[fold]['buggy_f1'][idx],
                                                 f'macro_f1_{fold+1}': train_results[fold]['macro_f1'][idx]}, idx)
            writer.add_scalars('Val_results', {f'buggy_f1_{fold+1}': val_results[fold]['buggy_f1'][idx],
                                               f'macro_f1_{fold+1}': val_results[fold]['macro_f1'][idx]}, idx)
            writer.add_scalars('Loss', {f'train_{fold+1}': train_results[fold]['loss'][idx],
                                        f'valid_{fold+1}': val_results[fold]['loss'][idx]}, idx)
    for idx, lr in enumerate(train_results[0]['lrs']):
        writer.add_scalar('Learning rate', lr, idx)
    writer.close()


def nodes_edges_correlation(graph_file_list, axis_plt, title):
    for graph_files in graph_file_list:
        fig, ax = plt.subplots()
        nodes = {}
        edges = {}
        for graph in graph_files:
            nx_graph = nx.read_gpickle(graph)
            nodes[graph] = len(nx_graph.nodes)
            edges[graph] = len(nx_graph.edges)
        axis_plt.scatter(list(nodes.values()), list(edges.values()), c=[np.random.rand(3,)], alpha=0.6, s=10)
    axis_plt.set_xlabel('Number of nodes', fontsize=8)
    axis_plt.set_ylabel('Number of edge', fontsize=8)
    axis_plt.set_title(title, fontsize=8)


def nodes_edges_compressed_graph_correlation(compressed_graphs, axis_plt, title):
    nx_graph = nx.read_gpickle(compressed_graphs)
    node_dict = {}
    edge_dict = {}
    for idx, node in nx_graph.nodes(data=True):
        file_name = node['source_file']
        if file_name in node_dict:
            node_dict[file_name] += 1
        else:
            node_dict[file_name] = 1
    for source, target, data in nx_graph.edges(data=True):
        assert nx_graph.nodes[source]['source_file'] == nx_graph.nodes[target]['source_file']
        file_name = nx_graph.nodes[source]['source_file']
        if file_name in edge_dict:
            edge_dict[file_name] += 1
        else:
            edge_dict[file_name] = 1
    assert len(node_dict) >= len(edge_dict)
    if len(node_dict) > len(edge_dict):
        for k in node_dict.keys():
            if k not in edge_dict:
                edge_dict[k] = 0
    
    # ax.set_xlim(0,3000)
    # ax.set_ylim(0,3000)
    axis_plt.scatter(list(node_dict.values()), list(edge_dict.values()), c=[np.random.rand(3,)], alpha=0.6, s=10)
    axis_plt.set_xlabel('Number of nodes', fontsize=8)
    axis_plt.set_ylabel('Number of edge', fontsize=8)
    axis_plt.set_title(title, fontsize=8)
    axis_plt.grid(True)
    # fig.tight_layout()
    # plt.savefig(output)
