from curses import BUTTON1_TRIPLE_CLICKED
import os
from os.path import join

import json
import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
# import pygraphviz as pgv
from matplotlib.collections import PolyCollection
from matplotlib import cm
from scipy.stats import poisson
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

from sco_models.visualization import nodes_edges_correlation
from sco_models.visualization import nodes_edges_compressed_graph_correlation
from sco_models.model_hgt import HGTVulGraphClassifier

def polygon_under_graph(x, y):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (x, y) line graph. This assumes x is in ascending order.
    """
    # print([(x[0], 0.), *zip(x, y), (x[-1], 0.)])
    return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]


def plot_graph(nxg):
    model = HGTVulGraphClassifier(nxg, node_feature='nodetype', hidden_size=128, num_layers=2,num_heads=8, use_norm=True, device='cpu')
    graph = model.symmetrical_global_graph
    ag = pgv.AGraph(strict=True, directed=False)
    for u, v, k in graph.canonical_etypes:
        ag.add_edge(u, v, label=k)
    ag.layout('dot')
    ag.draw('graph.png')




if __name__ == '__main__':
    ## Compressed graph forensic
    bug_type = {'access_control': 57, 'arithmetic': 60, 'denial_of_service': 46,
              'front_running': 44, 'reentrancy': 71, 'time_manipulation': 50, 
              'unchecked_low_level_calls': 95}
    # bug_type = {'access_control': 57}

    # nxg = './experiments/ge-sc-data/source_code/access_control/clean_57_buggy_curated_0/cfg_cg_compressed_graphs.gpickle'
    # plot_graph(nxg)

    # creation_path = '/home/minhnn/minhnn/ICSE/EtherSolve_ICPC2021_ReplicationPackage/data/reentrancy/creation/'
    # output = '/home/minhnn/minhnn/ICSE/ge-sc/forensics'
    # runtime_path = '/home/minhnn/minhnn/ICSE/EtherSolve_ICPC2021_ReplicationPackage/data/reentrancy/runtime/'
    # creation_files = [join(creation_path, f) for f in os.listdir(creation_path) if f.endswith('.sol')]
    # runtime_files = [join(runtime_path, f) for f in os.listdir(runtime_path) if f.endswith('.sol')]
    # nodes_edges_correlation([creation_files], join(output, 'creation_node_edge_correlation.png'))
    # nodes_edges_correlation([runtime_files], join(output, 'runtime_node_edge_correlation.png'))

    # for bug, count in bug_type.items():
    #     fig, axs = plt.subplots(2, 3, figsize=(10,10), sharex=True, sharey=True)
    #     compressed_graph = f'./experiments/ge-sc-data/source_code/{bug}/clean_{count}_buggy_curated_0/cfg_cg_compressed_graphs.gpickle'
    #     nodes_edges_compressed_graph_correlation(compressed_graph, axs[0, 0], title='CFG + CG')
    #     compressed_graph = f'./experiments/ge-sc-data/source_code/{bug}/clean_{count}_buggy_curated_0/cfg_compressed_graphs.gpickle'
    #     nodes_edges_compressed_graph_correlation(compressed_graph, axs[0, 1], title='CFG')
    #     compressed_graph = f'./experiments/ge-sc-data/source_code/{bug}/clean_{count}_buggy_curated_0/cg_compressed_graphs.gpickle'
    #     nodes_edges_compressed_graph_correlation(compressed_graph, axs[0, 2], title='CG')
    #     compressed_graph = f'./experiments/ge-sc-data/byte_code/smartbugs/creation/gpickles/{bug}/clean_{count}_buggy_curated_0/compressed_graphs/cfg_compressed_graphs.gpickle'
    #     nodes_edges_compressed_graph_correlation(compressed_graph, axs[1, 0], title='CFG creation')
    #     compressed_graph = f'./experiments/ge-sc-data/byte_code/smartbugs/runtime/gpickles/{bug}/clean_{count}_buggy_curated_0/compressed_graphs/cfg_compressed_graphs.gpickle'
    #     nodes_edges_compressed_graph_correlation(compressed_graph, axs[1, 1], title='CFG runtime')

    #     output = f'./forensics/{bug}_nodes_edges_correlation.png'
    #     fig.tight_layout()
    #     plt.savefig(output)

    # # Move graphs
    # from shutil import copy
    # bug_type = {'access_control': 57, 'arithmetic': 60, 'denial_of_service': 46,
    #           'front_running': 44, 'reentrancy': 71, 'time_manipulation': 50, 
    #           'unchecked_low_level_calls': 95}
    # for bug, count in bug_type.items():
    #     creation_source = f'./experiments/ge-sc-data/byte_code/smartbugs/creation/gpickles/{bug}/clean_{count}_buggy_curated_0/compressed_graphs/cfg_compressed_graphs.gpickle'
    #     creation_target = f'./experiments/ge-sc-data/byte_code/smartbugs/creation/gpickles/compressed_graphs/{bug}_cfg_compressed_graphs.gpickle'
    #     runtime_source = f'./experiments/ge-sc-data/byte_code/smartbugs/runtime/gpickles/{bug}/clean_{count}_buggy_curated_0/compressed_graphs/cfg_compressed_graphs.gpickle'
    #     runtime_target = f'./experiments/ge-sc-data/byte_code/smartbugs/runtime/gpickles/compressed_graphs/{bug}_cfg_compressed_graphs.gpickle'
    #     copy(creation_source, creation_target)
    #     copy(runtime_source, runtime_target)

## Generate historgram
    # last_hidden = '/home/minhnn/minhnn/ICSE/ge-sc/forensics/graph_hiddens/reentrancy/last_attention.pt'
    # hiddens = torch.load(last_hidden)
    # fig, axes = plt.subplots(nrows=2, ncols=2)
    # n_bins = np.linspace(torch.min(hiddens).item(), torch.max(hiddens).item(), num=200)
    # axes[0, 0].hist(hiddens, n_bins, histtype='step', fill=False)
    # axes[0, 0].set_title('stack step (unfilled)')
    # fig.tight_layout()
    # plt.show()
    # ax = plt.figure().add_subplot(projection='3d')
    # x = np.linspace(torch.min(hiddens).item(), torch.max(hiddens).item(), num=200)
    # lambdas = range(1, hiddens.shape[0]+1)
    # # verts[i] is a list of (x, y) pairs defining polygon i.
    # verts = [polygon_under_graph(x, poisson.pmf(l, x)) for l in lambdas]
    # facecolors = plt.colormaps['viridis_r'](np.linspace(0, 1, len(verts)))
    # poly = PolyCollection(verts, facecolors=facecolors, alpha=.7)
    # ax.add_collection3d(poly, zs=lambdas, zdir='y')
    # ax.set(xlim=(-3, 3), ylim=(0, 40), zlim=(0, 1500),
    #        xlabel='x', ylabel=r'$\lambda$', zlabel='probability')
    # plt.show()

## 3D last hidden visual
    # ax = plt.figure().add_subplot(projection='3d')

    # x = np.linspace(torch.min(hiddens).item(), torch.max(hiddens).item(), num=200)
    # hiddens_hist = [np.histogram(h.numpy(), bins=x, range=None, normed=None, weights=None, density=None)[0].tolist() for h in hiddens]
    # # print([np.sum(hid * np.diff(x)) for hid in hiddens_hist])
    # print(max(hiddens_hist[0]))
    # lambdas = lambdas = range(1, hiddens.shape[0]+1)
    # # verts[i] is a list of (x, y) pairs defining polygon i.
    # verts = [polygon_under_graph(x, y) for y in hiddens_hist]
    # # print(verts)
    # # facecolors = plt.colormaps()[0](np.linspace(0, 1, len(verts)))
    # viridis = cm.get_cmap('viridis', 8)
    # facecolors = viridis(np.linspace(0, 1, len(verts)))
    
    # poly = PolyCollection(verts, facecolors=facecolors, alpha=.6)
    # ax.add_collection3d(poly, zs=lambdas, zdir='y')

    # ax.set(xlim=(torch.min(hiddens).item(), torch.max(hiddens).item()), ylim=(0, 30), zlim=(0, 7),
    #        xlabel='x', ylabel='y', zlabel='counter')
    # plt.show()

## PCA last hidden state
    # plt.figure(figsize=(20, 18), dpi=80)
    # models = ['metapath2vec', 'nodetype', 'line', 'node2vec', 'random_32', 'random_64', 'random_128', 'zeros_32', 'zeros_64', 'zeros_128']
    # for idx, (bug, count) in enumerate(bug_type.items()):
    #     fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20,13), dpi=100)
    #     for model_idx, model in enumerate(models):
    #         ax = axes[int(model_idx/4), int(model_idx%4)]
    #         last_hidden = f'./experiments/logs/graph_classification/byte_code/smartbugs/runtime/han/cfg/{model}/{bug}/last_hiddens.json'
    #         with open(last_hidden, 'r') as f:
    #             content = json.load(f)
    #         hiddens_0 = [ann['hiddens'] for ann in content if ann['targets'] == 0]
    #         targets_0 = [ann['targets'] for ann in content if ann['targets'] == 0]
    #         contract_names_0 = [ann['contract_name'] for ann in content if ann['targets'] == 0]
    #         hiddens_1 = [ann['hiddens'] for ann in content if ann['targets'] == 1]
    #         targets_1 = [ann['targets'] for ann in content if ann['targets'] == 1]
    #         contract_names_1 = [ann['contract_name'] for ann in content if ann['targets'] == 1]
    #         assert len(hiddens_0) + len(hiddens_1) == len(content)
    #         print(len(hiddens_0)/(len(content)))

    #         # fig = plt.figure()
    #         # ax = fig.add_subplot(projection='3d')
    #         # ax = fig.add_subplot()
    #         # fig = plt.figure(1, figsize=(8, 6))
    #         # ax = Axes3D(fig, elev=-150, azim=110)
    #         # x_3dims = PCA(n_components=3).fit_transform(hiddens)
    #         hidden_0_2dims = PCA(n_components=2).fit_transform(np.array(hiddens_0))
    #         hidden_1_2dims = PCA(n_components=2).fit_transform(np.array(hiddens_1))
    #         # ax.scatter(x_3dims[:, 0], x_3dims[:, 1], x_3dims[:, 2], marker='o')
    #         ax.scatter(hidden_0_2dims[:, 0], hidden_0_2dims[:, 1], marker='o', label='normal')
    #         ax.scatter(hidden_1_2dims[:, 0], hidden_1_2dims[:, 1], marker='^', label='buggy')
    #         ax.set_title(f'{model}', fontsize=15)
    #         # ax.set_zlabel('Z Label')
    #     forensic_path = f'./forensics/last_hiddens/runtime/han/{bug}_last_hiddent.png'
    #     axes[0, 0].legend(ncol=1, bbox_to_anchor=(0, 1), loc='lower center', fontsize=15)
    #     plt.savefig(forensic_path)

    # ax.scatter(
    #     X_reduced[:, 0],
    #     X_reduced[:, 1],
    #     X_reduced[:, 2],
    #     c=y,
    #     cmap=plt.cm.Set1,
    #     edgecolor="k",
    #     s=40,
    # )
    # ax.set_title("First three PCA directions")
    # ax.set_xlabel("1st eigenvector")
    # ax.w_xaxis.set_ticklabels([])
    # ax.set_ylabel("2nd eigenvector")
    # ax.w_yaxis.set_ticklabels([])
    # ax.set_zlabel("3rd eigenvector")
    # ax.w_zaxis.set_ticklabels([])
    # plt.show()

## Dataset Satatistics
    # fig, axes = plt.subplots(nrows=7, ncols=1)
    # WIDTH = 0.9
    # for idx, (bug, count) in enumerate(bug_type.items()):
    #     contract_category_path = f'./experiments/ge-sc-data/source_code/{bug}/clean_{count}_buggy_curated_0/source_code_category.json'
    #     with open(contract_category_path, 'r') as f:
    #         contract_category = json.load(f)
    #     # annotation_path = f'./experiments/ge-sc-data/source_code/{bug}/clean_{count}_buggy_curated_0/contract_labels.json'
    #     annotation_path = f'./experiments/ge-sc-data/byte_code/smartbugs/contract_labels/{bug}/runtime_balanced_contract_labels.json'
    #     with open(annotation_path, 'r') as f:
    #         annotations = json.load(f)
    #     contract_counter = {'curated': {}, 'solidifi': {}, 'clean': {}}
    #     for record in annotations:
    #         source_name = record['contract_name'].split('-')[0] + '.sol'
    #         normal = 1 - record['targets']
    #         buggy = 1 - normal
    #         if source_name in list(contract_category['curated'].keys()):
    #             if source_name not in contract_counter['curated']:
    #                 contract_counter['curated'][source_name] = {'normal': normal, 'buggy': buggy}
    #             else:
    #                 contract_counter['curated'][source_name]['normal'] += normal
    #                 contract_counter['curated'][source_name]['buggy'] += buggy
    #         elif source_name in list(contract_category['solidifi'].keys()):
    #             if source_name not in contract_counter['solidifi']:
    #                 contract_counter['solidifi'][source_name] = {'normal': normal, 'buggy': buggy}
    #             else:
    #                 contract_counter['solidifi'][source_name]['normal'] += normal
    #                 contract_counter['solidifi'][source_name]['buggy'] += buggy
    #         else:
    #             if source_name not in contract_counter['clean']:
    #                 contract_counter['clean'][source_name] = {'normal': normal, 'buggy': buggy}
    #             else:
    #                 contract_counter['clean'][source_name]['normal'] += normal
    #                 contract_counter['clean'][source_name]['buggy'] += buggy
    #     total_buggy = 0
    #     total_contract = 0
    #     # Curated
    #     curated_ind = np.arange(len(contract_counter['curated']))
    #     normal_counter = [contract['normal'] for contract in contract_counter['curated'].values()]
    #     buggy_counter = [contract['buggy'] for contract in contract_counter['curated'].values()]
    #     c_normal = axes[idx].bar(curated_ind, normal_counter, WIDTH, label='normal curated')
    #     c_buggy = axes[idx].bar(curated_ind, buggy_counter, WIDTH, bottom=normal_counter, label='buggy curated')
    #     curated_text = f'Curated Buggy: {sum(buggy_counter)}/{sum(buggy_counter) + sum(normal_counter)} ({round(sum(buggy_counter)/(sum(buggy_counter) + sum(normal_counter))*100,2)}%)'
    #     total_buggy += sum(buggy_counter)
    #     total_contract += sum(buggy_counter) + sum(normal_counter)
    #     max_y = max(0, max(normal_counter + buggy_counter))
    #     # Solidifi
    #     buggy_ind = np.arange(len(curated_ind), len(curated_ind) + len(contract_counter['solidifi']))
    #     normal_counter = [contract['normal'] for contract in contract_counter['solidifi'].values()]
    #     buggy_counter = [contract['buggy'] for contract in contract_counter['solidifi'].values()]
    #     s_normal = axes[idx].bar(buggy_ind, normal_counter, WIDTH, label='normal solidifi')
    #     s_buggy = axes[idx].bar(buggy_ind, buggy_counter, WIDTH, bottom=normal_counter, label='buggy solidifi')
    #     solidifi_text = f'Solidifi Buggy: {sum(buggy_counter)}/{sum(buggy_counter) + sum(normal_counter)} ({round(sum(buggy_counter)/(sum(buggy_counter) + sum(normal_counter))*100,2)}%)'
    #     total_buggy += sum(buggy_counter)
    #     total_contract += sum(buggy_counter) + sum(normal_counter)
    #     max_y = max(max_y, max(normal_counter + buggy_counter))
    #     # Clean
    #     clean_ind = np.arange(len(curated_ind) + len(buggy_ind), len(curated_ind) + len(buggy_ind) + len(contract_counter['clean']))
    #     normal_counter = [contract['normal'] for contract in contract_counter['clean'].values()]
    #     buggy_counter = [contract['buggy'] for contract in contract_counter['clean'].values()]
    #     cl_normal = axes[idx].bar(clean_ind, normal_counter, WIDTH/3, label='clean')
    #     total_contract += sum(normal_counter)
    #     # cl_buggy = axes[idx].bar(clean_ind, buggy_counter, WIDTH, bottom=normal_counter, label='clean')

    #     total_text = f'Total Buggy: {total_buggy}/{total_contract} ({round(total_buggy/total_contract*100,2)})%)'

    #     # axes[idx].axhline(0, color='grey', linewidth=0.8)
    #     axes[idx].set_ylabel(bug[:10])
    #     max_y = max(max_y, max(normal_counter + buggy_counter + [0]))

    #     # Label with label_type 'center' instead of the default 'edge'
    #     axes[idx].bar_label(c_normal, label_type='center', fontsize = 7)
    #     axes[idx].bar_label(c_buggy, label_type='center', fontsize = 7)
    #     axes[idx].bar_label(s_normal, label_type='center', fontsize = 7)
    #     axes[idx].bar_label(s_buggy, label_type='center', fontsize = 7)
    #     axes[idx].bar_label(cl_normal, label_type='edge', fontsize = 7)
    #     axes[idx].text(-1, max_y, '\n'.join([curated_text, solidifi_text, total_text]), horizontalalignment='left', verticalalignment='top')

    # axes[0].legend(ncol=5, bbox_to_anchor=(0, 1),
    #           loc='lower left', fontsize=8)
    # plt.show()

    # Graph statistics
    for bug, count in bug_type.items():
        compressed_graph = f'./experiments/ge-sc-data/source_code/{bug}/buggy_curated/cfg_cg_compressed_graphs.gpickle'
        print(bug, ' : ', count)
        nx_graph = nx.read_gpickle(compressed_graph)
        print('num of nodes: ', len(nx_graph.nodes()))
        print('num of edges: ', len(nx_graph.edges()))
        bug_node = 0
        for n, data in nx_graph.nodes(data=True):
            if data['node_info_vulnerabilities'] is None:
                continue
            bug_node += 1
        print('bug node: ', bug_node)
