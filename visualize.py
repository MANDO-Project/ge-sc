import os
from os.path import join

import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pygraphviz as pgv
from matplotlib.collections import PolyCollection
from matplotlib import cm
from scipy.stats import poisson

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
    # nxg = './experiments/ge-sc-data/source_code/access_control/clean_57_buggy_curated_0/cfg_cg_compressed_graphs.gpickle'
    # plot_graph(nxg)


    # creation_path = '/home/minhnn/minhnn/ICSE/EtherSolve_ICPC2021_ReplicationPackage/data/reentrancy/creation/'
    # output = '/home/minhnn/minhnn/ICSE/ge-sc/forensics'
    # runtime_path = '/home/minhnn/minhnn/ICSE/EtherSolve_ICPC2021_ReplicationPackage/data/reentrancy/runtime/'
    # creation_files = [join(creation_path, f) for f in os.listdir(creation_path) if f.endswith('.sol')]
    # runtime_files = [join(runtime_path, f) for f in os.listdir(runtime_path) if f.endswith('.sol')]
    # nodes_edges_correlation([creation_files], join(output, 'creation_node_edge_correlation.png'))
    # nodes_edges_correlation([runtime_files], join(output, 'runtime_node_edge_correlation.png'))

    # # Compressed graph forensic
    # bug_type = {'access_control': 57, 'arithmetic': 60, 'denial_of_service': 46,
    #           'front_running': 44, 'reentrancy': 71, 'time_manipulation': 50, 
    #           'unchecked_low_level_calls': 95}
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

    # Generate historgram
    last_hidden = '/home/minhnn/minhnn/ICSE/ge-sc/forensics/graph_hiddens/reentrancy/last_attention.pt'
    hiddens = torch.load(last_hidden)
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

    ax = plt.figure().add_subplot(projection='3d')

    x = np.linspace(torch.min(hiddens).item(), torch.max(hiddens).item(), num=200)
    hiddens_hist = [np.histogram(h.numpy(), bins=x, range=None, normed=None, weights=None, density=None)[0].tolist() for h in hiddens]
    # print([np.sum(hid * np.diff(x)) for hid in hiddens_hist])
    print(max(hiddens_hist[0]))
    lambdas = lambdas = range(1, hiddens.shape[0]+1)
    # verts[i] is a list of (x, y) pairs defining polygon i.
    verts = [polygon_under_graph(x, y) for y in hiddens_hist]
    # print(verts)
    # facecolors = plt.colormaps()[0](np.linspace(0, 1, len(verts)))
    viridis = cm.get_cmap('viridis', 8)
    facecolors = viridis(np.linspace(0, 1, len(verts)))
    
    poly = PolyCollection(verts, facecolors=facecolors, alpha=.6)
    ax.add_collection3d(poly, zs=lambdas, zdir='y')

    ax.set(xlim=(torch.min(hiddens).item(), torch.max(hiddens).item()), ylim=(0, 30), zlim=(0, 7),
           xlabel='x', ylabel=r'$\lambda$', zlabel='probability')

    plt.show()
