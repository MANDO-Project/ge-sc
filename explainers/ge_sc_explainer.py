
import torch
import networkx as nx
import matplotlib.pyplot as plt

from . import configs
from .dataloader import GESCData
from .explainers import GraphSVX
from sco_models.model_hgt import HGTVulNodeClassifier as NodeClassifier

def main():
    args = configs.arg_parse()
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # else:
    #     device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    graph_path = './experiments/ge-sc-data/source_code/reentrancy/buggy_curated/cfg_cg_compressed_graphs.gpickle'
    checkpoint = './models/node_classification/source_code/nodetype/reentrancy/logs_hgt.pth'
    # Dataloader
    dataloader = GESCData(graph_path, split=None, gpu=False)
    dataset = dataloader.data

    # Get node id
    node_indexes = dataloader.get_nodes_of_source(source_file='0x23a91059fdc9579a9fbd0edc5f2ea0bfdb70deb4.sol')
    if len(node_indexes) > 0:
        print('Find nodes of source file')
        
    args.indexes = [3724]
    subgraph = dataloader.original_graph.subgraph(args.indexes)
    labels = {}
    for id, node in subgraph.nodes(data=True):
        nodetype = node['node_type'][:7]
        labels[id] = f'{id}_{nodetype}'
    # pos = nx.spring_layout(subgraph, seed=1)
    # pos = nx.circular_layout(subgraph) #,  k=0.3, iterations=50)
    # options = {
    # "font_size": 36,
    # "node_size": 3000,
    # "node_color": "white",
    # "edgecolors": "black",
    # "linewidths": 5,
    # "width": 5,
    # }
    # nx.draw_networkx(subgraph, pos, with_labels=False)
    # nx.draw_networkx_labels(subgraph, pos, labels, font_size=10)
    # ax = plt.gca()
    # ax.margins(0.20)
    # plt.axis("off")
    # plt.show()
    # plt.tight_layout()
    # plt.axis("off")
    # plt.show()
    # return
    # print(dataset)
    # std = dataset.x.std(axis=0)
    # mean = dataset.x.mean(axis=0)
    # print(f'mean: {mean}\n std: {std}')
    # mean_subgraph = dataset.x[250, :]
    # print('mean subgraph: ', mean_subgraph)
    # mean_subgraph = torch.where(mean_subgraph >= mean - 0.25*std, mean_subgraph,
    #                             torch.ones_like(mean_subgraph)*100)
    # mean_subgraph = torch.where(mean_subgraph <= mean + 0.25*std, mean_subgraph,
    #                             torch.ones_like(mean_subgraph)*100)
    # feat_idx = (mean_subgraph == 100).nonzero()
    # discarded_feat_idx = (mean_subgraph != 100).nonzero()
    # print('feat_idx: ', feat_idx)
    # print('discarded_feat_idx: ', discarded_feat_idx)
    # return
    # Explain it with GraphSVX
    model = NodeClassifier(graph_path, feature_extractor=None, node_feature='nodetype', device=device)
    model.load_state_dict(torch.load(checkpoint))
    model.to(device)
    model.eval()
    explainer = GraphSVX(dataset, model, gpu=False)
    explainer.explain(args.indexes,
                      args.hops,
                      args.num_samples,
                      args.info,
                      args.multiclass,
                      args.fullempty,
                      args.S,
                      args.hv,
                      args.feat,
                      args.coal,
                      args.g,
                      args.regu,
                      True)

if __name__ == '__main__':
    main()