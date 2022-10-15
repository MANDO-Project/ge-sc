
import os
import torch
import networkx as nx
import matplotlib.pyplot as plt
from shutil import copy

from . import configs
from .dataloader import GESCData
from .explainers import GraphSVX
from sco_models.model_node_classification import MANDONodeClassifier as NodeClassifier

def main():
    args = configs.arg_parse()
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # else:
    #     device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    smart_contracts = {
                        # 'reentrancy': 'simple_dao',
                        # 'access_control': [],
                        # 'arithmetic': 'integer_overflow_multitx_onefunc_feasible',
                        # 'denial_of_service': 'dos_simple',
                        # 'front_running': 'buggy_29',
                        # 'reentrancy': '0x23a91059fdc9579a9fbd0edc5f2ea0bfdb70deb4',
                        # 'time_manipulation': 'ether_lotto',
                        # 'unchecked_low_level_calls': 'lotto'
                    }
    # curated_bug_path = f'./experiments/ge-sc-data/source_code/access_control/curated/'
    # smart_contracts['access_control'] = [f.split('.')[0] for f in os.listdir(curated_bug_path) if f.endswith('.sol')][:1]
    bug_list = [
                # 'access_control',
                'arithmetic',
                # 'denial_of_service',
                'front_running',
                'reentrancy',
                'time_manipulation',
                'unchecked_low_level_calls'
                ]
    smart_contracts = {}
    for bug in bug_list:
        curated_bug_path = f'./experiments/ge-sc-data/source_code/{bug}/curated/'
        smart_contracts[bug] = [f.split('.')[0] for f in os.listdir(curated_bug_path) if f.endswith('.sol')]


    for bug_type, sc_paths in smart_contracts.items():
        print('***********************************************************************')
        curated_bug_path = f'./experiments/ge-sc-data/source_code/{bug_type}/curated/'
        graph_path = f'./experiments/ge-sc-data/source_code/{bug_type}/buggy_curated/cfg_cg_compressed_graphs.gpickle'
        checkpoint = f'./experiments/models/node_classification/hgt/cfg_cg/nodetype/buggy_curated/{bug_type}_hgt_0.pth'
        # checkpoint = f'./models/node_classification/source_code/hgt/cfg_cg/nodetype/{bug_type}/{bug_type}_hgt.pth'
        print('Loading graph: ', graph_path)
        print('Loading model:', checkpoint)
        if not os.path.exists(f'./results/ge_sc/{bug_type}'):
            os.mkdir(f'./results/ge_sc/{bug_type}')
        for sc_path in sc_paths:
            print(f'Processing {sc_path}')
            log_dir = f'./results/ge_sc/{bug_type}/{sc_path}'
            sc_name = sc_path + '.sol'
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
            copy(os.path.join(curated_bug_path, sc_name), os.path.join(log_dir))
            
        # Dataloader
            dataloader = GESCData(graph_path, split=None, gpu=False)
            dataset = dataloader.data
            # print(dataset.edge_index)

            # Init model
            model = NodeClassifier(graph_path, feature_extractor=None, node_feature='nodetype', device=device)
            model.load_state_dict(torch.load(checkpoint))
            model.to(device)
            model.eval()
            with torch.no_grad():
                logits, edge_attn = model(get_attn=True)
                logits = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
            # for canonical_etype in model.symmetrical_global_graph.canonical_etypes:
            model.symmetrical_global_graph.edata['t'] = edge_attn
                # print(model.symmetrical_global_graph.edge_ids[canonical_etype])
            # nx_graph = model.symmetrical_global_graph.to_net
            # return
            # Get node id
            node_indexes = dataloader.get_nodes_of_source(source_file=sc_name)
            if len(node_indexes) > 0:
                print('Find nodes of source file')
                
            # args.indexes = [3724, 7, 10, 20, 30, 40]
            args.indexes = node_indexes
            # args.indexes = [18308]
            print('Node indexes: ', args.indexes)

            # Generate node information

            subgraph = dataloader.original_graph.subgraph(args.indexes)
            labels = {}
            node_color = []
            # for src, dest, edge in subgraph.edges(data=True):
            #     print(edge)
            # return
            for id, node in subgraph.nodes(data=True):
                predited = logits[id][1]
                nodetype = node['node_type'][:10]
                node_source_code_lines = ','.join(map(str, node['node_source_code_lines']))
                labels[id] = f'\n{id}_{nodetype}\n{predited:4f}\n{node_source_code_lines}'
                gt = node['node_info_vulnerabilities']
                if gt:
                    node_color.append('#ff2d00')
                else:
                    node_color.append('#00a912')
                _gt = 'bug' if gt else 'non-bug'

            # pos = nx.spring_layout(subgraph, seed=args.seed)
            pos = nx.circular_layout(subgraph, scale=3) #,  k=0.3, iterations=50)
            options = {
            "font_size": 10,
            "node_size": 1000,
            "node_color": "white",
            "edgecolors": "black",
            "linewidths": 5,
            "width": 5,
            }
            edge_labels = {}
            for u, v, d in subgraph.edges(data=True):
                dgl_src_node = subgraph.nodes[u]['node_hetero_id']
                src_type = subgraph.nodes[u]['node_type']
                dgl_dst_node = subgraph.nodes[v]['node_hetero_id']
                dst_type = subgraph.nodes[v]['node_type']
                canonical_type = (src_type, d['edge_type'], dst_type)
                ed_id = model.symmetrical_global_graph.edge_id(dgl_src_node, dgl_dst_node, etype=canonical_type)
                edge_txt = torch.round(torch.flatten(edge_attn[canonical_type][ed_id]), decimals=4)
                edge_labels[(u, v)] = ','.join(map(str, edge_txt.tolist()))

            nx.draw_networkx(subgraph, pos, with_labels=False, node_color=node_color, node_size=50)
            nx.draw_networkx_labels(subgraph, pos, labels, font_size=5)
            # nx.draw_networkx_edge_labels(subgraph, pos, edge_labels, font_size=4, rotate=False)
            ax = plt.gca()
            ax.margins(0.05)
            plt.axis("off")
            plt.tight_layout()
            # plt.show()
            _sc_name = sc_name.split('.')[0]
            save_path = f'{log_dir}/{bug_type}_{_sc_name}'
            plt.savefig(save_path, dpi=1000)
            plt.clf()
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
                            True,
                            log_dir,
                            logits)

if __name__ == '__main__':
    main()