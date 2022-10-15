import json
import torch
import networkx as nx
import pandas as pd
from torch import nn

from sco_models.model_node_classification import MANDONodeClassifier as NodeClassifier
from sco_models.graph_utils import get_node_label

MODEL = 'hgt'

def convert_json_to_csv(json_spec, output):
    with open(json_spec, 'r') as f:
        graph_spec = json.load(f)
    columns = list(graph_spec[0].keys())
    df = pd.DataFrame.from_records(graph_spec, columns=columns)
    with pd.ExcelWriter(output, mode='w') as writer:  
                df.to_excel(writer)


def main():
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    bug_list = [
            'access_control', 
            'arithmetic', 
            'denial_of_service',
            'front_running', 
            'reentrancy', 
            'time_manipulation', 
            'unchecked_low_level_calls'
            ]
    feature_type = ['nodetype', 'metapath2vec', 'gae', 'line', 'node2vec']
    node_feature = 'nodetype'
    for bugtype in bug_list:
        graph_spec = []
        print(f"Loading {bugtype} model")
        graph_path = f'./experiments/ge-sc-data/source_code/{bugtype}/buggy_curated/cfg_cg_compressed_graphs.gpickle'
        checkpoint = f'./experiments/models/node_classification/hgt/cfg_cg/{node_feature}/buggy_curated/{bugtype}_hgt_0.pth'
        output_json = f'./forensics/specification/{bugtype}_{node_feature}_{MODEL}_spec.json'
        output_csv = f'./forensics/specification/{bugtype}_{node_feature}_{MODEL}_spec.xlsx'
        model = NodeClassifier(graph_path, feature_extractor=None, node_feature=node_feature, device=device)
        model.load_state_dict(torch.load(checkpoint))
        model.to(device)
        model.eval()
        # targets, _, _ = get_node_label(model.nx_graph)
        # targets = torch.tensor(targets, device=device).cpu().numpy()
        # with torch.no_grad():
        #     logits = model()
        #     logits = nn.functional.softmax(logits, dim=1).cpu().numpy()
        #     print(logits[:10])
        # print(model.nx_graph)
        # pageranks = nx.pagerank(model.nx_graph)
        # betweenness_centrality = nx.betweenness_centrality(model.nx_graph)
        # voteranks = nx.voterank(model.nx_graph)
        # for nid in range(len(model.nx_graph)):
        #     node_spec = {}
        #     node_spec['id'] = nid
        #     node_spec['predict'] = float(logits[nid][1])
        #     node_spec['ground_truth'] = float(targets[nid])
        #     node_spec['pageranks'] = float(pageranks[nid])
        #     node_spec['betweenness_centrality'] = float(betweenness_centrality[nid])
        #     node_spec['source_file'] = model.nx_graph.nodes[nid]['source_file']

        #     try:
        #         rank = voteranks.index(nid)
        #     except ValueError:
        #         rank = -1
        #     node_spec['voteranks'] = rank
        #     graph_spec.append(node_spec)
        # print(graph_spec[10])
        # with open(output_json, 'w') as f:
        #     json.dump(graph_spec, f, indent=4)
        # convert_json_to_csv(output_json, output_csv)
        print(model.nx_graph.node)

if __name__ == '__main__':
    main()