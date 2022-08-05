
import torch

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
    dataset = GESCData(graph_path, split=0.2, gpu=False)
    print(dataset.data)
    # Explain it with GraphSVX
    model = NodeClassifier(graph_path, feature_extractor=None, node_feature='nodetype', device=device)
    model.load_state_dict(torch.load(checkpoint))
    model.to(device)
    model.eval()
    explainer = GraphSVX(dataset.data, model, gpu=False)
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