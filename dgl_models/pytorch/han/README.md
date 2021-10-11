# Heterogeneous Graph Attention Network (HAN) with DGL

This is an attempt to apply HAN for Vulnerability detection in smart contracts.
## Training

`python main.py --dataset ./dataset/aggregate/source_code --compressed_graph ./dataset/aggregate/compressed_graph/compress_graphs.gpickle --label ./dataset/aggregate/labels.json`

## Testing

`python main.py --test --testset ./dataset/smartbugs/source_code --dataset ./dataset/aggregate/source_code --compressed_graph ./dataset/aggregate/compressed_graph/compress_graphs.gpickle --label ./dataset/aggregate/labels.json --checkpoint ./models/model_han_fold_2.pth`

## Visuallization

`tensorboard --logdir ./logs`