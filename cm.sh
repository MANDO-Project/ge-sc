buggy_type = ['access_control', 'arithmetic', 'denial_of_service',
              'front_running', 'reentrancy', 'time_manipulation', 
              'unchecked_low_level_calls']
python node_classifier.py -ld ./logs/node_classification/cfg/node2vec/access_control --output_models ./models/node_classification/cfg/node2vec/access_control --dataset ./ge-sc-data/node_classification/cfg/access_control/buggy_curated --compressed_graph ./ge-sc-data/node_classification/cfg/access_control/buggy_curated/compressed_graphs.gpickle --node_feature nodetype --testset ./ge-sc-data/node_classification/cfg/curated/access_control --seed 1
 --test --checkpoint ./models/node_classification/cfg/nodetype/arithmetic/han_fold_0.pth

# Han
python node_classifier.py -ld ./logs/node_classification/cfg/line_han/reentrancy --output_models ./models/node_classification/cfg/line_han/reentrancy --dataset ./ge-sc-data/node_classification/cfg/reentrancy/buggy_curated --compressed_graph ./ge-sc-data/node_classification/cfg/reentrancy/buggy_curated/compressed_graphs.gpickle --testset ./ge-sc-data/node_classification/cfg/curated/reentrancy --seed 1  --node_feature line --feature_compressed_graph ./ge-sc-data/node_classification/cfg/gesc_matrices_node_embeddingmatrix_line_dim128_of_core_graph_of_reentrancy_compressed_graphs.pkl --cfg_feature_extractor ./data/smartbugs_wild/embeddings_buggy_currated_mixed/cfg_mixed/gesc_matrices_node_embedding/matrix_line_dim128_of_core_graph_of_reentrancy_compressed_graphs.pkl --feature_extractor ./models/node_classification/cfg/line/reentrancy/han_fold_0.pth


#########################
buggy_type = ['access_control', 'arithmetic', 'denial_of_service',
              'front_running', 'reentrancy', 'time_manipulation', 
              'unchecked_low_level_calls']
# GAE node2vec line cfg
python node_classifier.py -ld ./logs/node_classification/cfg/nodetype/unchecked_low_level_calls --output_models ./models/node_classification/cfg/nodetype/unchecked_low_level_calls --dataset ./ge-sc-data/node_classification/cfg/unchecked_low_level_calls/buggy_curated --compressed_graph ./ge-sc-data/node_classification/cfg/unchecked_low_level_calls/buggy_curated/compressed_graphs.gpickle --node_feature nodetype --seed 1 --testset ./ge-sc-data/node_classification/cfg/curated/unchecked_low_level_calls --feature_extractor ./ge-sc-data/node_classification/cfg/graph_embedding/undirected/gesc_matrices_node_embedding/matrix_nodetype_dim128_of_core_graph_of_unchecked_low_level_calls_compressed_graphs.pkl
--test --checkpoint ./models/node_classification/cfg/node2vec/denial_of_service/han_fold_0.pth

# GAE node2vec line call graph
python node_classifier.py -ld ./logs/node_classification/cfg/gae/access_control --output_models ./models/node_classification/cfg/gae/access_control --dataset ./ge-sc-data/node_classification/cfg/access_control/buggy_curated --compressed_graph ./ge-sc-data/node_classification/cfg/access_control/buggy_curated/compressed_graphs.gpickle --node_feature gae --feature_extractor ./ge-sc-data/node_classification/cfg/gesc_matrices_node_embedding/matrix_gae_dim128_of_core_graph_of_access_control_compressed_graphs.pkl --testset ./ge-sc-data/node_classification/cfg/curated --seed 1

# Graph classification
buggy_type = ['access_control', 'arithmetic', 'denial_of_service',
              'front_running', 'reentrancy', 'time_manipulation', 
              'unchecked_low_level_calls']

# Clean 50 + buggy + curated
## CFGs
python graph_classifier.py -ld ./logs/graph_classification/cfg/line/reentrancy --output_models ./models/graph_classification/cfg/line/reentrancy --dataset ./ge-sc-data/node_classification/cfg/reentrancy/clean_50_buggy_curated/ --compressed_graph ./ge-sc-data/node_classification/cfg/reentrancy/clean_50_buggy_curated/compressed_graphs.gpickle --label ./ge-sc-data/node_classification/cfg/reentrancy/clean_50_buggy_curated/graph_labels.json --node_feature line --feature_extractor ./ge-sc-data/node_classification/cfg/embeddings_cfg_clean50_buggy_curated_undirected/gesc_matrices_node_embedding/matrix_line_dim128_of_core_graph_of_reentrancy_compressed_graphs.pkl --seed 1 --testset ./ge-sc-data/ijcai20/cfg/reentrancy/source_code  

## CG not ijcai
python graph_classifier.py -ld ./logs/graph_classification/cg/nodetype/clean_50_buggy_curated/access_control --output_models ./models/graph_classification/cg/nodetype/clean_50_buggy_curated/access_control --dataset ./ge-sc-data/graph_classification/cg/access_control/clean_50_buggy_curated/ --compressed_graph ./ge-sc-data/graph_classification/cg/access_control/clean_50_buggy_curated/compress_call_graphs_no_solidity_calls_buggy.gpickle --label ./ge-sc-data/graph_classification/cg/access_control/clean_50_buggy_curated/graph_labels.json --node_feature nodetype --seed 1 --testset ./ge-sc-data/ijcai20/cfg/timestamp/source_code --feature_extractor ./ge-sc-data/node_classification/cg/embeddings_cg_clean50_buggy_curated_undirected/gesc_matrices_node_embedding/matrix_line_dim128_of_core_graph_of_access_control_compressed_graphs.pkl

## CG with ijcai
python graph_classifier.py -ld ./logs/graph_classification/cg/ijcai20_testset/line/time_manipulation --output_models ./models/graph_classification/cg/ijcai20_testset/line/time_manipulation --dataset ./ge-sc-data/node_classification/cg/time_manipulation/clean_50_buggy_curated_ijcai20 --compressed_graph ./ge-sc-data/node_classification/cg/time_manipulation/clean_50_buggy_curated_ijcai20/compressed_graphs.gpickle --label ./ge-sc-data/node_classification/cg/time_manipulation/clean_50_buggy_curated_ijcai20/graph_labels.json --node_feature line --testset ./ge-sc-data/ijcai20/cg/time_manipulation/source_code --seed 1 --feature_extractor ./ge-sc-data/node_classification/cg/embeddings_cg_clean50_buggy_curated_ijcai20/gesc_matrices_node_embedding/matrix_line_dim128_of_core_graph_of_time_manipulation_compressed_graphs.pkl

## CFG with ijcai
python graph_classifier.py -ld ./logs/graph_classification/cfg/ijcai20_testset/node2vec/reentrancy --output_models ./models/graph_classification/cfg/ijcai20_testset/node2vec/reentrancy --dataset ./ge-sc-data/node_classification/cfg/reentrancy/clean_50_buggy_curated_ijcai20 --compressed_graph ./ge-sc-data/node_classification/cfg/reentrancy/clean_50_buggy_curated_ijcai20/compressed_graphs.gpickle --label ./ge-sc-data/node_classification/cfg/reentrancy/clean_50_buggy_curated_ijcai20/graph_labels.json --node_feature node2vec --testset ./ge-sc-data/ijcai20/cfg/reentrancy/source_code --seed 1 --feature_extractor ./ge-sc-data/node_classification/cfg/embeddings_cfg_clean50_buggy_curated_ijcai20/gesc_matrices_node_embedding/matrix_node2vec_dim128_of_core_graph_of_reentrancy_compressed_graphs.pkl

# ijcai20
## CFGs
python graph_classifier.py -ld ./logs/graph_classification/cfg/ijcai20/line/timestamp --output_models ./models/graph_classification/cfg/ijcai20/line/timestamp --dataset ./ge-sc-data/ijcai20/timestamp/source_code --compressed_graph ./ge-sc-data/ijcai20/timestamp/source_code/compressed_graphs.gpickle --label ./ge-sc-data/ijcai20/timestamp/graph_labels.json --node_feature line --seed 1 --feature_extractor ./ge-sc-data/ijcai20/cfg/embeddings_ijcai20/gesc_matrices_node_embedding/matrix_line_dim128_of_core_graph_of_timestamp_compressed_graphs.pkl

## CGs
python graph_classifier.py -ld ./logs/graph_classification/cg/ijcai20/line/reentrancy --output_models ./models/graph_classification/cg/ijcai20/line/reentrancy --dataset ./ge-sc-data/ijcai20/cg/reentrancy/source_code --compressed_graph ./ge-sc-data/ijcai20/cg/reentrancy/source_code/compressed_graphs.gpickle --label ./ge-sc-data/ijcai20/cg/reentrancy/graph_labels.json --node_feature line --seed 1 --feature_extractor ./ge-sc-data/ijcai20/cg/embeddings_cg_ijcai20/gesc_matrices_node_embedding/matrix_line_dim128_of_core_graph_of_reentrancy_compressed_graphs.pkl
--testset ./ge-sc-data/ijcai20/cfg/reentrancy/source_code

# RGCN

## CFGs
python rgcn.py -ld ./logs/graph_classification/cfg/rgcn/node2vec/reentrancy --output_models ./models/graph_classification/cfg/rgcn/node2vec/reentrancy --dataset ./ge-sc-data/node_classification/cfg/reentrancy/clean_50_buggy_curated/ --compressed_graph ./ge-sc-data/node_classification/cfg/reentrancy/clean_50_buggy_curated/compressed_graphs.gpickle --label ./ge-sc-data/node_classification/cfg/reentrancy/clean_50_buggy_curated/graph_labels.json --node_feature node2vec --seed 1 --feature_extractor ./ge-sc-data/node_classification/cfg/embeddings_cfg_clean50_buggy_curated_undirected/gesc_matrices_node_embedding/matrix_node2vec_dim128_of_core_graph_of_reentrancy_compressed_graphs.pkl

## CGs
python rgcn.py -ld ./logs/graph_classification/cg/rgcn/node2vec/reentrancy --output_models ./models/graph_classification/cg/rgcn/node2vec/reentrancy --dataset ./ge-sc-data/node_classification/cg/reentrancy/clean_50_buggy_curated/ --compressed_graph ./ge-sc-data/node_classification/cg/reentrancy/clean_50_buggy_curated/compressed_graphs.gpickle --label ./ge-sc-data/node_classification/cg/reentrancy/clean_50_buggy_curated/graph_labels.json --node_feature node2vec --seed 1 --feature_extractor ./ge-sc-data/node_classification/cg/embeddings_cg_clean50_buggy_curated_undirected/gesc_matrices_node_embedding/matrix_node2vec_dim128_of_core_graph_of_reentrancy_compressed_graphs.pkl


# HGT
buggy_type = ['access_control', 'arithmetic', 'denial_of_service',
              'front_running', 'reentrancy', 'time_manipulation', 
              'unchecked_low_level_calls']

python hgt_nc.py -ld ./logs/node_classification/hgt/cfg_cg/nodetype/access_control --output_models ./models/node_classification/hgt/cfg_cg/nodetype/access_control --dataset ./ge-sc-data/node_classification/cfg_cg/access_control/buggy_curated --compressed_graph ./ge-sc-data/node_classification/cfg_cg/access_control/buggy_curated/compressed_graphs.gpickle --node_feature nodetype --seed 1 --testset ./ge-sc-data/node_classification/cfg/curated/access_control --feature_extractor ./ge-sc-data/node_classification/cfg_cg/gesc_matrices_node_embedding/matrix_nodetype_dim128_of_core_graph_of_access_control_compressed_graphs.pkl


python graph_classifier.py -ld ./logs/graph_classification/hgt/cfg_cg/nodetype/access_control --output_models ./models/graph_classification/hgt/cfg_cg/nodetype/access_control --dataset ./experiments/ge-sc-data/source_code/access_control/clean_57_buggy_curated_0 --compressed_graph ./experiments/ge-sc-data/source_code/access_control/clean_57_buggy_curated_0/cfg_cg_compressed_graphs.gpickle --label ./ge-sc-data/node_classification/cfg/access_control/clean_50_buggy_curated/graph_labels.json --node_feature nodetype --feature_extractor ./ge-sc-data/node_classification/cfg/embeddings_cfg_clean50_buggy_curated_undirected/gesc_matrices_node_embedding/matrix_line_dim128_of_core_graph_of_access_control_compressed_graphs.pkl

python graph_classifier.py -ld ./logs/graph_classification/hgt/cfg_cg/nodetype/reentrancy --output_models ./models/graph_classification/hgt/cfg_cg/nodetype/reentrancy --dataset ./experiments/ge-sc-data/source_code/reentrancy/clean_71_buggy_curated_0 --compressed_graph ./experiments/ge-sc-data/source_code/reentrancy/clean_71_buggy_curated_0/cfg_cg_compressed_graphs.gpickle --label ./experiments/ge-sc-data/source_code/reentrancy/clean_71_buggy_curated_0/graph_labels.json --node_feature nodetype --feature_extractor ./experiments/ge-sc-data/source_code/gesc_matrices_node_embedding/matrix_gae_dim128_of_core_graph_of_reentrancy_cfg_cg_buggy_curated.pkl --seed 1

python graph_classifier.py -ld ./logs/graph_classification/byte_code/hgt/cfg/nodetype/reentrancy --output_models ./models/graph_classification/byte_code/hgt/cfg/nodetype/reentrancy --dataset ./experiments/ge-sc-data/byte_code/smartbugs/creation/gpickles/reentrancy/clean_71_buggy_curated_0 --compressed_graph ./experiments/ge-sc-data/byte_code/smartbugs/creation/gpickles/access_control/clean_57_buggy_curated_0/compressed_graphs/cfg_compressed_graphs.gpickle --label ./experiments/ge-sc-data/byte_code/smartbugs/creation/gpickles/reentrancy/clean_71_buggy_curated_0/graph_labels.json --node_feature nodetype --feature_extractor ./experiments/ge-sc-data/source_code/gesc_matrices_node_embedding/matrix_gae_dim128_of_core_graph_of_reentrancy_cfg_buggy_curated.pkl --seed 1

python graph_classifier.py -ld ./logs/graph_classification/byte_code/han/cfg/nodetype/access_control --output_models ./models/graph_classification/byte_code/han/cfg/nodetype/access_control --compressed_graph ./experiments/ge-sc-data/byte_code/smartbugs/creation/gpickles/access_control/clean_57_buggy_curated_0/compressed_graphs/creation_balanced_compressed_graphs.gpickle --label ./experiments/ge-sc-data/byte_code/smartbugs/contract_labels/access_control/creation_balanced_contract_labels.json --node_feature nodetype --feature_extractor ./experiments/ge-sc-data/source_code/gesc_matrices_node_embedding/matrix_gae_dim128_of_core_graph_of_access_control_cfg_buggy_curated.pkl --seed 1

## Graph classifier ethor byte code
python graph_classifier.py -ld ./logs/graph_classification/byte_code/hgt/cfg/nodetype/ethor --output_models ./models/graph_classification/byte_code/hgt/cfg/nodetype/ethor --dataset ./experiments/ge-sc-data/byte_code/ethor --compressed_graph ./experiments/ge-sc-data/byte_code/ethor/compressed_graphs/cfg_compressed_graphs.gpickle --label ./experiments/ge-sc-data/byte_code/ethor/graph_labels.json --node_feature nodetype --seed 1 --feature_extractor ./ge-sc-data/node_classification/cfg/embeddings_cfg_clean50_buggy_curated_undirected/gesc_matrices_node_embedding/matrix_line_dim128_of_core_graph_of_ethor_compressed_graphs.pkl

python graph_classifier.py -ld ./logs/graph_classification/byte_code/hgt/cfg/nodetype/ethersolve --output_models ./models/graph_classification/byte_code/hgt/cfg/nodetype/ethersolve --dataset /home/minhnn/minhnn/ICSE/EtherSolve_ICPC2021_ReplicationPackage/data/reentrancy/creation --compressed_graph /home/minhnn/minhnn/ICSE/EtherSolve_ICPC2021_ReplicationPackage/data/reentrancy/creation/cfg_compressed_graphs.gpickle --label /home/minhnn/minhnn/ICSE/crytic-compile/data/reentrancy/cfg/graph_labels.json --node_feature nodetype --seed 1 --feature_extractor ./ge-sc-data/node_classification/cfg/embeddings_cfg_clean50_buggy_curated_undirected/gesc_matrices_node_embedding/matrix_line_dim128_of_core_graph_of_ethersolve_compressed_graphs.pkl
