import os
from os.path import join
from copy import deepcopy
from tqdm import tqdm

import networkx as nx
from slither.slither import Slither
from slither.core.cfg.node import NodeType


def compress_full_smart_contracts(smart_contracts, output):
    full_graph = None
    for sc in tqdm(smart_contracts):
        slither = Slither(sc)
        merge_contract_graph = None
        for contract in slither.contracts:
            merged_graph = None
            for idx, function in enumerate(contract.functions + contract.modifiers):  

                nx_g = nx.MultiDiGraph()
                for nidx, node in enumerate(function.nodes):
                    node_label = "Node Type: {}\n".format(str(node.type))
                    node_type = str(node.type)
                    if node.expression:
                        node_label += "\nEXPRESSION:\n{}\n".format(node.expression)
                        node_expression = str(node.expression)
                    else:
                        node_expression = None
                    if node.irs:
                        node_label += "\nIRs:\n" + "\n".join([str(ir) for ir in node.irs])
                        node_irs = "\n".join([str(ir) for ir in node.irs])
                    else:
                        node_irs = None
                    nx_g.add_node(node.node_id, label=node_label,
                                node_type=node_type, node_expression=node_expression, node_irs=node_irs,
                                function_fullname=function.full_name, contract_name=contract.name)
                    
                    if node.type in [NodeType.IF, NodeType.IFLOOP]:
                        true_node = node.son_true
                        if true_node:
                            nx_g.add_edge(node.node_id, true_node.node_id, edge_type='if_true', label='True')
                        false_node = node.son_false
                        if false_node:
                            nx_g.add_edge(node.node_id, false_node.node_id, edge_type='if_false', label='False')
                    else:
                        for son in node.sons:
                            nx_g.add_edge(node.node_id, son.node_id, edge_type='next', label='Next')

                nx_graph = nx_g
                # add FUNCTION_NAME node
                nx_graph.add_node('function.name', label=contract.name + '_' + function.full_name,
                                node_type='FUNCTION_NAME', node_expression=None, node_irs=None,
                                function_fullname=function.full_name, contract_name=contract.name)
                nx_graph.add_edge('function.name', 0, edge_type='next', label='Next')

                if merged_graph is None:
                    nx_graph = nx.relabel_nodes(nx_graph, lambda x: contract.name + '_' + function.name + '_' + str(x), copy=False)
                    merged_graph = deepcopy(nx_graph)
                else:
                    nx_graph = nx.relabel_nodes(nx_graph, lambda x: contract.name + '_' + function.name + '_' + str(x), copy=False)
                    merged_graph = nx.disjoint_union(merged_graph, nx_graph)

            if merge_contract_graph is None:
                merge_contract_graph = deepcopy(merged_graph)
            elif merged_graph is not None:
                merge_contract_graph = nx.disjoint_union(merge_contract_graph, merged_graph)
        
        if full_graph is None:
            full_graph = deepcopy(merge_contract_graph)
        elif merge_contract_graph is not None:
            full_graph = nx.disjoint_union(full_graph, merge_contract_graph)
    
    nx.nx_agraph.write_dot(full_graph, join(output, 'compress_graphs.dot'))
    nx.write_gpickle(merge_contract_graph, join(output, 'compress_graphs.gpickle'))


if __name__ == '__main__':
    smart_contract_path = '/home/minhnn/minhnn/ICSE/datasets/Etherscan_Contract/extracted_source_code' 
    output_path = '/home/minhnn/minhnn/ICSE/datasets/Etherscan_Contract/compressed_smart_contracts'
    smart_contracts = [join(smart_contract_path, f) for f in os.listdir(smart_contract_path)]
    compress_full_smart_contracts(smart_contracts, output_path)