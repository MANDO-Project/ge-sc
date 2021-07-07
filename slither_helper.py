from networkx.algorithms import cluster
import pygraphviz as pgv
import networkx as nx

from copy import deepcopy
from slither.slither import Slither

# fn = 'data/reentrancy/source_code/simple_dao'
fn = 'data/reentrancy/source_code/Bank'

slither = Slither(fn + '.sol')

merge_contract_graph = None
for contract in slither.contracts:  
    print('Contract: ', contract.name)

    merged_graph = None
    for function in contract.functions + contract.modifiers:  
        print('Function: {}'.format(function.name))  
  
        # print('\tRead: {}'.format([v.name for v in function.state_variables_read]))  
        # print('\tWritten {}'.format([v.name for v in function.state_variables_written]))

        for node in function.nodes:
            print('Node:', node, 'NodeType:', node.type, 'NodeExpression:', node.expression)
        
        filename = "{}-{}-{}.dot".format(fn, contract.name, function.full_name)
        function.slithir_cfg_to_dot(filename)
        
        filename = "{}-{}-{}_dom.dot".format(fn, contract.name, function.full_name)
        function.dominator_tree_to_dot(filename)

        data = function.slithir_cfg_to_dot_str()

        gv_graph = pgv.AGraph(data)
        # print(gv_graph)

        nx_graph = nx.nx_agraph.from_agraph(gv_graph)
        print(nx.info(nx_graph))

        print(nx_graph.nodes(data=True))

        nx_graph.add_node('function.name', label=contract.name + '_' + function.name, contract=contract.name)
        nx_graph.add_edge('function.name', 0)

        print(nx_graph.edges(data=True))

        if merged_graph is None:
            nx_graph = nx.relabel_nodes(nx_graph, lambda x: contract.name + '_' + function.name + '_' + str(x), copy=False)
            merged_graph = deepcopy(nx_graph)
        else:
            nx_graph = nx.relabel_nodes(nx_graph, lambda x: contract.name + '_' + function.name + '_' + str(x), copy=False)
            merged_graph = nx.union(merged_graph, nx_graph)

    print(nx.info(merged_graph))

    print(merged_graph.nodes(data=True))
    print(merged_graph.edges(data=True))

    nx.nx_agraph.write_dot(merged_graph, fn + '_' + contract.name + '_merged_graph.dot')
    
    if merge_contract_graph is None:
        merge_contract_graph = deepcopy(merged_graph)
    else:
        merge_contract_graph = nx.union(merge_contract_graph, merged_graph)

print(nx.info(merge_contract_graph))
print(merged_graph.edges(data=True))
nx.nx_agraph.write_dot(merge_contract_graph, fn + '_merge_contract_graph.dot')
nx.write_gpickle(merge_contract_graph, fn + '_merge_contract_graph.gpickle')

gv_graph = pgv.AGraph(fn + '.sol' + '.all_contracts.call-graph.dot')
for subgraph in gv_graph.subgraphs():
    print(subgraph.get_name())
    print(subgraph.nodes())

print(gv_graph.nodes())
print(gv_graph.edges())

temp_graph = nx.MultiDiGraph()
for node in merge_contract_graph.nodes(data=True):
    if 'function.name' in node[0]:
        cluster_name = node[1]['contract']
        for subgraph in gv_graph.subgraphs():
            if cluster_name in subgraph.get_name():
                print(subgraph.nodes())
                print(subgraph.get_name())
                print(node)
                
