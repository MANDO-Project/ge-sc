import networkx as nx
import dgl

nx_graph = nx.read_gpickle('ge-sc/data/reentrancy/source_code/Bank_merge_contract_graph.gpickle')
print(nx.info(nx_graph))
print(nx_graph.nodes(data=True))
print(nx_graph.edges(data=True))

dgl_graph = dgl.from_networkx(nx_graph, edge_attrs=['label'])
# dgl_graph = dgl.from_networkx(nx_graph, node_attrs=['label'])
print(dgl_graph)
print(dgl_graph.nodes())
print(dgl_graph.edges())

# hetero_dgl_graph = dgl.to_heterogeneous(dgl_graph, metagraph=nx_graph)
# print(hetero_dgl_graph)

