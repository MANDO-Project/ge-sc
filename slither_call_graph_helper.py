from networkx.algorithms import cluster
import logging
import pygraphviz as pgv
import networkx as nx

from copy import deepcopy
from scipy.integrate._ivp.radau import C
from slither.slither import Slither
from collections import defaultdict

from slither.core.cfg.node import Node, NodeType

from slither.printers.call import call_graph
from slither.printers.abstract_printer import AbstractPrinter
from slither.core.declarations.solidity_variables import SolidityFunction
from slither.core.declarations.function import Function
from slither.core.variables.variable import Variable

logger = logging.getLogger("Slither-simil")

# return graph node (with optional label)
def _node(node, label=None):
    return (
            f"{node}",
            f"{label}" if label is not None else "",
        )

# return graph edge with edge type
def _edge(from_node, to_node, edge_type, label):
    return (f"{from_node}", f"{to_node}", edge_type, label)
    

# return unique id for contract function to use as node name
def _function_node(contract, function):
    return f"{contract.id}_{contract.name}_{function.full_name}"

# return unique id for solidity function to use as node name
def _solidity_function_node(solidity_function):
    return f"[Solidity]_{solidity_function.full_name}"

# pylint: disable=too-many-arguments
def _process_internal_call(
    contract,
    function,
    internal_call,
    contract_calls,
    solidity_functions,
    solidity_calls,
):
    if isinstance(internal_call, (Function)):
        contract_calls[contract].add(
            _edge(
                _function_node(contract, function),
                _function_node(contract, internal_call),
                edge_type='internal_call',
                label='internal_call'
            )
        )

    elif isinstance(internal_call, (SolidityFunction)):
        solidity_functions.add(
            _node(_solidity_function_node(internal_call),
                  _solidity_function_node(internal_call)),
        )
        solidity_calls.add(
            _edge(
                _function_node(contract, function),
                _solidity_function_node(internal_call),
                edge_type='solidity_call',
                label='solidity_call'
            )
        )

def _process_external_call(
    contract,
    function,
    external_call,
    contract_functions,
    external_calls,
    all_contracts,
):
    external_contract, external_function = external_call

    if not external_contract in all_contracts:
        return

    # add variable as node to respective contract
    if isinstance(external_function, (Variable)):
        contract_functions[external_contract].add(
            _node(
                _function_node(external_contract, external_function),
                f"{external_contract.name}_{external_function.full_name}"
            )
        )

    external_calls.add(
        _edge(
            _function_node(contract, function),
            _function_node(external_contract, external_function),
            edge_type='external_call',
            label='external_call'
        )
    )

def _render_internal_calls(nx_graph, contract, contract_functions, contract_calls):
    if len(contract_functions[contract]) > 0:
        for contract_function in contract_functions[contract]:
            node_id = contract_function[0]
            node_label = contract_function[1]
            if 'fallback' in node_id:
                node_type = 'fallback_function'
            else:
                node_type = 'contract_function'                
            nx_graph.add_node(node_id, label=node_label, node_type=node_type)
    
    if len(contract_calls[contract]) > 0:
        for contract_call in contract_calls[contract]:
            # print('contract_call:', contract_call)
            source = contract_call[0]
            target = contract_call[1]
            edge_type = contract_call[2]
            edge_label = contract_call[3]
            nx_graph.add_edge(source, target, label=edge_label, edge_type=edge_type)

def _render_solidity_calls(nx_graph, solidity_functions, solidity_calls):
    if len(solidity_functions) > 0:
        for solidity_function in solidity_functions:
            print(solidity_function)
            node_id = solidity_function[0]
            node_label = solidity_function[1]
            node_type = 'solidity_function'
            nx_graph.add_node(node_id, label=node_label, node_type=node_type)
    
    if len(solidity_calls) > 0:
        for solidity_call in solidity_calls:
            source = solidity_call[0]
            target = solidity_call[1]
            edge_type = solidity_call[2]
            edge_label = solidity_call[3]
            nx_graph.add_edge(source, target, label=edge_label, edge_type=edge_type)

def _render_external_calls(nx_graph, external_calls):
    if len(external_calls) > 0:
        for external_call in external_calls:
            source = external_call[0]
            target = external_call[1]
            edge_type = external_call[2]
            edge_label = external_call[3]
            nx_graph.add_edge(source, target, label=edge_label, edge_type=edge_type)

def _process_function(
    contract,
    function,
    contract_functions,
    contract_calls,
    solidity_functions,
    solidity_calls,
    external_calls,
    all_contracts,
):
    contract_functions[contract].add(
        _node(_function_node(contract, function), 
              f"{contract.name}_{function.full_name}"),
    )

    for internal_call in function.internal_calls:
        _process_internal_call(
            contract,
            function,
            internal_call,
            contract_calls,
            solidity_functions,
            solidity_calls,
        )

    for external_call in function.high_level_calls:
        _process_external_call(
            contract,
            function,
            external_call,
            contract_functions,
            external_calls,
            all_contracts,
        )

def _process_functions(functions):
    contract_functions = defaultdict(set)  # contract -> contract functions nodes
    contract_calls = defaultdict(set)  # contract -> contract calls edges

    solidity_functions = set()  # solidity function nodes
    solidity_calls = set()  # solidity calls edges
    external_calls = set()  # external calls edges

    all_contracts = set()

    for function in functions:
        all_contracts.add(function.contract_declarer)

    for function in functions:
        _process_function(
            function.contract_declarer,
            function,
            contract_functions,
            contract_calls,
            solidity_functions,
            solidity_calls,
            external_calls,
            all_contracts,
        )

    # print('contract_functions:', contract_functions)
    # print('solidity_functions:', solidity_functions)
    # print('contract_calls:', contract_calls)
    # print('solidity_calls:', solidity_calls)
    # print('external_calls:', external_calls)
    # print('all_contracts:', all_contracts)

    all_contracts_graph = nx.MultiDiGraph()
    for contract in all_contracts:
        # print('contract_functions:', contract_functions[contract])
        # print('contract_calls:', contract_calls[contract])
        _render_internal_calls(all_contracts_graph, contract,
                               contract_functions, contract_calls)
    
    _render_solidity_calls(all_contracts_graph, solidity_functions, solidity_calls)
    _render_external_calls(all_contracts_graph, external_calls)

    return all_contracts_graph


class GESCPrinters(AbstractPrinter):
    ARGUMENT = "call-graph"
    HELP = "Export the call-graph of the contracts to a dot file and a gpickle file"

    WIKI = "https://github.com/trailofbits/slither/wiki/Printer-documentation#call-graph"

    def output(self, filename):
        """
        Output the graph in filename
        Args:
            filename(string)
        """

        all_contracts_filename = ""
        if not filename.endswith(".dot"):
            all_contracts_filename = f"{filename}.all_contracts.call-graph"
        if filename == ".dot":
            all_contracts_filename = "all_contracts"

        # Avoid dupplicate funcitons due to different compilation unit
        all_functionss = [
            compilation_unit.functions for compilation_unit in self.slither.compilation_units
        ]
        all_functions = [item for sublist in all_functionss for item in sublist]
        all_functions_as_dict = {
            function.canonical_name: function for function in all_functions
        }

        all_contracts_call_graph = _process_functions(all_functions_as_dict.values())
        print(nx.info(all_contracts_call_graph))
        print(all_contracts_call_graph.nodes(data=True))
        print(all_contracts_call_graph.edges(data=True))
    
        # Dump call graph to gpickle and DOT file
        nx.nx_agraph.write_dot(all_contracts_call_graph, all_contracts_filename + '.dot')
        print('Dumped:', all_contracts_filename + '.dot')
        nx.write_gpickle(all_contracts_call_graph, all_contracts_filename + '.gpickle')
        print('Dumped:', all_contracts_filename + '.gpickle')

        for derived_contract in self.slither.contracts_derived:
            derived_output_filename = f"{filename}.{derived_contract.name}.call-graph"
            derived_contract_call_graph = _process_functions(derived_contract.functions)

            # Dump call graph to gpickle and DOT file
            nx.nx_agraph.write_dot(derived_contract_call_graph, derived_output_filename + '.dot')
            print('Dumped:', derived_output_filename + '.dot')
            nx.write_gpickle(derived_contract_call_graph, derived_output_filename + '.gpickle')
            print('Dumped:', derived_output_filename + '.gpickle')

# Source code smart contract file
# fn = 'data/reentrancy/source_code/simple_dao'
# fn = 'data/reentrancy/source_code/Bank'
# fn = 'data/reentrancy/source_code/cross-function-reentrancy'
# fn = 'data/reentrancy/source_code/6881'
# fn = 'data/reentrancy/source_code/22247'
fn = 'data/reentrancy/source_code/PrivateBank'

slither = Slither(fn + '.sol')

printer = GESCPrinters(slither, logger)
printer.output(fn + '-GESC')
