import os
import json

from os.path import join
from shutil import copy
from copy import deepcopy
from re import L
from typing import Pattern
from tqdm import tqdm
import re

import networkx as nx
from slither.slither import Slither
from slither.core.cfg.node import NodeType
from solc import install_solc


pattern =  re.compile(r'\d.\d.\d+')
def get_solc_version(source):
    with open(source, 'r') as f:
        line = f.readline()
        while line:
            if 'pragma solidity' in line:
                if len(pattern.findall(line)) > 0:
                    return pattern.findall(line)[0]
                else:
                    return '0.4.25'
            line = f.readline()
    return '0.4.25'


def get_node_info(node, list_vulnerabilities_info_in_sc):
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

    node_source_code_lines = node.source_mapping['lines']
    node_info_vulnerabilities = get_vulnerabilities_of_node_by_source_code_line(node_source_code_lines, list_vulnerabilities_info_in_sc)
    
    return node_label, node_type, node_expression, node_irs, node_info_vulnerabilities, node_source_code_lines

def get_vulnerabilities(file_name_sc, vulnerabilities):
    list_vulnerability_in_sc = None
    if vulnerabilities is not None:
        for vul_item in vulnerabilities:
            if file_name_sc == vul_item['name']:
                list_vulnerability_in_sc = vul_item['vulnerabilities']
            
    return list_vulnerability_in_sc

def get_vulnerabilities_of_node_by_source_code_line(source_code_lines, list_vul_info_sc):
    if list_vul_info_sc is not None:
        list_vulnerability = []
        for vul_info_sc in list_vul_info_sc:
            vulnerabilities_lines = vul_info_sc['lines']
            # for source_code_line in source_code_lines:
            #     for vulnerabilities_line in vulnerabilities_lines:
            #         if source_code_line == vulnerabilities_line:
            #             list_vulnerability.append(vul_info_sc)
            interset_lines = set(vulnerabilities_lines).intersection(set(source_code_lines))
            if len(interset_lines) > 0:
                list_vulnerability.append(vul_info_sc)

    else:
        list_vulnerability = None
    
    if list_vulnerability is None or len(list_vulnerability) == 0:
        node_info_vulnerabilities = None
    else:
        node_info_vulnerabilities = list_vulnerability

    return node_info_vulnerabilities

def compress_full_smart_contracts(smart_contracts, input_graph, output, vulnerabilities=None):
    full_graph = None
    if input_graph is not None:
        full_graph = nx.read_gpickle(input_graph)
    count = 0
    for sc in tqdm(smart_contracts):
        sc_version = get_solc_version(sc)
        print(f'{sc} - {sc_version}')
        solc_compiler = f'.solc-select/artifacts/solc-{sc_version}'
        if not os.path.exists(solc_compiler):
            solc_compiler = f'.solc-select/artifacts/solc-0.4.25'
        file_name_sc = sc.split('/')[-1:][0]
        bug_type = sc.split('/')[-2]
        try:
            slither = Slither(sc, solc=solc_compiler)
            count += 1
        except Exception as e:
            print('exception ', e)
            continue

        list_vul_info_sc = get_vulnerabilities(file_name_sc, vulnerabilities)

        print(file_name_sc, list_vul_info_sc)

        merge_contract_graph = None
        for contract in slither.contracts:
            merged_graph = None
            for idx, function in enumerate(contract.functions + contract.modifiers):  

                nx_g = nx.MultiDiGraph()
                for nidx, node in enumerate(function.nodes):             
                    node_label, node_type, node_expression, node_irs, node_info_vulnerabilities, node_source_code_lines = get_node_info(node, list_vul_info_sc)
                    
                    nx_g.add_node(node.node_id, label=node_label,
                                  node_type=node_type, node_expression=node_expression, node_irs=node_irs,
                                  node_info_vulnerabilities=node_info_vulnerabilities,
                                  node_source_code_lines=node_source_code_lines,
                                  function_fullname=function.full_name, contract_name=contract.name, source_file=file_name_sc)
                    
                    if node.type in [NodeType.IF, NodeType.IFLOOP]:
                        true_node = node.son_true
                        if true_node:
                            if true_node.node_id not in nx_g.nodes():
                                node_label, node_type, node_expression, node_irs, node_info_vulnerabilities, node_source_code_lines = get_node_info(true_node, list_vul_info_sc)
                                nx_g.add_node(true_node.node_id, label=node_label,
                                              node_type=node_type, node_expression=node_expression, node_irs=node_irs,
                                              node_info_vulnerabilities=node_info_vulnerabilities,
                                              node_source_code_lines=node_source_code_lines,
                                              function_fullname=function.full_name, contract_name=contract.name, source_file=file_name_sc)
                            nx_g.add_edge(node.node_id, true_node.node_id, edge_type='if_true', label='True')
                        
                        
                        false_node = node.son_false
                        if false_node:
                            if false_node.node_id not in nx_g.nodes():
                                node_label, node_type, node_expression, node_irs, node_info_vulnerabilities, node_source_code_lines = get_node_info(false_node, list_vul_info_sc)
                                nx_g.add_node(false_node.node_id, label=node_label,
                                              node_type=node_type, node_expression=node_expression, node_irs=node_irs,
                                              node_info_vulnerabilities=node_info_vulnerabilities,
                                              node_source_code_lines=node_source_code_lines,
                                              function_fullname=function.full_name, contract_name=contract.name, source_file=file_name_sc)
                            nx_g.add_edge(node.node_id, false_node.node_id, edge_type='if_false', label='False')
                            
                    else:
                        for son_node in node.sons:
                            if son_node:
                                if son_node.node_id not in nx_g.nodes():
                                    node_label, node_type, node_expression, node_irs, node_info_vulnerabilities, node_source_code_lines = get_node_info(son_node, list_vul_info_sc)
                                    nx_g.add_node(son_node.node_id, label=node_label,
                                                  node_type=node_type, node_expression=node_expression, node_irs=node_irs,
                                                  node_info_vulnerabilities=node_info_vulnerabilities,
                                                  node_source_code_lines=node_source_code_lines,
                                                  function_fullname=function.full_name, contract_name=contract.name, source_file=file_name_sc)
                                nx_g.add_edge(node.node_id, son_node.node_id, edge_type='next', label='Next')

                nx_graph = nx_g
                # add FUNCTION_NAME node
                node_function_name = file_name_sc + '_' + contract.name + '_' + function.full_name
                node_function_source_code_lines = function.source_mapping['lines']
                node_function_info_vulnerabilities = get_vulnerabilities_of_node_by_source_code_line(node_function_source_code_lines, list_vul_info_sc)
                nx_graph.add_node(node_function_name, label=node_function_name,
                                  node_type='FUNCTION_NAME', node_expression=None, node_irs=None,
                                  node_info_vulnerabilities=node_function_info_vulnerabilities,
                                  node_source_code_lines=node_function_source_code_lines,
                                  function_fullname=function.full_name, contract_name=contract.name, source_file=file_name_sc)
                
                if 0 in nx_graph.nodes():
                    nx_graph.add_edge(node_function_name, 0, edge_type='next', label='Next')

                nx_graph = nx.relabel_nodes(nx_graph, lambda x: contract.name + '_' + function.full_name + '_' + str(x), copy=False)

                if merged_graph is None:
                    merged_graph = deepcopy(nx_graph)
                else:
                    merged_graph = nx.disjoint_union(merged_graph, nx_graph)

            if merge_contract_graph is None:
                merge_contract_graph = deepcopy(merged_graph)
            elif merged_graph is not None:
                merge_contract_graph = nx.disjoint_union(merge_contract_graph, merged_graph)
        
        if full_graph is None:
            full_graph = deepcopy(merge_contract_graph)
        elif merge_contract_graph is not None:
            full_graph = nx.disjoint_union(full_graph, merge_contract_graph)

    # for node, node_data in full_graph.nodes(data=True):
    #     if node_data['node_info_vulnerabilities'] is not None:
    #         print('Node has vulnerabilities:', node, node_data)
    print(f'{count}/{len(smart_contracts)}')
    # nx.nx_agraph.write_dot(full_graph, output.replace('.gpickle', '.dot'))
    nx.write_gpickle(full_graph, output)

def merge_data_from_vulnerabilities_json_files(list_vulnerabilities_json_files):
    result = list()
    for f1 in list_vulnerabilities_json_files:
        with open(f1, 'r') as infile:
            result.extend(json.load(infile))

    return result

def check_extract_graph(source_path):
    sc_version = get_solc_version(source_path)
    solc_compiler = f'~/.solc-select/artifacts/solc-{sc_version}'
    if not os.path.exists(solc_compiler):
        solc_compiler = f'~/.solc-select/artifacts/solc-0.4.25'
    try:
        slither = Slither(source_path, solc=solc_compiler)
        return 1
    except Exception as e:
        return 0


def extract_graph(source_path, output, vulnerabilities=None):
    sc_version = get_solc_version(source_path)
    solc_compiler = f'~/.solc-select/artifacts/solc-{sc_version}'
    if not os.path.exists(solc_compiler):
        solc_compiler = f'~/.solc-select/artifacts/solc-0.4.25'
    file_name_sc = source_path.split('/')[-1]
    try:
        slither = Slither(source_path, solc=solc_compiler)
    except Exception as e:
        print('exception ', e)
        return 0

    list_vul_info_sc = get_vulnerabilities(file_name_sc, vulnerabilities)

    merge_contract_graph = None
    for contract in slither.contracts:
        merged_graph = None
        for idx, function in enumerate(contract.functions + contract.modifiers):  

            nx_g = nx.MultiDiGraph()
            for nidx, node in enumerate(function.nodes):             
                node_label, node_type, node_expression, node_irs, node_info_vulnerabilities, node_source_code_lines = get_node_info(node, list_vul_info_sc)
                
                nx_g.add_node(node.node_id, label=node_label,
                                node_type=node_type, node_expression=node_expression, node_irs=node_irs,
                                node_info_vulnerabilities=node_info_vulnerabilities,
                                node_source_code_lines=node_source_code_lines,
                                function_fullname=function.full_name, contract_name=contract.name, source_file=file_name_sc)
                
                if node.type in [NodeType.IF, NodeType.IFLOOP]:
                    true_node = node.son_true
                    if true_node:
                        if true_node.node_id not in nx_g.nodes():
                            node_label, node_type, node_expression, node_irs, node_info_vulnerabilities, node_source_code_lines = get_node_info(true_node, list_vul_info_sc)
                            nx_g.add_node(true_node.node_id, label=node_label,
                                            node_type=node_type, node_expression=node_expression, node_irs=node_irs,
                                            node_info_vulnerabilities=node_info_vulnerabilities,
                                            node_source_code_lines=node_source_code_lines,
                                            function_fullname=function.full_name, contract_name=contract.name, source_file=file_name_sc)
                        nx_g.add_edge(node.node_id, true_node.node_id, edge_type='if_true', label='True')
                    
                    
                    false_node = node.son_false
                    if false_node:
                        if false_node.node_id not in nx_g.nodes():
                            node_label, node_type, node_expression, node_irs, node_info_vulnerabilities, node_source_code_lines = get_node_info(false_node, list_vul_info_sc)
                            nx_g.add_node(false_node.node_id, label=node_label,
                                            node_type=node_type, node_expression=node_expression, node_irs=node_irs,
                                            node_info_vulnerabilities=node_info_vulnerabilities,
                                            node_source_code_lines=node_source_code_lines,
                                            function_fullname=function.full_name, contract_name=contract.name, source_file=file_name_sc)
                        nx_g.add_edge(node.node_id, false_node.node_id, edge_type='if_false', label='False')
                        
                else:
                    for son_node in node.sons:
                        if son_node:
                            if son_node.node_id not in nx_g.nodes():
                                node_label, node_type, node_expression, node_irs, node_info_vulnerabilities, node_source_code_lines = get_node_info(son_node, list_vul_info_sc)
                                nx_g.add_node(son_node.node_id, label=node_label,
                                                node_type=node_type, node_expression=node_expression, node_irs=node_irs,
                                                node_info_vulnerabilities=node_info_vulnerabilities,
                                                node_source_code_lines=node_source_code_lines,
                                                function_fullname=function.full_name, contract_name=contract.name, source_file=file_name_sc)
                            nx_g.add_edge(node.node_id, son_node.node_id, edge_type='next', label='Next')

            nx_graph = nx_g
            # add FUNCTION_NAME node
            node_function_name = file_name_sc + '_' + contract.name + '_' + function.full_name
            node_function_source_code_lines = function.source_mapping['lines']
            node_function_info_vulnerabilities = get_vulnerabilities_of_node_by_source_code_line(node_function_source_code_lines, list_vul_info_sc)
            nx_graph.add_node(node_function_name, label=node_function_name,
                                node_type='FUNCTION_NAME', node_expression=None, node_irs=None,
                                node_info_vulnerabilities=node_function_info_vulnerabilities,
                                node_source_code_lines=node_function_source_code_lines,
                                function_fullname=function.full_name, contract_name=contract.name, source_file=file_name_sc)
            
            if 0 in nx_graph.nodes():
                nx_graph.add_edge(node_function_name, 0, edge_type='next', label='Next')

            nx_graph = nx.relabel_nodes(nx_graph, lambda x: contract.name + '_' + function.full_name + '_' + str(x), copy=False)

            if merged_graph is None:
                merged_graph = deepcopy(nx_graph)
            else:
                merged_graph = nx.disjoint_union(merged_graph, nx_graph)

        if merge_contract_graph is None:
            merge_contract_graph = deepcopy(merged_graph)
        elif merged_graph is not None:
            merge_contract_graph = nx.disjoint_union(merge_contract_graph, merged_graph)
    
    nx.write_gpickle(merge_contract_graph, join(output, file_name_sc))
    return 1

if __name__ == '__main__':

    # smart_contract_path = 'data/clean_71_buggy_curated_0'
    # input_graph = None
    # output_path = 'data/clean_71_buggy_curated_0/cfg_compress_graphs.gpickle'
    # smart_contracts = [join(smart_contract_path, f) for f in os.listdir(smart_contract_path) if f.endswith('.sol')]

    # data_vulnerabilities = None
    # list_vulnerabilities_json_files = [
    #     'data/solidifi_buggy_contracts/reentrancy/vulnerabilities.json',
    #     # 'data/solidifi_buggy_contracts/access_control/vulnerabilities.json',
    #     'data/smartbug-dataset/vulnerabilities.json']
    
    # data_vulnerabilities = merge_data_from_vulnerabilities_json_files(list_vulnerabilities_json_files)
    
    # compress_full_smart_contracts(smart_contracts, input_graph, output_path, vulnerabilities=data_vulnerabilities)

    ROOT = './experiments/ge-sc-data/source_code'
    bug_type = {'access_control': 57, 'arithmetic': 60, 'denial_of_service': 46,
              'front_running': 44, 'reentrancy': 71, 'time_manipulation': 50, 
              'unchecked_low_level_calls': 95}
    for bug, counter in bug_type.items():
        # source = f'{ROOT}/{bug}/buggy_curated'
        # output = f'{ROOT}/{bug}/buggy_curated/cfg_compressed_graphs.gpickle'
        source = f'{ROOT}/{bug}/curated'
        output = f'{ROOT}/{bug}/curated/cfg_compressed_graphs.gpickle'
        smart_contracts = [join(source, f) for f in os.listdir(source) if f.endswith('.sol')]
        data_vulnerabilities = None
        list_vulnerabilities_json_files = ['data/solidifi_buggy_contracts/reentrancy/vulnerabilities.json',
        # 'data/solidifi_buggy_contracts/access_control/vulnerabilities.json',
        'data/smartbug-dataset/vulnerabilities.json']
        data_vulnerabilities = merge_data_from_vulnerabilities_json_files(list_vulnerabilities_json_files)
        compress_full_smart_contracts(smart_contracts, None, output, vulnerabilities=data_vulnerabilities)