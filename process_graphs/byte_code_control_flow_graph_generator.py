import os
import sys
import subprocess
import collections
from os.path import join
from copy import deepcopy
from shutil import copy

import re
from textwrap import indent
import dgl
import json
import torch
import networkx as nx
from tqdm import tqdm
from slither.slither import Slither
from numpy import source


EDGE_DICT = {('None', 'None', 'None'): '0', ('None', 'None', 'orange'): '1', ('Msquare', 'None', 'gold'): '2', ('None', 'None', 'lemonchiffon'): '3', ('Msquare', 'crimson', 'crimson'): '4', ('None', 'None', 'crimson'): '5', ('Msquare', 'crimson', 'None'): '6', ('Msquare', 'crimson', 'lemonchiffon'): '7'}
DRY_RUNS = 0


def get_solc_version(source):
    PATTERN = re.compile(r"pragma solidity\s*(?:\^|>=|<=)?\s*(\d+\.\d+\.\d+)")
    solc_select = '/home/minhnn/.solc-select/artifacts'
    solc_version = [v.split('-')[-1] for v in os.listdir(solc_select)]
    with open(join(source), encoding="utf8") as file_desc:
        buf = file_desc.read()
    version = PATTERN.findall(buf)
    version = '0.4.25' if len(version) == 0 else version[0]
    if version not in solc_version:
        if version.startswith('0.4.'):
            solc_path = join(solc_select, 'solc-' + '0.4.25')
        elif version.startswith('0.5.'):
            solc_path = join(solc_select, 'solc-' + '0.5.11')
        else:
            solc_path = join(solc_select, 'solc-' + '0.8.6')
    else:
        solc_path = join(solc_select, 'solc-' + version)
    return solc_path


def creat_node_label(g,d):
    nodes = g.nodes()
    int2label = {}
    for idx,node_idx in enumerate(nodes):
        obj = g._node[node_idx]
        if 'shape' not in obj:
            shape = 'None'
        else:
            shape = obj['shape']
        if 'color' not in obj:
            color = 'None'
        else:
            color = obj['color']
        if 'fillcolor' not in obj:
            fillcolor = 'None'
        else:
            fillcolor = obj['fillcolor']
        t = (shape,color,fillcolor)
        node_type = d[t]
        int2label[node_idx] = node_type
    return int2label


def creat_edge_label(g):
    edgeLabel = {}
    edgedata = g.edges.data()
    for u,v,fe in edgedata:
        edgeLabel[(u, v, 0)] = {'edge_type': 'CF'}
    return edgeLabel


def createLabel(g, d):
        nodes = g.nodes()
        int2label = {}
        startwith, endwith = 0,1
        node_info = {}
        edgeLabel = {}
        for idx,node_idx in enumerate(nodes):
            obj = g._node[node_idx]
            if 'shape' not in obj:
                shape = 'None'
            else:
                shape = obj['shape']
            if 'color' not in obj:
                color = 'None'
            else:
                color = obj['color']
            if 'fillcolor' not in obj:
                fillcolor = 'None'
            else:
                fillcolor = obj['fillcolor']
            t = (shape,color,fillcolor)
            node_type = d[t]
            # nodetype without number
            # psb_node_type = g._node[node_idx]['label'].splitlines()[0]
            # colonPos = psb_node_type.find(':') + 1
            # node_type = psb_node_type[colonPos:]
            int2label[node_idx] = node_type
            info = obj['label'].split(':')
            node_info[node_idx] = [info[1][1:9],info[-1][1:-3]]

        for u,v in g.edges():
            if node_info[u][endwith] == 'JUMPI':
                if node_info[v][startwith] == 'JUMPDEST':
                    edgeLabel[(u,v,0)] = {'edge_type': 'True'}
                else:
                    edgeLabel[(u,v,0)] = {'edge_type': 'Else'}
            else:
                edgeLabel[(u,v,0)] = {'edge_type': 'CF'}
        return int2label, edgeLabel


def dot2gpickle(dot_file, gpickle_file):
    source_file = gpickle_file.split('/')[-1]
    nx_g = nx.drawing.nx_pydot.read_dot(dot_file)
    # node_lables = creat_node_label(nx_g, EDGE_DICT)
    # edge_labels = creat_edge_label(nx_g)
    node_lables, edge_labels = createLabel(nx_g, EDGE_DICT)
    nx.set_node_attributes(nx_g, node_lables, name='node_type')
    nx.set_node_attributes(nx_g, source_file, name='source_file')
    nx.set_edge_attributes(nx_g, edge_labels)
    nx.write_gpickle(nx_g, gpickle_file)


def merge_byte_code_cfg(source_path, graph_list, output):
    merged_graph = None
    for graph in graph_list:
        nx_graph = nx.read_gpickle(join(source_path, graph))
        if merged_graph is None:
            merged_graph = deepcopy(nx_graph)
        else:
            merged_graph = nx.disjoint_union(merged_graph, nx_graph)
    nx.write_gpickle(merged_graph, output)


def travelsalDir(filepath):
    count_0, count_1 = 0, 0
    with open(filepath,'r') as f:
        load_dict = json.load(f)
    location = './data/bytecode_cfg_set/'

    pathDict = {}
    labels = collections.defaultdict(int)
    name_list = []
    for name in load_dict:
        if load_dict[name] == 0 and count_0 < 100:
            count_0 += 1
            labels[name] = torch.LongTensor([load_dict[name]])
            pathDict[name] = location + name + '.dot'
            name_list.append(name)
        elif load_dict[name] == 1 and count_1 < 100:
            count_1 += 1
            labels[name] = torch.LongTensor([load_dict[name]])
            pathDict[name] = location + name + '.dot'
            name_list.append(name)

    return pathDict, labels, name_list


def format_label(label_file, output):
    with open(label_file, 'r') as f:
        labels = json.load(f)
    new_labels = []
    for contract, target in labels.items():
        new_labels.append({'targets': target, 'contract_name': contract+'.sol'})
    with open(output, 'w') as f:
        json.dump(new_labels, f)


def forencis_gpickle(graph_path):
    nx_graph = nx.read_gpickle(graph_path)
    print(nx_graph.is_multigraph())
    print(nx_graph.nodes()[0])
    for idx, node in nx_graph.nodes(data=True):
        print(idx, node['node_type'])
    print(list(nx_graph.edges.data())[0])
    # print(nx_graph.edges.data())
    # for source, target, data in list(nx_graph.edges(data=True)):
        # print(source, target, data)


def generate_crytic_evm(sourcecode_path, output):
    os.makedirs(output, exist_ok=True)
    PATTERN = re.compile(r"pragma solidity\s*(?:\^|>=|<=)?\s*(\d+\.\d+\.\d+)")
    contracts = [f for f in os.listdir(sourcecode_path) if f.endswith('.sol')]
    solc_select = '/home/minhnn/.solc-select/artifacts'
    solc_version = [v.split('-')[-1] for v in os.listdir(solc_select)]
    for sc in contracts:
            sc_path = join(sourcecode_path, sc)
            with open(join(sourcecode_path, sc), encoding="utf8") as file_desc:
                buf = file_desc.read()
            version = PATTERN.findall(buf)
            version = '0.4.25' if len(version) == 0 else version[0]
            if version not in solc_version:
                if version.startswith('0.4.'):
                    solc_path = join(solc_select, 'solc-' + '0.4.25')
                elif version.startswith('0.5.'):
                    solc_path = join(solc_select, 'solc-' + '0.5.11')
                else:
                    solc_path = join(solc_select, 'solc-' + '0.8.6')
            else:
                solc_path = join(solc_select, 'solc-' + version)
            subprocess.run(['crytic-compile', sc_path, '--export-format', 'standard', '--export-dir', output, '--solc-solcs-bin', solc_path])


def generate_evm(crytic_evm_path, creation_output, runtime_output):
    os.makedirs(creation_output, exist_ok=True)
    os.makedirs(runtime_output, exist_ok=True)
    byte_codes = [f for f in os.listdir(crytic_evm_path) if f.endswith('.json')]
    for bc in byte_codes:
        with open(join(crytic_evm_path, bc), 'r') as f:
            annotation = json.load(f)
        details = list(annotation['compilation_units'].values())[0]['contracts']
        for sc in details.keys():
            creation_code = details[sc]['bin']
            runtime_code = details[sc]['bin-runtime']
            if len(creation_code) == 0:
                assert len(creation_code) == len(runtime_code)
                continue
            creation_file_name = bc.replace('.sol.json', f'-{sc}.evm')
            runtime_file_name = bc.replace('.sol.json', f'-{sc}.evm')
            with open(join(creation_output, creation_file_name), 'w') as f:
                f.write(creation_code)
            with open(join(runtime_output, runtime_file_name), 'w') as f:
                f.write(runtime_code)


def generate_graph_from_evm(evm_path, output, evm_type):
    os.makedirs(output, exist_ok=True)
    code_type = '-c' if evm_type == 'creation' else '-r'
    evm_files = [f for f in os.listdir(evm_path) if f.endswith('.evm')]
    for evm in evm_files:
        evm_file = join(evm_path, evm)
        subprocess.run(['java', '-jar', 'EtherSolve.jar', code_type, '-d', '-o', join(output, evm.replace('.evm', '.dot')), evm_file])


## Dump to file to save time
def get_contract_code_line(source_files, output=None):
    # Init slither
    solc_compiler = get_solc_version(source_files)
    slither = Slither(source_files, solc=solc_compiler)
    # Get the contract
    contract_lines = {}
    for contract in slither.contracts:
        lines = contract.source_mapping['lines']
        contract_lines[contract.name] = {'start': min(lines), 'end': max(lines)}
    if output:
        with open(output, 'w') as f:
            json.dump(contract_lines, f, indent=4)
    return contract_lines


def create_source_code_category(source_path, clean_source_files, output):
    contract_files = [f for f in os.listdir(source_path) if f.endswith('.sol')]
    source_category = {'curated': [], 'solidifi': [], 'clean': []}
    for s in contract_files:
        if s in clean_source_files:
            source_category['clean'].append(s)
        elif s.startswith('buggy_'):
            source_category['solidifi'].append(s)
        else:
            source_category['curated'].append(s)
    contract_detail = {'curated': {}, 'solidifi': {}, 'clean': {}}
    for cate, contracts in source_category.items():
        for sc in contracts:
            contract_lines = get_contract_code_line(join(source_path, sc))
            contract_detail[cate][sc] = contract_lines
    if output:
        with open(output, 'w') as f:
            json.dump(contract_detail, f, indent=4)
    return contract_detail


def generate_evm_annotations(source_code_category, curated_annotaion, solidifi_annotation, bug_type, output):
    annotations = []
    for cat, source_files in source_code_category.items():
        for source, contracts in source_files.items():
            for sc, loc in contracts.items():
                contract_annotation = {}
                contract_annotation['contract_name'] = source.replace('.sol', f'-{sc}.sol')
                if cat == 'curated':
                    contract_annotation['targets'] = int(len(set(range(loc['start'], loc['end'])) & set(curated_annotaion[source][bug_type])) > 0)
                elif cat == 'solidifi':
                    contract_annotation['targets'] = int(len(set(range(loc['start'], loc['end'])) & set(solidifi_annotation[source][bug_type])) > 0)
                else:
                    contract_annotation['targets'] = 0
                annotations.append(contract_annotation)
    with open(output, 'w') as f:
        json.dump(annotations, f, indent=4)


def _convert_curated_annotation_to_dict(curated_annotation, output=None):
    with open(curated_annotation, 'r') as f:
        annotation = json.load(f)
    curated_dict = {}
    for anno in annotation:
        anno_dict = {}
        bug_type = None
        for vul in anno['vulnerabilities']:
            bug_type = vul['category']
            if bug_type not in anno_dict:
                anno_dict[bug_type] = vul['lines']
            else:
                anno_dict[bug_type] += vul['lines']
        if bug_type is not None and anno['name'] not in curated_dict:
            curated_dict[anno['name']] = anno_dict
        else:
            curated_dict[anno['name']].update(anno_dict)
    if output:
        with open(output, 'w') as f:
            json.dump(curated_dict, f, indent=4)
    return curated_dict


def _convert_solidifi_annotation_to_dict(solidifi_annotation, output=None):
    with open(solidifi_annotation, 'r') as f:
        annotation = json.load(f)
    solidifi_dict = {}
    for anno in annotation:
        anno_dict = {}
        for vul in anno['vulnerabilities']:
            bug_type = vul['category']
            if bug_type not in anno_dict:
                anno_dict[bug_type] = vul['lines']
            else:
                anno_dict[bug_type] += vul['lines']
        solidifi_dict[anno['name']] = anno_dict
    if output:
        with open(output, 'w') as f:
            json.dump(solidifi_dict, f, indent=4)
    return solidifi_dict


if __name__ == '__main__':
    bug_type = {'access_control': 57, 'arithmetic': 60, 'denial_of_service': 46,
              'front_running': 44, 'reentrancy': 71, 'time_manipulation': 50, 
              'unchecked_low_level_calls': 95}

    # # Forencis gpickle graph
    # source_compressed_graph = './experiments/ge-sc-data/source_code/access_control/clean_57_buggy_curated_0/cfg_compressed_graphs.gpickle'
    # # forencis_gpickle(source_compressed_graph)
    # forencis_gpickle(compressed_graph)
    
    # # Create annotation file
    # clean_source_code = './ge-sc-data/smartbugs-wild-clean-contracts'
    # clean_source_files = [f for f in os.listdir(clean_source_code) if f.endswith('.sol')]
    # curated_annotation_path = './data/smartbug-dataset/vulnerabilities.json'
    # # CURATED_DICT = _convert_curated_annotation_to_dict(curated_annotation_path, join('./experiments/ge-sc-data/source_code', 'curated_labels.json'))
    # with open(join('./experiments/ge-sc-data/source_code', 'curated_labels.json'), 'r') as f:
    #     CURATED_DICT = json.load(f)
    # for bug, count in bug_type.items():
    #     sourcecode_path = f'./experiments/ge-sc-data/source_code/{bug}/clean_{count}_buggy_curated_0/'
    #     output_label = f'./experiments/ge-sc-data/byte_code/smartbugs/contract_labels/{bug}'
    #     os.makedirs(output_label, exist_ok=True)
    #     output_label = join(output_label, 'contract_labels.json')
    #     # output_label = join(sourcecode_path, 'contract_labels.json')
    #     solidifi_annotation_path = f'./data/solidifi_buggy_contracts/{bug}/vulnerabilities.json'
    #     # SOLIDIFI_DICT = _convert_solidifi_annotation_to_dict(solidifi_annotation_path, join(sourcecode_path, 'solidifi_labels.json'))
    #     with open(join(sourcecode_path, 'solidifi_labels.json'), 'r') as f:
    #         SOLIDIFI_DICT = json.load(f)
    #     # source_code_category = create_source_code_category(sourcecode_path, clean_source_files, join(sourcecode_path, 'contract_details.json'))
    #     source_code_category_path = f'./experiments/ge-sc-data/source_code/{bug}/clean_{count}_buggy_curated_0/source_code_category.json'
    #     with open(source_code_category_path, 'r') as f:
    #         source_code_category = json.load(f)
    #     generate_evm_annotations(source_code_category, CURATED_DICT, SOLIDIFI_DICT, bug, output_label)


    # get_contract_code_line('./experiments/ge-sc-data/source_code/access_control/clean_57_buggy_curated_0/0x0a5dc2204dfc6082ef3bbcfc3a468f16318c4168.sol')

    # # Generate crytic evm files
    # for bug, counter in bug_type.items():
    #     sourcecode_path = f'./experiments/ge-sc-data/source_code/{bug}/clean_{counter}_buggy_curated_0'
    #     output = f'./experiments/ge-sc-data/byte_code/smartbugs/crytic_evm/{bug}/clean_{counter}_buggy_curated_0'
    #     generate_crytic_evm(sourcecode_path, output)

    # # Generate creation & runtime evm files
    # for bug, counter in bug_type.items():
    #     crytic_evm_path = f'./experiments/ge-sc-data/byte_code/smartbugs/crytic_evm/{bug}/clean_{counter}_buggy_curated_0'
    #     creation_output = f'./experiments/ge-sc-data/byte_code/smartbugs/creation/evm/{bug}/clean_{counter}_buggy_curated_0'
    #     runtime_output = f'./experiments/ge-sc-data/byte_code/smartbugs/runtime/evm/{bug}/clean_{counter}_buggy_curated_0'
    #     generate_evm(crytic_evm_path, creation_output, runtime_output)

    # # Generate graph from evm files by EtherSolve
    # for bug, counter in bug_type.items():
    #     creation_path = f'./experiments/ge-sc-data/byte_code/smartbugs/creation/evm/{bug}/clean_{counter}_buggy_curated_0'
    #     creation_output = f'./experiments/ge-sc-data/byte_code/smartbugs/creation/graphs/{bug}/clean_{counter}_buggy_curated_0'
    #     generate_graph_from_evm(creation_path, creation_output, 'creation')
    #     runtime_path = f'./experiments/ge-sc-data/byte_code/smartbugs/runtime/evm/{bug}/clean_{counter}_buggy_curated_0'
    #     runtime_output = f'./experiments/ge-sc-data/byte_code/smartbugs/runtime/graphs/{bug}/clean_{counter}_buggy_curated_0'
    #     generate_graph_from_evm(runtime_path, runtime_output, 'runtime')        


    # # Convert dot to gpickle
    # for bug, counter in bug_type.items():
    #     creation_graph_path = f'./experiments/ge-sc-data/byte_code/smartbugs/creation/graphs/{bug}/clean_{counter}_buggy_curated_0'
    #     runtime_graph_path = f'./experiments/ge-sc-data/byte_code/smartbugs/runtime/graphs/{bug}/clean_{counter}_buggy_curated_0'
    #     creation_dot_files = [f for f in os.listdir(creation_graph_path) if f.endswith('.dot')]
    #     runtime_dot_files = [f for f in os.listdir(runtime_graph_path) if f.endswith('.dot')]
    #     creation_gpickle_output = f'./experiments/ge-sc-data/byte_code/smartbugs/creation/gpickles/{bug}/clean_{counter}_buggy_curated_0'
    #     runtime_gpickle_output = f'./experiments/ge-sc-data/byte_code/smartbugs/runtime/gpickles/{bug}/clean_{counter}_buggy_curated_0'
    #     os.makedirs(creation_gpickle_output, exist_ok=True)
    #     os.makedirs(runtime_gpickle_output, exist_ok=True)
    #     for dot in creation_dot_files:
    #         dot2gpickle(join(creation_graph_path, dot), join(creation_gpickle_output, dot.replace('.dot', '.gpickle')))
    #     for dot in runtime_dot_files:
    #         dot2gpickle(join(runtime_graph_path, dot), join(runtime_gpickle_output, dot.replace('.dot', '.gpickle')))

    # # Filter dataset
    # HAVE_CLEAN = True
    # for bug, count in bug_type.items():
    #     source_code_category_path = f'./experiments/ge-sc-data/source_code/{bug}/clean_{count}_buggy_curated_0/source_code_category.json'
    #     creation_gpickle_path = f'./experiments/ge-sc-data/byte_code/smartbugs/creation/gpickles/{bug}/clean_{count}_buggy_curated_0'
    #     creation_gpickle_files = [f for f in os.listdir(creation_gpickle_path) if f.endswith('.gpickle')]
    #     runtime_gpickle_path = f'./experiments/ge-sc-data/byte_code/smartbugs/runtime/gpickles/{bug}/clean_{count}_buggy_curated_0'
    #     runtime_gpickle_files = [f for f in os.listdir(runtime_gpickle_path) if f.endswith('.gpickle')]
    #     with open(source_code_category_path, 'r') as f:
    #             source_code_category = json.load(f)
    #     curated_files = list(source_code_category['curated'].keys())
    #     solidifi_files = list(source_code_category['solidifi'].keys())
    #     clean_files = list(source_code_category['clean'].keys())
    #     annotation_path = f'./experiments/ge-sc-data/byte_code/smartbugs/contract_labels/{bug}/contract_labels.json'
    #     with open(annotation_path, 'r') as f:
    #         contract_labels = json.load(f)
    #     creation_balance_dataset = []
    #     runtime_balance_dataset = []
    #     creation_buggy_count = 0
    #     runtime_buggy_count = 0
    #     for contract in contract_labels:
    #         source_name = contract['contract_name'].split('-')[0] + '.sol'
    #         gpickle_name = contract['contract_name'].replace('.sol', '.gpickle')
    #         if source_name in curated_files or source_name in solidifi_files:
    #             if gpickle_name in creation_gpickle_files:
    #                 creation_balance_dataset.append(contract)
    #                 creation_buggy_count += contract['targets']
    #             if gpickle_name in runtime_gpickle_files:
    #                 runtime_balance_dataset.append(contract)
    #                 runtime_buggy_count += contract['targets']

    #     if HAVE_CLEAN:
    #         for contract in contract_labels:
    #             if creation_buggy_count/len(creation_balance_dataset) <= 0.5:
    #                 break
    #             source_name = contract['contract_name'].split('-')[0] + '.sol'
    #             gpickle_name = contract['contract_name'].replace('.sol', '.gpickle')
    #             if source_name in clean_files and gpickle_name in creation_gpickle_files:
    #                 creation_balance_dataset.append(contract)
    #         for contract in contract_labels:
    #             if runtime_buggy_count/len(runtime_balance_dataset) <= 0.5:
    #                 break
    #             source_name = contract['contract_name'].split('-')[0] + '.sol'
    #             gpickle_name = contract['contract_name'].replace('.sol', '.gpickle')
    #             if source_name in clean_files and gpickle_name in runtime_gpickle_files:
    #                 runtime_balance_dataset.append(contract)
    #     creation_balanced_output = f'./experiments/ge-sc-data/byte_code/smartbugs/contract_labels/{bug}/creation_balanced_contract_labels.json'
    #     with open(creation_balanced_output, 'w') as f:
    #         json.dump(creation_balance_dataset, f, indent=4)
    #     runtime_balanced_output = f'./experiments/ge-sc-data/byte_code/smartbugs/contract_labels/{bug}/runtime_balanced_contract_labels.json'
    #     with open(runtime_balanced_output, 'w') as f:
    #         json.dump(runtime_balance_dataset, f, indent=4)
    


    # # Merge gpickle files
    # for bug, count in bug_type.items():
    #     creation_gpickle_path = f'./experiments/ge-sc-data/byte_code/smartbugs/creation/gpickles/{bug}/clean_{count}_buggy_curated_0'
    #     runtime_gpickle_path = f'./experiments/ge-sc-data/byte_code/smartbugs/runtime/gpickles/{bug}/clean_{count}_buggy_curated_0'
    #     creation_output = f'./experiments/ge-sc-data/byte_code/smartbugs/creation/gpickles/{bug}/clean_{count}_buggy_curated_0/compressed_graphs'
    #     runtime_output = f'./experiments/ge-sc-data/byte_code/smartbugs/runtime/gpickles/{bug}/clean_{count}_buggy_curated_0/compressed_graphs'
    #     os.makedirs(creation_output, exist_ok=True)
    #     os.makedirs(runtime_output, exist_ok=True)
    #     # creation_gpickle_files = [f for f in os.listdir(creation_gpickle_path) if f.endswith('.gpickle')]
    #     # runtime_gpickle_files = [f for f in os.listdir(runtime_gpickle_path) if f.endswith('.gpickle')]

    #     # Try to balance dataset
    #     creation_balanced_labels = f'./experiments/ge-sc-data/byte_code/smartbugs/contract_labels/{bug}/creation_balanced_contract_labels.json'
    #     with open(creation_balanced_labels, 'r') as f:
    #         creation_annotations = json.load(f)
    #     creation_gpickle_files = [ann['contract_name'].replace('.sol', '.gpickle') for ann in creation_annotations]
    #     balanced_labels = f'./experiments/ge-sc-data/byte_code/smartbugs/contract_labels/{bug}/balanced_contract_labels.json'
    #     creation_balanced_compressed_graph = join(creation_output, 'creation_balanced_compressed_graphs.gpickle')

    #     runtime_balanced_labels = f'./experiments/ge-sc-data/byte_code/smartbugs/contract_labels/{bug}/runtime_balanced_contract_labels.json'
    #     with open(runtime_balanced_labels, 'r') as f:
    #         runtime_annotations = json.load(f)
    #     runtime_gpickle_files = [ann['contract_name'].replace('.sol', '.gpickle') for ann in runtime_annotations]
    #     balanced_labels = f'./experiments/ge-sc-data/byte_code/smartbugs/contract_labels/{bug}/balanced_contract_labels.json'
    #     runtime_balanced_compressed_graph = join(runtime_output, 'runtime_balanced_compressed_graphs.gpickle')

    #     merge_byte_code_cfg(creation_gpickle_path, creation_gpickle_files, creation_balanced_compressed_graph)
    #     merge_byte_code_cfg(runtime_gpickle_path, runtime_gpickle_files, runtime_balanced_compressed_graph)
    #     copy(creation_balanced_compressed_graph, f'./experiments/ge-sc-data/byte_code/smartbugs/creation/compressed_graphs/{bug}_creation_balanced_cfg_compressed_graphs.gpickle')
    #     copy(runtime_balanced_compressed_graph, f'./experiments/ge-sc-data/byte_code/smartbugs/runtime/compressed_graphs/{bug}_runtime_balanced_cfg_compressed_graphs.gpickle')
    
    # # Convert .sol to .gpickle in annotation files
    # for bug, count in bug_type.items():
    #     annotation_path = f'./experiments/ge-sc-data/byte_code/smartbugs/contract_labels/{bug}'
    #     annotation_list = [f for f in os.listdir(annotation_path) if f.endswith('.json')]
    #     for annotation in annotation_list:
    #         with open(join(annotation_path, annotation), 'r') as f:
    #             content = json.load(f)
    #         new_annotation = [{'contract_name': contract['contract_name'].replace('.sol', '.gpickle'), 'targets':  contract['targets']} for contract in content]
    #         with open(join(annotation_path, annotation), 'w') as f:
    #             json.dump(new_annotation, f, indent=4)