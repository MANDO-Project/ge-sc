import random
import os
import shutil
import json
import time
from os.path import join
from shutil import copy2, move

from process_graphs.call_graph_generator import compress_full_smart_contracts

ROOT = '.'
CLEAN_WILD = f'{ROOT}/ge-sc-data/smartbugs-wild-clean-contracts'
REPEAT = 5

models = ['nodetype', 'metapath2vec', 'gae', 'line', 'node2vec']
bug_list = ['access_control', 'arithmetic', 'denial_of_service',
            'front_running', 'reentrancy', 'time_manipulation', 
            'unchecked_low_level_calls']


def migrate_file(filelist, source, dest):
    for f in filelist:
        # cmd = f'cp {join(source, f)} {join(dest, f)}'
        # os.popen(cmd)
        dst = copy2(join(source, f), join(dest, f))
    # time.sleep(3)


def copy_file(source, dest):
    files = [f for f in os.listdir(source) if f.endswith('.sol')]
    for f in files:
        copy2(join(source, f), join(dest, f))


def collect_dataset(ratio):
    for i in range(REPEAT):
        print(f'Repetitions: {i}')
        for bug in bug_list:
            print(f'Buggy: {bug}')
            source_path = f'{ROOT}/ge-sc-data/node_classification/cg/{bug}/buggy_curated'
            filelist = [f for f in os.listdir(source_path) if f.endswith('.sol')]
            dest_path = f'{ROOT}/ge-sc-data/graph_classification/cg/{bug}/clean_{ratio*len(filelist)}_buggy_curated_{i}'
            if os.path.exists(dest_path):
                try:
                    shutil.rmtree(dest_path)
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))
            os.makedirs(dest_path)
            clean_file_list = [f for f in os.listdir(CLEAN_WILD) if f.endswith('.sol')]
            choiced_clean = list(map(str, random.choices(clean_file_list, k=len(filelist)*ratio)))
            migrate_file(choiced_clean, CLEAN_WILD, dest_path)
            total_files = [f for f in os.listdir(dest_path) if f.endswith('.sol')]
            print(f'{len(total_files)} == {len(filelist)*ratio}')
            apart_files = [f for f in choiced_clean if f not in total_files]
            print('apart files: ', apart_files)
            # assert len(choiced_clean) == len(filelist) * ratio
            # assert len(choiced_clean) == len(total_files)
            migrate_file(filelist, source_path, dest_path)
            total_files = [f for f in os.listdir(dest_path)]
            print(f'{len(choiced_clean)} == {len(filelist)*ratio}')
            print(f'buggy + curated: {len(filelist)}')
            print(f'clean file: {len(choiced_clean)}')
            print(f'total files: {len(total_files)}')
            # assert len(choiced_clean) == len(filelist) * ratio
            # assert len(total_files) == len(choiced_clean) + len(filelist)


def compressed_graph(ratio):
    for i in range(REPEAT):
        for bug in bug_list:
            source_path = f'{ROOT}/ge-sc-data/node_classification/cg/{bug}/buggy_curated'
            filelist = [f for f in os.listdir(source_path) if f.endswith('.sol')]
            dest_path = f'{ROOT}/ge-sc-data/graph_classification/cg/{bug}/clean_{ratio*len(filelist)}_buggy_curated_{i}'
            print(dest_path)
            smart_contracts = [join(dest_path, f) for f in os.listdir(dest_path) if f.endswith('.sol')]
            compress_full_smart_contracts(smart_contracts, join(dest_path, 'compressed_graphs.gpickle'), vulnerabilities=None)


def collect_compressed_graph(ratio):
    for i in range(REPEAT):
        for bug in bug_list:
            storage = f'{ROOT}/ge-sc-data/graph_classification/cg/compressed_graphs/{ratio}_clean_{i}'
            source_path = f'{ROOT}/ge-sc-data/node_classification/cg/{bug}/buggy_curated'
            filelist = [f for f in os.listdir(source_path) if f.endswith('.sol')]
            dest_path = f'{ROOT}/ge-sc-data/graph_classification/cg/{bug}/clean_{ratio*len(filelist)}_buggy_curated_{i}'
            graph_path = join(dest_path, 'compressed_graphs.gpickle')
            os.makedirs(join(storage, bug), exist_ok=True)
            copy2(graph_path, join(storage, bug, 'compressed_graphs.gpickle'))


def make_annotaiton(ratio):
    for i in range(REPEAT):
        for bug in bug_list:
            source_path = f'{ROOT}/ge-sc-data/node_classification/cg/{bug}/buggy_curated'
            source_annotation = f'{ROOT}/ge-sc-data/graph_classification/cg/{bug}/clean_50_buggy_curated/graph_labels.json'
            filelist = [f for f in os.listdir(source_path) if f.endswith('.sol')]
            with open(source_annotation, 'r') as f:
                annotation = json.load(f)
            filted_clean_files = [{'targets': 1, 'contract_name': f} for f in filelist]
            dest_path = f'{ROOT}/ge-sc-data/graph_classification/cg/{bug}/clean_{ratio*len(filelist)}_buggy_curated_{i}'
            smart_contracts = [f for f in os.listdir(dest_path) if f.endswith('.sol')]
            clean_files = list(set(smart_contracts).difference(set(filelist)))
            filted_clean_files += [{'targets': 0, 'contract_name': f} for f in clean_files]
            with open(join(dest_path, 'graph_labels.json'), 'w') as f:
                json.dump(filted_clean_files, f)


if __name__ == '__main__':
    ratio = 1
    # copy_file(CLEAN_WILD, './ge-sc-data/smartbugs-wild-clean-contracts-copied')
    # collect_dataset(1)
    # collect_dataset(2)
    # compressed_graph(1)
    # compressed_graph(2)
    # collect_compressed_graph(1)
    # collect_compressed_graph(2)
    # make_annotaiton(1)
    # make_annotaiton(2)
