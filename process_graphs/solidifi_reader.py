import json
import os
import pandas as pd

from shutil import copyfile

from os.path import join, dirname, basename

# buggy_contract_dir = 'data/solidifi_buggy_contracts/access_control'
# buggy_contract_dir = 'data/solidifi_buggy_contracts/arithmetic'
# buggy_contract_dir = 'data/solidifi_buggy_contracts/denial_of_service'
# buggy_contract_dir = 'data/solidifi_buggy_contracts/front_running'
# buggy_contract_dir = 'data/solidifi_buggy_contracts/reentrancy'
# buggy_contract_dir = 'data/solidifi_buggy_contracts/time_manipulation'
buggy_contract_dir = 'data/solidifi_buggy_contracts/unchecked_low_level_calls'

bug_category = basename(buggy_contract_dir)
# print(bug_category)


csv_smart_contracts = [join(buggy_contract_dir, f) for f in os.listdir(buggy_contract_dir) if f.endswith('.csv')]

vulnerabilities = []
for csv_sc in csv_smart_contracts:
    fp_sol_sc = os.path.splitext(csv_sc)[0].replace('BugLog', 'buggy') + '.sol'
    fn_sol_sc =  basename(fp_sol_sc)
    df_csv_sc = pd.read_csv(csv_sc)
    buggy_lines = []
    for index, row in df_csv_sc.iterrows():
        for i in range(row['length']):
            buggy_lines.append(row['loc'] + i)

    dict_sc_info = {
        'name': fn_sol_sc,
        'path': fp_sol_sc,
        'source': '',
        'vulnerabilities': [
            {
                'lines': buggy_lines,
                'category': bug_category
            }
        ]
    }

    vulnerabilities.append(dict_sc_info)

fp_vulnerabilities_json = join(buggy_contract_dir, 'vulnerabilities.json')
with open(fp_vulnerabilities_json, 'w') as outfile:
        json.dump(vulnerabilities, outfile, indent=4)
print('Dumped:', fp_vulnerabilities_json)
