import sys
import os
import re
import subprocess

pattern =  re.compile(r"pragma solidity\s*(?:\^|>=|<=)?\s*(\d+\.\d+\.\d+)")
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

smart_contract_path = './ge-sc-data/solidifi_buggy_contracts/Re-entrancy'
smart_contracts = [os.path.join(smart_contract_path, f) for f in os.listdir(smart_contract_path) if f.endswith('.sol')]
count = 0
for sc in smart_contracts:
    sc_version = get_solc_version(sc)
    try:
        subprocess.run(['solc-select', 'install', sc_version])
        count += 1
    except:
        print(sc_version)
print(f'Extract {count}/{len(smart_contracts)} sources')
