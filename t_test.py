
import json
from statistics import mean
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
from scipy.stats import f_oneway
from scipy.stats import kruskal
from scipy.stats import alexandergovern


if __name__ == '__main__':
    model_list = ['nodetype', 'metapath2vec', 'line', 'node2vec', 'non_feature/random_2',
                  'non_feature/random_8', 'non_feature/random_16', 'non_feature/random_32'
                  , 'non_feature/random_64', 'non_feature/random_128',
                  'non_feature/zeros_2', 'non_feature/zeros_8', 'non_feature/zeros_16', 
                  'non_feature/zeros_32', 'non_feature/zeros_64', 'non_feature/zeros_128',]
    model_list = ['nodetype', 'metapath2vec', 'line', 'node2vec']
    bug_type = {'access_control': 57, 'arithmetic': 60, 'denial_of_service': 46,
              'front_running': 44, 'reentrancy': 71, 'time_manipulation': 50, 
              'unchecked_low_level_calls': 95}
    bug_type = {'ethor': 0}
    archs = ['han', 'hgt']

    f_oneway_reports = {}
    kruskal_reports = {}
    alexandergovern_reports = {}
    for bugtype, num_sc in bug_type.items():
        model_reports = {'buggy_f1': [], 'macro_f1': []}
        for model in model_list:
            report_path = f'./experiments/logs/graph_classification/byte_code/hgt/cfg/{model}/{bugtype}/test_report.json'
            with open(report_path, 'r') as f:
                results = json.load(f)
            model_reports['buggy_f1'].append([rep['1']['f1-score'] for rep in results])
            model_reports['macro_f1'].append([rep['macro avg']['f1-score'] for rep in results])
            # print(model_reports['buggy_f1'])
        # f_oneway_buggy_t, f_oneway_buggy_p = f_oneway(model_reports['buggy_f1'])
        # f_oneway_macro_t, f_oneway_macro_p = f_oneway(model_reports['macro_f1'])
        

        f_oneway_reports[bugtype] = {'buggy_f1': f_oneway(*model_reports['buggy_f1']), 'macro_f1': f_oneway(*model_reports['macro_f1'])}
        kruskal_reports[bugtype] = {'buggy_f1': kruskal(*model_reports['buggy_f1']), 'macro_f1': kruskal(*model_reports['macro_f1'])}
        alexandergovern_reports[bugtype] = {'buggy_f1': alexandergovern(*model_reports['buggy_f1']), 'macro_f1': alexandergovern(*model_reports['macro_f1'])}

    for bugtype in bug_type:
        print(f_oneway_reports[bugtype]['buggy_f1'].pvalue, end=' ')
    print()
    for bugtype in bug_type:
        print(f_oneway_reports[bugtype]['macro_f1'].pvalue, end=' ')
    print()
    for bugtype in bug_type:
        print(kruskal_reports[bugtype]['buggy_f1'].pvalue, end=' ')
    print()
    for bugtype in bug_type:
        print(kruskal_reports[bugtype]['macro_f1'].pvalue, end=' ')
    print()
    for bugtype in bug_type:
        print(alexandergovern_reports[bugtype]['buggy_f1'].pvalue, end=' ')
    print()
    for bugtype in bug_type:
        print(alexandergovern_reports[bugtype]['macro_f1'].pvalue, end=' ')
    print()
