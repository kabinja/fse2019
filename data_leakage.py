import utils
import itertools
import pandas
import time
import numpy as np
import matplotlib.pyplot as plt
import h5py

def parse_classier(approach_name):
    return ['BagOfWords', 'CodeMetrics', 'FunctionCalls', 'Includes'] if approach_name == 'all' else [approach_name]

def compute_data_leakage(approach_name):
    data_leakage = pandas.DataFrame()
    approach = parse_classier(approach_name)

    for key in list(itertools.product(utils.get_projects(), utils.get_class_models(), utils.get_experiments(), ['ideal'], ['no_smote'], approach, utils.get_classifiers())):
        key_realistic = (key[0], key[1], key[2], 'realistic', key[4], key[5], key[6])

        experimental = utils.get_item(key, 'confusion')
        experimental['knowledge'] = 'experimental'
        experimental['approach'] = key[5]
        experimental['project'] = key[0]
        data_leakage = data_leakage.append(experimental)

        noiseless = utils.get_item(key_realistic, 'confusion', 'realistic')
        noiseless['knowledge'] = 'noiseless'
        noiseless['approach'] = key[5]
        noiseless['project'] = key[0]
        data_leakage = data_leakage.append(noiseless)

        realistic = utils.get_item(key_realistic, 'confusion')
        realistic['knowledge'] = 'realistic'
        realistic['approach'] = key[5]
        realistic['project'] = key[0]
        data_leakage = data_leakage.append(realistic)
        
    return data_leakage


def compute_data_leakage_bug(approach_name):
    data_leakage = pandas.DataFrame()
    approach = parse_classier(approach_name)

    for key in list(itertools.product(utils.get_projects(), utils.get_class_models(), utils.get_experiments(), ['ideal'], ['no_smote'], approach, utils.get_classifiers())):
        key_realistic = (key[0], key[1], key[2], 'realistic', key[4], key[5], key[6])

        experimental = utils.get_item(key, 'confusion', 'bug')
        experimental['knowledge'] = 'experimental'
        experimental['approach'] = key[5]
        experimental['project'] = key[0]
        data_leakage = data_leakage.append(experimental)

        realistic = utils.get_item(key_realistic, 'confusion', 'bug')
        realistic['knowledge'] = 'realistic'
        realistic['approach'] = key[5]
        realistic['project'] = key[0]
        data_leakage = data_leakage.append(realistic)
        
    return data_leakage


def draw_by_approach():
    for approach in ['BagOfWords', 'CodeMetrics', 'FunctionCalls', 'Includes']:
        data_leakage = compute_data_leakage(approach)
        utils.boxplot_multiple(data_leakage, 'mcc_data_leakage_' + approach, 'mcc', 'knowledge', 'MCC', True)
        utils.boxplot_multiple(data_leakage, 'recall_data_leakage_' + approach, 'recall', 'knowledge', 'Recall', False)
        utils.boxplot_multiple(data_leakage, 'precision_data_leakage_' + approach, 'precision', 'knowledge', 'Precision', False)

def draw_by_knowledge():
    data = compute_data_leakage('all')
    data_bug = compute_data_leakage_bug('all')

    for knowledge in ['experimental', 'noiseless', 'realistic']:
        utils.boxplot_multiple(data.loc[data['knowledge'] == knowledge], 'mcc_data_leakage_no_smote_' + knowledge, 'mcc', 'approach', 'MCC', True)
        utils.boxplot_multiple(data.loc[data['knowledge'] == knowledge], 'recall_data_leakage_no_smote_' + knowledge, 'recall', 'approach', 'Recall', False)
        utils.boxplot_multiple(data.loc[data['knowledge'] == knowledge], 'precision_data_leakage_no_smote_' + knowledge, 'precision', 'approach', 'Precision', False)

        if knowledge == 'noiseless':
            continue

        utils.boxplot_multiple(data_bug.loc[data_bug['knowledge'] == knowledge], 'mcc_data_leakage_bug_' + knowledge, 'mcc', 'approach', 'MCC', True)
        utils.boxplot_multiple(data_bug.loc[data_bug['knowledge'] == knowledge], 'recall_data_leakage_bug_' + knowledge, 'recall', 'approach', 'Recall', False)
        utils.boxplot_multiple(data_bug.loc[data_bug['knowledge'] == knowledge], 'precision_data_leakage_bug_' + knowledge, 'precision', 'approach', 'Precision', False)


def print_values():
    data = compute_data_leakage('BagOfWords')
    data = data.loc[data['knowledge'] != 'noiseless']
    data = data.groupby(["knowledge", "project"]).mcc.describe().unstack()['mean'].reset_index()
    print('weight: normal') 
    print(data) 
    
    data_bug = compute_data_leakage_bug('BagOfWords').set_index('project')
    data_bug = data_bug.groupby(["knowledge", "project"]).mcc.describe().unstack()['mean'].reset_index()
    print('weight: bug')
    print(data_bug) 

    #improvement = data.sub(data_bug)
    #print(improvement) 

if __name__ == "__main__":
    t_start = time.perf_counter()

    #f = h5py.File('cache_normal.h5', 'r')
    #for key in f.keys():
    #        print(key)

    print_values()
    draw_by_approach()
    draw_by_knowledge()

    t_stop = time.perf_counter()

    print("--------------------------------------------------")
    print("Elapsed time: %.1f [sec]" % ((t_stop-t_start)))
    print("--------------------------------------------------")