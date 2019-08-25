import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

def get_sorted(data, release):
    release = data.loc[data['Experiment'] == release]
    
    release = release.sort_values(by='Probability ofVulnerable', ascending=False, na_position='last')
    release.index = np.arange(1, len(release) + 1)

    return release[['CVSS', 'Class']]


def compute_score(sorted_list, severe):
    if severe == 'severe':
        results = sorted_list.loc[(sorted_list['Class'] == 'Vulnerability') & (sorted_list['CVSS'] > 7.)]

        if results.shape[0] == 0:
            return sorted_list.shape[0]

        return results.index[-1] / sorted_list.shape[0]
    else:
        return sorted_list.loc[sorted_list['Class'] == 'Vulnerability'].index[-1] / sorted_list.shape[0]


def compute_topn(sorted_list, top, severe):
    if severe == 'severe':
        return sorted_list.head(top).loc[(sorted_list['Class'] == 'Vulnerability') & (sorted_list['CVSS'] > 7.)].shape[0]
    else:
        return sorted_list.head(top).loc[sorted_list['Class'] == 'Vulnerability'].shape[0]


def compute_severe(topn):
    data = pd.DataFrame()

    for project in utils.get_projects():
        normal_raw = utils.get_item((project, 'last_3_releases', 'Vulnotvul', 'ideal', 'smote', 'BagOfWords', 'RandomForest'), 'data')
        normal =  build_df(normal_raw, topn, 'normal')
        normal['CVSS'] = 'normal'
        normal['project'] = project 
        data = data.append(normal)

        severe =  build_df(normal_raw, topn, 'severe')
        severe['CVSS'] = 'severe'
        severe['project'] = project 
        data = data.append(severe)

    utils.boxplot_multiple(data, 'score_cvss', 'score', 'CVSS', 'Score', False)
    utils.boxplot_multiple(data, 'top' + str(topn) + '_cvss', 'topn', 'CVSS', 'Top ' + str(topn), 'smote', [0, topn])

def compute_data(topn, realistic, smote='no_smote', weight='normal'):
    data = pd.DataFrame()

    for project in utils.get_projects():
        for approach in utils.get_approaches():
            normal_raw = utils.get_item((project, 'last_3_releases', 'Vulnotvul', realistic, smote, approach, 'RandomForest'), 'data', weight)
            normal =  build_df(normal_raw, topn, 'normal')
            normal['project'] = project 
            normal['approach'] = approach
            data = data.append(normal)

    world = 'realistic' if realistic else 'experimental'

    print_values(data)
    utils.boxplot_multiple(data, 'score_' + weight + '_' + world, 'score', 'approach', 'Resolution Effort Ratio', True)
    utils.boxplot_multiple(data, 'top' + str(topn) + '_' + weight + '_' + world, 'topn', 'approach', 'Top ' + str(topn), False, [0, topn])

def compute_alternative(topn):
    data = pd.DataFrame()

    for project in utils.get_projects():
        normal_raw = utils.get_item((project, 'last_3_releases', 'Vulnotvul', 'ideal', 'no_smote', 'BagOfWords', 'RandomForest'), 'data')   
        normal = build_df(normal_raw, topn, 'severe')
        normal['weight'] = 'normal'
        normal['project'] = project 
        data = data.append(normal)

        smote_raw = utils.get_item((project, 'last_3_releases', 'Vulnotvul', 'ideal', 'no_smote', 'BagOfWords', 'RandomForest'), 'data')  
        smote = build_df(smote_raw, topn, 'severe')
        smote['weight'] = 'smote'
        smote['project'] = project       
        data = data.append(smote)

        cvss_raw = utils.get_item((project, 'last_3_releases', 'Vulnotvul', 'ideal', 'no_smote', 'BagOfWords', 'RandomForest'), 'data', 'cvss')
        cvss = build_df(cvss_raw, topn, 'severe')
        cvss['weight'] = 'cvss'
        cvss['project'] = project   
        data = data.append(cvss)

        severe_raw = utils.get_item((project, 'last_3_releases', 'Vulnotvul', 'ideal', 'no_smote', 'BagOfWords', 'RandomForest'), 'data', 'severe')
        severe = build_df(severe_raw, topn, 'severe')
        severe['weight'] = 'severe'
        severe['project'] = project       
        data = data.append(severe)

        bug_raw = utils.get_item((project, 'last_3_releases', 'Vulnotvul', 'ideal', 'no_smote', 'BagOfWords', 'RandomForest'), 'data', 'bug')
        bug = build_df(bug_raw, topn, 'severe')
        bug['weight'] = 'bug'
        bug['project'] = project     
        data = data.append(bug)

    utils.boxplot_multiple(data, 'score_weight', 'score', 'weight', 'Score', False)
    utils.boxplot_multiple(data, 'top' + str(topn) + '_weight', 'topn', 'weight', 'Top ' + str(topn), True, [0, topn])


def build_df(data, topn, severe):
    releases = data['Experiment'].unique()
    df = pd.DataFrame(data=np.nan, index=releases, columns=['score', 'topn'])

    for release in releases:
        sorted_list = get_sorted(data, release)
        df.at[release, 'score'] = compute_score(sorted_list, severe)
        df.at[release, 'topn'] = compute_topn(sorted_list, topn, severe)

    df['release'] = df.reset_index().index

    return df


def print_values(data):
    data = data.groupby(["approach", "project"]).topn.describe().unstack()['mean'].reset_index()
    print(data) 


if __name__ == "__main__":
    t_start = time.perf_counter()
    compute_data(10, 'realistic')
    compute_data(10, 'ideal')
    compute_data(10, 'realistic', weight='bug')
    compute_data(10, 'realistic')
    compute_data(10, 'ideal')
    compute_severe(10)
    compute_alternative(10)
    t_stop = time.perf_counter()

    print("--------------------------------------------------")
    print("Elapsed time: %.1f [sec]" % ((t_stop-t_start)))
    print("--------------------------------------------------") 