import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas
import subprocess
import os
import itertools
import gzip

from scipy.stats import wilcoxon

import matplotlib
matplotlib.rcParams['text.usetex'] = True

PROJECTS = ['openssl', 'linux_kernel', 'wireshark']
CLASS_MODELS = ['last_3_releases']
EXPERIMENTS = ['Vulnotvul']
REALISTIC = ['ideal', 'realistic']
SMOTE = ['smote', 'no_smote']
APPROACHES = ['BagOfWords', 'CodeMetrics', 'FunctionCalls', 'Includes']
CLASSIFIERS = ['RandomForest']

PROJECTS_ALTERNATIVE = ['openssl', 'linux_kernel', 'wireshark']
CLASS_MODELS_ALTERNATIVE = ['last_3_releases']
EXPERIMENTS_ALTERNATIVE = ['Vulnotvul']
REALISTIC_ALTERNATIVE = ['ideal', 'realistic']
SMOTE_ALTERNATIVE = ['no_smote']
APPROACHES_ALTERNATIVE = ['BagOfWords', 'CodeMetrics', 'FunctionCalls', 'Includes']
CLASSIFIERS_ALTERNATIVE = ['RandomForest']

HDF5_CACHE = 'cache_{}.h5'
ROOT_FOLDER = "data/stats-{}/"
FIGURE_FOLDER = 'results/figures/'
TABLE_FOLDER = 'results/tables/'

GRAPHIX_EXTENSION = '.pdf'

def load_data(input_info, weight):
    csv = get_filename(input_info, weight)
    print(csv)

    data = pandas.read_csv(csv, sep=',', compression='gzip')
    clean_names(data)

    return data


def clean_names(data):
    if 'Experiment' in data.columns:
        data['Experiment'] = data['Experiment'].apply(rename_releases)

    if 'project' in data.columns:
        data['project'] = data['project'].apply(rename_projects)

    if 'approach' in data.columns:
        data['approach'] = data['approach'].apply(rename_approach)


def rename_releases(release):
    if 'OpenSSL_' in release:
        release = release.replace('OpenSSL_', 'v')
        release = release.replace('_', '.')
    elif 'wireshark-' in release:
        release = release.replace('wireshark-', 'v')

    return release


def rename_projects(project):
    if 'linux_kernel' == project:
        return 'Linux Kernel'
    elif 'openssl' == project:
        return 'OpenSSL'
    elif 'wireshark' == project:
        return 'Wireshark'
    else:
        return project


def rename_approach(approach):
    if 'BagOfWords' == approach:
        return 'Bag Of Words'
    elif 'CodeMetrics' == approach:
        return 'Code Metrics'
    elif 'FunctionCalls' == approach:
        return 'Function Calls'
    elif 'Includes' == approach:
        return 'Imports'
    else:
        return approach


def get_filename(input_info, weight):
    csv = ROOT_FOLDER.format(weight) 
    csv += input_info['project'] 
    csv += '/' + input_info['class_model'] 
    csv += '/' + input_info['experiment'] 
    csv += '/' + input_info['approach']
    csv += '/' + input_info['realistic']
    csv += '/' + input_info['classifier']

    return (csv + '-smote.csv.gz') if input_info['smote'] == 'smote' else (csv + '.csv.gz')


def get_prefix(folder, input_info):
    prefix = folder + input_info['project'] +'_' + input_info['class_model'] + '_' + input_info['approach'] + '_' + input_info['classifier']
    
    if input_info['realistic']:
        prefix += '_realistic'
    
    if input_info['smote']:
        prefix += '_smote'

    return prefix


def get_total_prefix(folder, input_info):
    prefix = folder + input_info['project'] + '_' + input_info['class_model']

    if input_info['realistic']:
        prefix += '_realistic'
    
    if input_info['smote']:
        prefix += '_smote'

    return prefix


def get_item(key, type, weight='normal'):
    if not os.path.isfile(HDF5_CACHE.format(weight)):
        print('generating cache at ' + HDF5_CACHE.format(weight))
        hdf = pandas.HDFStore(HDF5_CACHE.format(weight))
        load_hdf5(hdf, weight)
        hdf.close()
        print('cached generated')

    key = '/'.join(list(map(str, key))) + '/' + type

    return pandas.read_hdf(HDF5_CACHE.format(weight), key=key)


def get_projects(weight='normal'):
    return PROJECTS if weight == 'normal' else PROJECTS_ALTERNATIVE


def get_class_models(weight='normal'):
    return CLASS_MODELS if weight == 'normal' else CLASS_MODELS_ALTERNATIVE


def get_experiments(weight='normal'):
    return EXPERIMENTS if weight == 'normal' else EXPERIMENTS_ALTERNATIVE


def get_realistic(weight='normal'):
    if weight == 'realistic':
        return ['realistic']

    return REALISTIC if weight == 'normal' else REALISTIC_ALTERNATIVE


def get_smote(weight='normal'):
    return SMOTE if weight == 'normal' or weight == 'realistic' else SMOTE_ALTERNATIVE


def get_approaches(weight='normal'):
    return APPROACHES if weight == 'normal' else APPROACHES_ALTERNATIVE


def get_classifiers(weight='normal'):
    return CLASSIFIERS if weight == 'normal' else CLASSIFIERS_ALTERNATIVE


def export_table(file, data):
    filename = file + '.tex'

    template = r'''
    \documentclass{{article}}
    \usepackage{{pdflscape}}
    \usepackage{{booktabs}}
    \begin{{document}}
    \begin{{landscape}}
    {0}
    \end{{landscape}}
    \end{{document}}
    '''

    cwd = os.getcwd()
    os.chdir(TABLE_FOLDER)

    with open(filename, 'w') as f:
        f.write(template.format(data.to_latex()))
    
    with open(os.devnull, "w") as f:
        subprocess.call(['pdflatex', filename], stdout=f)

    os.chdir(cwd)


def boxplot_single(data, name, variable, y_label):
    fig = plt.figure(figsize=(4, 3))

    sns.set(style="ticks")
    sns.boxplot(x="project", y=variable, data=data, palette="PRGn", linewidth=1)

    plt.ylabel(y_label)

    fig.tight_layout()

    plt.savefig(FIGURE_FOLDER + name + GRAPHIX_EXTENSION)
    plt.close('all')


def boxplot_multiple(data, name, variable, hue, y_label, legend, y_lim = [0,1]):
    drawing = data.copy()
    clean_names(drawing)

    fig = plt.figure(figsize=(6,6))

    sns.set(style="ticks")
    g = sns.boxplot(x="project", y=variable, hue=hue, data=drawing, palette="PRGn", linewidth=1)
    sns.despine(offset=10, trim=False)

    plt.tick_params(axis='both', which='both', labelsize=20)

    plt.title(y_label, fontsize=23)
    plt.ylabel('', fontsize=23)
    plt.xlabel('', fontsize=23)

    if not legend:
        g.legend_.remove()
    else:
        plt.legend(loc='best', prop={'size': 15})

    if len(y_lim) == 2:
        plt.ylim(y_lim)

    fig.tight_layout()

    plt.savefig(FIGURE_FOLDER + name + GRAPHIX_EXTENSION, dpi=1000, bbox_inches='tight')
    plt.close('all')

def scatter_releases(data, name, project, variable, hue, y_label):
    plt.figure(figsize=(2, 1.5))

    sns.set(style='white')

    sns.set_palette("PRGn")
    g = sns.lmplot(x="release", y=variable, hue=hue, data=data, fit_reg=False)
    
    current_palette = sns.color_palette()

    medians = data.groupby([hue]).median()[variable].tolist()
    for i in range(len(medians)):
        g.ax.axhline(medians[i],  c=current_palette[i])    

    plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

    plt.ylabel(y_label)
    plt.xlabel('Releases')

    plt.savefig(FIGURE_FOLDER + project + '_' + name + GRAPHIX_EXTENSION, dpi=1000, bbox_inches='tight')

    plt.close('all')

def boxplot_approaches(data, name, variable, y_label, legend, y_lim = [0,1]):
    boxplot_multiple(data, name, variable, 'approach', y_label, legend, y_lim)

def pairplot_approaches(data, name, y_label):
    fig = plt.figure(figsize=(15, 15))

    sns.set(style="ticks")
    sns.pairplot(data=data, diag_kind="kde", hue="project", markers="+", kind="reg")

    plt.ylabel(y_label)

    fig.tight_layout()

    plt.savefig(FIGURE_FOLDER + name + GRAPHIX_EXTENSION)
    plt.close('all')


def get_value_table(data, value):
    return data.pivot(index='Experiment', columns='File', values=value)


def a12(lst1,lst2,rev=True):
    more = same = 0.0
    for x in lst1:
        for y in lst2:
            if   x==y : 
                same += 1
            elif rev and x > y :
                 more += 1
            elif not rev and x < y : 
                more += 1

    return (more + 0.5*same)  / (len(lst1)*len(lst2))


def compute_confusion_matrix(data, weight, only_severe=False):
    index = data.groupby('Experiment').groups.keys()

    columns=['true_positive', 'true_negative', 'false_positive', 'false_negative']
    confusion_matrix = pandas.DataFrame(data=0, index=index, columns=columns)
    
    for index, row in data.iterrows():
        if row['Class'] == 'Vulnerability' and row['Prediction'] == 'Vulnerable':
            if only_severe and row['CVSS'] < 7:
                confusion_matrix.at[row['Experiment'], 'false_positive'] += 1
            else :
                confusion_matrix.at[row['Experiment'], 'true_positive'] += 1
        elif  row['Class'] != 'Vulnerability' and row['Prediction'] == 'Not Vulnerable':
            confusion_matrix.at[row['Experiment'], 'true_negative'] += 1
        elif  row['Class'] != 'Vulnerability' and row['Prediction'] == 'Vulnerable':
            confusion_matrix.at[row['Experiment'], 'false_positive'] += 1
        elif  row['Class'] == 'Vulnerability' and row['Prediction'] == 'Not Vulnerable':
            if only_severe and row['CVSS'] < 7:
                confusion_matrix.at[row['Experiment'], 'true_negative'] += 1
            else :
                confusion_matrix.at[row['Experiment'], 'false_negative'] += 1
        else:
            raise Exception('Wrong values: ' + row['Expected'] + ' and ' + row['Prediction'])

    add_precision(confusion_matrix)
    add_recall(confusion_matrix)
    add_mcc(confusion_matrix)

    return confusion_matrix


def add_mcc(confusion):
    tp = confusion['true_positive']
    tn = confusion['true_negative']
    fp = confusion['false_positive']
    fn = confusion['false_negative']
    
    numerator = (tp.multiply(tn)).subtract(fp.multiply(fn))
    denominator = np.sqrt((tp.add(fp)).multiply(tp.add(fn)).multiply(tn.add(fp)).multiply(tn.add(fn)))

    confusion['mcc'] = numerator.divide(denominator)


def add_precision(confusion):
        tp = confusion['true_positive']
        fp = confusion['false_positive']

        detected_positive = tp.add(fp)
        
        confusion['precision'] = tp.divide(detected_positive)


def add_recall(confusion):
        tp = confusion['true_positive']
        fn = confusion['false_negative']

        real_positive = tp.add(fn)

        confusion['recall'] = tp.divide(real_positive)


def wilcoxon_approach_matrix(data, type):
    table = pandas.DataFrame(data= np.nan, index= get_approaches(), columns= get_approaches())

    for column in get_approaches():
        for index in get_approaches():
            _, table.at[index, column] = wilcoxon(data[index], data[column])

    export_table('wilcoxon_' + type, table)


def a12_approach_matrix(data, type):
    table = pandas.DataFrame(data= np.nan, index= get_approaches(), columns= get_approaches())

    for column in get_approaches():
        for index in get_approaches():
            table.at[index, column] = a12(data[index], data[column])

    export_table('a12_' + type, table)


def create_approach_comparison(confusion):
    comparison = confusion.pivot(columns='approach', values='mcc')
    comparison['project'] = comparison.apply(lambda x: extract_project(x.name), axis=1)

    return comparison


def extract_project(name):
    for project in get_projects():
        if project in name:
            return project


def load_hdf5(hdf, weight):   
    input_info = dict()

    for project in get_projects(weight):
        input_info['project'] = project

        for class_model in get_class_models(weight):
            input_info['class_model'] = class_model

            for experiment in get_experiments(weight):
                input_info['experiment'] = experiment
                
                for realistic in get_realistic(weight):
                    input_info['realistic'] = realistic

                    for smote in get_smote(weight):
                        input_info['smote'] = smote

                        for approach in get_approaches(weight):
                            input_info['approach'] = approach

                            for classifier in get_classifiers(weight):
                                input_info['classifier'] = classifier

                                data = load_data(input_info, weight)
                                key = '/'.join(list(map(str, input_info.values())))

                                if data.empty:
                                    print(get_filename(input_info, weight) + ' is empty!')
                                    continue

                                print(key)

                                hdf.put(key + '/data', data)

                                # precision, recall and mcc
                                confusion_matrix = compute_confusion_matrix(data, weight)
                                hdf.put(key + '/confusion', confusion_matrix)
                                
                                severe_confusion_matrix = compute_confusion_matrix(data, weight, True)
                                hdf.put(key + '/severe_confusion', severe_confusion_matrix)



if __name__ == "__main__":
    print('UTILS FUNCTIONS')