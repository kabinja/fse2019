import utils
import itertools
import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

def draw_performances():
    confusion_total = pandas.DataFrame()

    for weight in ['normal', 'smote', 'cvss', 'severe', 'bug']:
        key_weight = weight

        for key in list(itertools.product(utils.get_projects(weight), utils.get_class_models(weight), utils.get_experiments(weight), ['ideal'], ['no_smote'], ['BagOfWords'], utils.get_classifiers(weight))):
            if weight == 'smote':
                key = (key[0], key[1], key[2], key[3], 'smote', key[5], key[6])
                key_weight = 'normal'
            
            alternative = utils.get_item(key, 'severe_confusion', key_weight)
            alternative['project'] = key[0]
            alternative['weight'] = weight

            confusion_total = confusion_total.append(alternative, sort=False)

    utils.boxplot_multiple(confusion_total, 'mcc_strategies', 'mcc', 'weight', 'MCC', True)
    utils.boxplot_multiple(confusion_total, 'precision_strategies', 'precision', 'weight', 'Precision', False)
    utils.boxplot_multiple(confusion_total, 'recall_strategies', 'recall', 'weight', 'Recall', False)

if __name__ == "__main__":
    t_start = time.perf_counter()
    draw_performances()
    t_stop = time.perf_counter()

    print("--------------------------------------------------")
    print("Elapsed time: %.1f [sec]" % ((t_stop-t_start)))
    print("--------------------------------------------------") 