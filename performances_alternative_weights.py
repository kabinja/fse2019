import utils
import itertools
import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

def draw_performances(weight):
    confusion_total = pandas.DataFrame()
        
    for key in list(itertools.product(utils.get_projects(weight), utils.get_class_models(weight), utils.get_experiments(weight), ['ideal'], ['no_smote'], utils.get_approaches(weight), utils.get_classifiers(weight))):
        alternative = utils.get_item(key, 'severe_confusion', weight)
        normal = utils.get_item(key, 'severe_confusion', 'normal')
        alternative['difference_mcc'] = alternative['mcc'].subtract(normal['mcc'])
        alternative['project'] = key[0]
        alternative['approach'] = key[5]

        confusion_total = confusion_total.append(alternative, sort=False)

    utils.boxplot_approaches(confusion_total, 'mcc_strategy_' + weight, 'mcc', 'MCC', True)
    utils.boxplot_approaches(confusion_total, 'recall_strategy_' + weight, 'recall', 'Recall', False)
    utils.boxplot_approaches(confusion_total, 'precision_strategy_' + weight, 'precision', 'Precision', False)
    utils.boxplot_approaches(confusion_total, 'difference_mcc_strategy_' + weight, 'difference_mcc', 'Difference of MCC', False, [-0.6,0.3])
    

if __name__ == "__main__":
    t_start = time.perf_counter()
    draw_performances('normal')
    draw_performances('severe')
    draw_performances('bug')
    t_stop = time.perf_counter()

    print("--------------------------------------------------")
    print("Elapsed time: %.1f [sec]" % ((t_stop-t_start)))
    print("--------------------------------------------------") 