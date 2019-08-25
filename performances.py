import utils
import itertools
import pandas
import numpy as np
import time

def draw_performances(type, realistic, smote):
    confusion_total = pandas.DataFrame()
    matrix = 'severe_confusion' if type == 'severe' else 'confusion'
    world = 'realistic' if realistic else 'experimental'
        
    for key in list(itertools.product(utils.get_projects(type), utils.get_class_models(type), utils.get_experiments(type), [realistic], [smote], utils.get_approaches(type), utils.get_classifiers(type))):
        data = utils.get_item(key, matrix)
        data['project'] = key[0]
        data['approach'] = key[5]

        data.index =  '_'.join(list(map(str, (key[0], key[1], key[2], key[3], key[4], key[6])))) + '_' + data.index

        confusion_total = confusion_total.append(data, sort=False)

    utils.boxplot_approaches(confusion_total, 'mcc_' + type + '_' + world, 'mcc', 'MCC', True)
    utils.boxplot_approaches(confusion_total, 'recall_' + type + '_' + world, 'recall', 'Recall', False)
    utils.boxplot_approaches(confusion_total, 'precision_' + type + '_' + world, 'precision', 'Precision', False)

    comparison = utils.create_approach_comparison(confusion_total)
    utils.pairplot_approaches(comparison, 'comparison_mcc_' + type, 'MCC')
    #utils.wilcoxon_approach_matrix(comparison, type)
    #utils.a12_approach_matrix(comparison, type)
            

if __name__ == "__main__":
    t_start = time.perf_counter()
    draw_performances('normal', 'realistic', 'no_smote')
    draw_performances('normal', 'ideal', 'smote')
    draw_performances('bug', 'realistic', 'no_smote')
    t_stop = time.perf_counter()

    print("--------------------------------------------------")
    print("Elapsed time: %.1f [sec]" % ((t_stop-t_start)))
    print("--------------------------------------------------") 