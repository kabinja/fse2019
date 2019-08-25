import utils
import time
import pandas
import itertools


def compute_before_after_series(data, shift):
    expected = utils.get_value_table(data, 'Expected')
    expected_s = expected.shift(shift)
    
    condition = ((expected == 'Not Vulnerable') & (expected_s == 'Vulnerable'))  
    condition = condition.transpose(copy=False)  
    condition = condition[condition.any(axis='columns')] 
    condition = condition.transpose(copy=False)

    probability_of_vulnerability = utils.get_value_table(data, 'Probability ofVulnerable')

    results = []

    for column, values in condition.iteritems():
        for row, value in values.iteritems():
            if value == True:
                before = probability_of_vulnerability[column][previous]
                after = probability_of_vulnerability[column][row]
                difference = after - before
                results.append({'before': before, 'after': after, 'difference': difference})

            previous = row

    return pandas.DataFrame(results)

def compute(shift):
    before_after = pandas.DataFrame()
    
    for key in list(itertools.product(utils.get_projects(), utils.get_class_models(), utils.get_experiments(), ['ideal'], utils.get_smote(), utils.get_approaches(), utils.get_classifiers())):
        data = utils.get_item(key, 'data')

        before_after_local = compute_before_after_series(data, shift)
        before_after_local['project'] = key[0]
        before_after_local['approach'] = key[5]

        before_after = before_after.append(before_after_local, sort=False)

    utils.boxplot_approaches(before_after, 'before_after_fix_' + str(shift), 'difference', '', True, [-1, 1])


if __name__ == "__main__":
    t_start = time.perf_counter()
    compute(1)
    #compute(2)
    t_stop = time.perf_counter()

    print("--------------------------------------------------")
    print("Elapsed time: %.1f [sec]" % ((t_stop-t_start)))
    print("--------------------------------------------------") 