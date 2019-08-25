import utils
import time
import matplotlib.pyplot as plt
import seaborn as sns

def draw_distribution_vulnerable_files(project, data, severe):
    tp = data['true_positive']
    tn = data['true_negative']
    fp = data['false_positive']
    fn = data['false_negative']
    
    total = tp.add(tn).add(fp).add(fn)
    vulnerable = tp.add(fn)
    percentage = vulnerable.divide(total) * 100

    draw_distribution_files(project, 'total', 'Number of vulnerable files', severe, vulnerable)
    draw_distribution_files(project, 'percentage', 'Percentage of vulnerable files', severe, percentage)

def draw_distribution_files(project, type, label, severe, data):
    plt.figure(figsize=(8, 6))
    sns.set(style='white')

    data.plot.bar(color='darkgrey', width=1.0)
    plt.ylabel(label, fontsize=23)
    plt.xlabel('Releases', fontsize=23)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    plt.tick_params(axis='both', which='both', labelsize=20)

    severe_name = 'severe_' if severe else ''
    plt.savefig(utils.FIGURE_FOLDER + project + '_distribution_' + severe_name + type + utils.GRAPHIX_EXTENSION, dpi=1)
    plt.close('all')

if __name__ == "__main__":
    t_start = time.perf_counter()
    for project in utils.get_projects():
        confusion = utils.get_item((project, 'last_3_releases', 'Vulnotvul', 'ideal', 'no_smote', 'BagOfWords', 'RandomForest'), 'confusion')
        draw_distribution_vulnerable_files(project, confusion, False)

        print(project + ' releases: ' + str(len(confusion.index.unique())))

    t_stop = time.perf_counter()

    print("--------------------------------------------------")
    print("Elapsed time: %.1f [sec]" % ((t_stop-t_start)))
    print("--------------------------------------------------") 

