from matplotlib import pyplot as plt
import numpy as np


### visualize training loss - over splits
def viz_training_folds(vizdata):
    ### expected format: (folds, timeseries)
    pass


def viz_training_models():
    ### expected format: (models, folds, timeseries)
    pass

def viz_training_metrics(data, groups, gcolors, labels, vtitle, saveto):
    ### expected data format: [model][fold](y/yhat shape)
    ### [model]
    ### [model]

    ### [model][dist]

    bar_mean = []
    for i in range(len(data)):
        data[i] = np.array(data[i])
        data[i] = data[i].mean(axis=tuple(range(1, data[i].ndim))).flatten()
        bar_mean.append(data[i].mean())

    print(bar_mean)
    print(data)
    alignto = np.arange(1, len(data)+1)
    apply_colors = [gcolors[groups[ii]] for ii in range(len(groups))]
    fig, ax = plt.subplots()
    ax.bar(alignto, bar_mean, color=apply_colors)#, labels=labels)
    ax.boxplot(data, vert=True, widths=(0.4), labels=labels)
    ax.set_title(vtitle)
    plt.savefig(saveto)


def viz_fold_y_yhat():
    ### per fold: y, yhat
    pass

def vis_fraction_under(data, groups, gcolors, labels, vtitle, saveto, linspace_size=1000, avg=False):
    ### data - either [model][fold][data] or [fold][data]
    ### need to avg over folds in case 1
    construction = np.linspace(0, 1, linspace_size)
    fracs_under_data = []
    for i in range(len(data)):
        fracs_under_data.append([])
        if avg:
            fold_means = np.zeros(data[i][j])
            frac_under_step = np.zeros(construction.shape)
            data[i] = np.array(data[i])
            data[i] = data[i].mean(axis=tuple(range(1, data[i].ndim))).flatten()
            for k in range(len(construction)):
                frac_under_step[k] = len(np.where(data[i] > construction[k])[0])/len(data[i])
        else:
            frac_under_step = np.zeros(construction.shape)
            for k in range(len(construction)):
                frac_under_step[k] = len(np.where(data[i][j] > construction[k])[0]) / len(data[i][j])

    ###



def viz_geo():
    pass