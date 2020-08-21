import numpy as np

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rc

def get_xy(point, origin, vector_x, vector_y):
    return np.array([np.dot(point - origin, vector_x), np.dot(point - origin, vector_y)])


def plot_single_interpolation(x, y, path):
    sns.set(style="whitegrid")
    sns.set_context("paper",rc={"lines.linewidth": 3.5,
                    'xtick.labelsize':20,
                    'ytick.labelsize':20,
                    'lines.markersize' : 15,
                    'legend.fontsize': 20,
                    'axes.labelsize': 20,
                    'legend.handlelength': 1,
                    'legend.handleheight':1,})

    color = 'green' if 'acc' in path else 'orange'
    suffix = 'acc' if 'acc' in path else 'loss'
    plt.plot(x, y, color=color)
    plt.xlabel('Interpolation')
    plt.ylabel('Validation {}'.format(suffix))
    # plt.title(title)
    # plt.ylim((0.0, 0.005))
    plt.tight_layout()
    plt.savefig(path+".png", dpi=200)
    plt.savefig(path+".pdf", dpi=200)

    plt.close()

def plot_multi_interpolations(x, ys, y_labels, path):
    sns.set(style="whitegrid")
    sns.set_context("paper",rc={"lines.linewidth": 3.5,
                    'xtick.labelsize':23,
                    'ytick.labelsize':23,
                    'lines.markersize' : 15,
                    'legend.fontsize': 18,
                    'axes.labelsize': 23,
                    'legend.handlelength': 1,
                    'legend.handleheight':1,})

    colors = ['blue', 'orange', 'green', 'red', 'purple', 'gold', 'darkcyan', 'crimson']
    suffix = 'Acc' if 'acc' in path else 'Loss'
    for i, y in enumerate(ys):
        plt.plot(x, ys[i], color=colors[i], label=y_labels[i])
    plt.xlabel('Interpolation')
    plt.ylabel('Validation {}'.format(suffix))
    # plt.title(title)
    plt.legend(title='Paths', title_fontsize=19, ncol=2)
    # plt.ylim((0.0, 0.005))
    plt.tight_layout()
    plt.savefig(path+".png", dpi=200)
    plt.savefig(path+".pdf", dpi=200)
    plt.close()


class LogNormalize(colors.Normalize):

    def __init__(self, vmin=None, vmax=None, clip=None, log_alpha=None):
        self.log_alpha = log_alpha
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        log_v = np.ma.log(value - self.vmin)
        log_v = np.ma.maximum(log_v, self.log_alpha)
        return 0.9 * (log_v - self.log_alpha) / (np.log(self.vmax - self.vmin) - self.log_alpha)


def plot_heat_map(matrix, labels, path, xlabel='', ylabel=''):
    # matrix = np.vectorize(lambda x: np.round(x, 2))(matrix)
    sns.set(style="ticks")
    sns.set_context("paper",rc={"lines.linewidth": 2.5,
                    'xtick.labelsize':22,
                    'ytick.labelsize':22,
                    'lines.markersize' : 15,
                    'legend.fontsize': 24,
                    'axes.labelsize': 22,
                    'axes.titlesize': 22,
                    'legend.handlelength': 1,
                    'legend.handleheight':1,})
    rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    
    ax = sns.heatmap(matrix, annot=True, fmt='.2f', cbar=True, \
        vmin=0, vmax=1, xticklabels=labels, yticklabels=labels,\
        annot_kws={'size': 21}, cmap='YlGnBu')
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    ax.get_figure().savefig(path+'.png', dpi=200)
    ax.get_figure().savefig(path+'.pdf', dpi=200)
    plt.close()

def plot_accs(num_tasks, seq_data, mtl_data, path):
    sns.set(style="whitegrid")
    sns.set_context("paper",rc={"lines.linewidth": 4.0,
                    'xtick.labelsize':24,
                    'ytick.labelsize':24,
                    'lines.markersize' : 12,
                    'legend.fontsize': 18,
                    'axes.labelsize': 24,
                    'legend.handlelength': 1,
                    'legend.handleheight':1,})
    rc('text', usetex=True)

    colors = ['#33BBEE', '#FF7043', '#009988', '#EE3377', '#CC3311']
    plt.axhline(y=np.mean(mtl_data[-1]), color='black', linestyle='dashed', label='Multitask')
    for task in range(1, num_tasks+1):
        plt.plot(range(task, num_tasks+1), seq_data[task-1:,task-1], color=colors[task-1], marker='o', label='Task {}'.format(task))


    plt.xticks(range(1, num_tasks+1))
    plt.legend()
    plt.xlabel(r'Tasks Learned')
    plt.ylabel(r'Validation Accuracy')
    plt.tight_layout()
    plt.savefig(path+'.png', dpi=200)
    plt.savefig(path+'.pdf', dpi=200)
    plt.close()

def plot_l2_map(matrix, labels, path):
    # matrix = np.vectorize(lambda x: np.round(x, 2))(matrix)
    sns.set(style="ticks")
    sns.set_context("paper",rc={"lines.linewidth": 2.5,
                    'xtick.labelsize':24,
                    'ytick.labelsize':24,
                    'lines.markersize' : 15,
                    'legend.fontsize': 24,
                    'axes.labelsize': 24,
                    'legend.handlelength': 1,
                    'legend.handleheight':1,})
    rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    
    ax = sns.heatmap(matrix, annot=True, fmt='.1f', cbar=True, \
        xticklabels=labels, yticklabels=labels,\
        annot_kws={'size': 18}, cmap='plasma')
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0)
    plt.xlabel(r'Parameters')
    plt.tight_layout()
    ax.get_figure().savefig(path+'.png', dpi=200)
    ax.get_figure().savefig(path+'.pdf', dpi=200)
    plt.close()

def plot_contour(grid, values, coords, vmax=None, log_alpha=-5, N=7, path='default.png', cmap='jet_r', w_labels=[]):
    sns.set(style="ticks")
    sns.set_context("paper",rc={"lines.linewidth": 2.5,
                    'xtick.labelsize':24,
                    'ytick.labelsize':24,
                    'lines.markersize' : 15,
                    'legend.fontsize': 24,
                    'axes.labelsize': 24,
                    'legend.handlelength': 1,
                    'legend.handleheight':1,})
    rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    
    cmap = plt.get_cmap(cmap)
    if vmax is None:
        clipped = values.copy()
    else:
        clipped = np.minimum(values, vmax)
    log_gamma = (np.log(clipped.max() - clipped.min()) - log_alpha) / N
    levels = clipped.min() + np.exp(log_alpha + log_gamma * np.arange(N + 1))
    levels[0] = clipped.min()
    levels[-1] = clipped.max()
    levels = np.sort(np.concatenate((levels, [10e10])))

    # print(levels)
    #norm = LogNormalize(clipped.min() - 1e-8, clipped.max() + 1e-8, log_alpha=log_alpha)
    #levels = [35, 40, 50, 60, 70, 80, 90]
    # levels = [0.04, 0.05, 0.06, 0.07, 0.08, 1.0]
    contour = plt.contour(grid[:, :, 0], grid[:, :, 1], values, cmap=cmap,# norm=norm,
                          linewidths=2.5,
                          zorder=1,
                          levels=7)
    contourf = plt.contourf(grid[:, :, 0], grid[:, :, 1], values, cmap=cmap,# norm=norm,
                            levels=7,
                            zorder=0,
                            alpha=0.5)
    colorbar = plt.colorbar(format='%.2g')

    labels = list(colorbar.ax.get_yticklabels())
    labels[-1].set_text(r'$>\,$' + labels[-2].get_text())
    colorbar.ax.set_yticklabels(labels)
    plt.scatter(coords[0, 0], coords[0, 1], marker='o', c='k', s=120, zorder=2)

    plt.text(coords[0, 0]+0.05, coords[0, 1]+0.05, w_labels[0], fontsize=22)
    

    plt.scatter(coords[1, 0], coords[1, 1], marker='o', c='k', s=120, zorder=2)
    plt.text(coords[1, 0]+0.05, coords[1, 1]+0.05, w_labels[1], fontsize=22)

    plt.scatter(coords[2, 0], coords[2, 1], marker='o', c='k', s=120, zorder=2)
    plt.text(coords[2, 0]+0.05, coords[2, 1]+0.05, w_labels[2], fontsize=22)


    
    plt.margins(0.0)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    colorbar.ax.tick_params(labelsize=20)
    plt.savefig(path+'.png', dpi=200, bbox_inches='tight')
    plt.savefig(path+'.pdf', dpi=200, bbox_inches='tight')
    plt.close()
