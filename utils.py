import torch
import numpy as np
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rc
import random
import string

from models import ResNet18, MLP

def get_random_string(length):
    # Random string with the combination of lower and upper case
    letters = string.ascii_letters
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def save_model(model, path):
    torch.save(model.cpu(), path)

def load_model(path):
    model = torch.load(path)
    return model

def flatten_params(m, numpy_output=True):
    total_params = []
    for param in m.parameters():
            total_params.append(param.view(-1))
    total_params = torch.cat(total_params)
    if numpy_output:
        return total_params.cpu().detach().numpy()
    return total_params

def assign_weights(m, weights):
    state_dict = m.state_dict(keep_vars=True)
    index = 0
    with torch.no_grad():
        for param in state_dict.keys():
            if 'running_mean' in param or 'running_var' in param or 'num_batches_tracked' in param:
                continue
            # print(param, index)
            param_count = state_dict[param].numel()
            param_shape = state_dict[param].shape
            state_dict[param] =  nn.Parameter(torch.from_numpy(weights[index:index+param_count].reshape(param_shape)))
            index += param_count
    m.load_state_dict(state_dict)
    return m


def assign_grads(m, grads):
    state_dict = m.state_dict(keep_vars=True)
    index = 0
    for param in state_dict.keys():
        if 'running_mean' in param or 'running_var' in param or 'num_batches_tracked' in param:
                continue
        param_count = state_dict[param].numel()
        param_shape = state_dict[param].shape
        state_dict[param].grad =  grads[index:index+param_count].view(param_shape).clone()
        index += param_count
    m.load_state_dict(state_dict)
    return m

def get_norm_distance(m1, m2):
    m1 = flatten_params(m1)
    m2 = flatten_params(m2)
    return torch.norm(m1-m2, 2)


def get_cosine_similarity(m1, m2):
    m1 = flatten_params(m1)
    m2 = flatten_params(m2)
    cosine = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    return cosine(m1, m2)

def get_xy(point, origin, vector_x, vector_y):
    return np.array([np.dot(point - origin, vector_x), np.dot(point - origin, vector_y)])



def plot_interpolation(x, y, title, path):
    sns.set(style="whitegrid")
    sns.set_context("paper",rc={"lines.linewidth": 3,
                    'xtick.labelsize':18,
                    'ytick.labelsize':18,
                    'lines.markersize' : 15,
                    'legend.fontsize': 18,
                    'axes.labelsize': 18,
                    'legend.handlelength': 1,
                    'legend.handleheight':1,})

    color = 'green' if 'acc' in path else 'orange'
    suffix = 'acc' if 'acc' in path else 'loss'
    plt.plot(x, y, color=color)
    plt.xlabel('Interpolation')
    plt.ylabel('Val {}'.format(suffix))
    plt.title(title)
    # plt.ylim((0.0, 0.005))
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()



class LogNormalize(colors.Normalize):

    def __init__(self, vmin=None, vmax=None, clip=None, log_alpha=None):
        self.log_alpha = log_alpha
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        log_v = np.ma.log(value - self.vmin)
        log_v = np.ma.maximum(log_v, self.log_alpha)
        return 0.9 * (log_v - self.log_alpha) / (np.log(self.vmax - self.vmin) - self.log_alpha)


def contour_plot(grid, values, coords, vmax=None, log_alpha=-5, N=7, path='default.png', cmap='jet_r', lmc=False):
    rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
    
    cmap = plt.get_cmap(cmap)
    if vmax is None:
        clipped = values.copy()
    else:
        clipped = np.minimum(values, vmax)
    log_gamma = (np.log(clipped.max() - clipped.min()) - log_alpha) / N
    levels = clipped.min() + np.exp(log_alpha + log_gamma * np.arange(N + 1))
    levels[0] = clipped.min()
    levels[-1] = clipped.max()
    levels = np.sort(np.concatenate((levels, [1e10])))

    # print(levels)
    # norm = LogNormalize(clipped.min() - 1e-8, clipped.max() + 1e-8, log_alpha=log_alpha)
    contour = plt.contour(grid[:, :, 0], grid[:, :, 1], values, cmap=cmap,
                          linewidths=2.5,
                          zorder=1,
                          levels=10)
    contourf = plt.contourf(grid[:, :, 0], grid[:, :, 1], values, cmap=cmap,
                            levels=10,
                            zorder=0,
                            alpha=0.5)
    colorbar = plt.colorbar(format='%.2g')

    # labels = list(colorbar.ax.get_yticklabels())
    # labels[-1].set_text(r'$>\,$' + labels[-2].get_text())
    # colorbar.ax.set_yticklabels(labels)
    plt.scatter(coords[0, 0], coords[0, 1], marker='o', c='k', s=120, zorder=2)

    plt.text(coords[0, 0]+0.05, coords[0, 1]+0.05, r"$\hat{w}_1$", fontsize=20)
    

    plt.scatter(coords[1, 0], coords[1, 1], marker='o', c='k', s=120, zorder=2)
    if lmc:
        plt.text(coords[1, 0]+0.05, coords[1, 1]+0.05, r"$\hat{w}_\text{lmc}$", fontsize=20)
    else:
        plt.text(coords[1, 0]+0.05, coords[1, 1]+0.05, r"$\hat{w}_\text{mtl}$", fontsize=20)

    plt.scatter(coords[2, 0], coords[2, 1], marker='o', c='k', s=120, zorder=2)
    plt.text(coords[2, 0]+0.05, coords[2, 1]+0.05, r"$\hat{w}_2$", fontsize=20)


    
    plt.margins(0.0)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    colorbar.ax.tick_params(labelsize=18)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    m = MLP(10, 10)#ResNet18() #MLP(10, 10)

    for n, p in m.named_parameters():
        print(n, p.grad)
    flats = flatten_params(m, numpy_output=True)

    m = assign_weights_old(m , flats)
    print("**"*10)
    for n, p in m.named_parameters():
        print(n, p.grad)
    # print(flats.shape)
    # dic = m.state_dict()
    # count = 0
    # for p in dic.keys():
    #     # print(p, count)
    #     count += dic[p].numel()

    # count = 0
    # print('**'*20)
    # for n, p in m.named_parameters():
    #     print(n, count)
    #     count += p.numel()
    # print(count)

    # m = assign_weights(m, p)

#     # model = MLP(100, 10)
#     init = load_model('./checkpoints/YOGby/init.pth')
#     t1 = load_model('./checkpoints/YOGby/t_1_seq.pth')
#     w1 = flatten_params(init, True)
#     w2 = flatten_params(t1, True)

#     m = assign_grads(init, torch.tensor((w1+w2)/3.0))
#     for n, p in m.named_parameters():
#         print(n, p.grad)

    # w3 = flatten_params(m, True)

    # for check in [1, 1000, 12000, 25000, -1]:
    #     print((w1[check] + w2[check])/2.0, w3[check])
