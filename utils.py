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


def flatten_params(m, numpy_output=False):
	params = torch.cat((torch.flatten(m.W1.weight.data), torch.flatten(m.W1.bias.data), torch.flatten(m.W2.weight.data), torch.flatten(m.W2.bias.data), torch.flatten(m.W3.weight.data), torch.flatten(m.W3.bias.data)))
	# print(params.shape)
	if numpy_output:
		return params.cpu().numpy()
	else:
		return params

def assign_grads(m, grads):
	p = [784, 100, 100, 10]
	idx =[0]
	for i in range(len(p)-1):
		last = idx[-1]
		w_i = last + p[i] * p[i+1]
		b_i = last + p[i] * p[i+1] + p[i+1]
		idx.append(w_i)
		idx.append(b_i)

	# print(idx)
	W1 = grads[idx[0]:idx[1]].view(p[1], p[0])
	b1 = grads[idx[1]:idx[2]]

	W2 = grads[idx[2]:idx[3]].view(p[2], p[1])
	b2 = grads[idx[3]:idx[4]]

	W3 = grads[idx[4]:idx[5]].view(p[3], p[2])
	b3 = grads[idx[5]:idx[6]]

	# W1 = torch.from_numpy(W1)
	m.W1.weight.grad = W1.clone()
	# b1 = torch.from_numpy(b1)
	m.W1.bias.grad = b1.clone()

	# W2 = torch.from_numpy(W2)
	m.W2.weight.grad = W2.clone()
	# b2 = torch.from_numpy(b2)
	m.W2.bias.grad = b2.clone()

	# W3 = torch.from_numpy(W3)
	m.W3.weight.grad = W3.clone()
	# b3 = torch.from_numpy(b3)
	m.W3.bias.grad = b3.clone()
	return m


def assign_weights(m, params):
	p = [784, 100, 100, 10]
	idx =[0]
	for i in range(len(p)-1):
		last = idx[-1]
		w_i = last + p[i] * p[i+1]
		b_i = last + p[i] * p[i+1] + p[i+1]
		idx.append(w_i)
		idx.append(b_i)

	# print(idx)
	W1 = params[idx[0]:idx[1]].reshape(p[1], p[0])
	b1 = params[idx[1]:idx[2]]

	W2 = params[idx[2]:idx[3]].reshape(p[2], p[1])
	b2 = params[idx[3]:idx[4]]

	W3 = params[idx[4]:idx[5]].reshape(p[3], p[2])
	b3 = params[idx[5]:idx[6]]

	with torch.no_grad():
			W1 = torch.from_numpy(W1)
			m.W1.weight = nn.Parameter(W1)
			b1 = torch.from_numpy(b1)
			m.W1.bias = nn.Parameter(b1)

			W2 = torch.from_numpy(W2)
			m.W2.weight = nn.Parameter(W2)
			b2 = torch.from_numpy(b2)
			m.W2.bias = nn.Parameter(b2)

			W3 = torch.from_numpy(W3)
			m.W3.weight = nn.Parameter(W3)
			b3 = torch.from_numpy(b3)
			m.W3.bias = nn.Parameter(b3)
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



def plot(x, y, title, path):
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
	plt.xlabel('interpolation')
	plt.ylabel('validation {}'.format(suffix))
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


def contour_plot(grid, values, coords, vmax=None, log_alpha=-5, N=7, cmap='jet_r'):
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
	plt.text(coords[0, 0]+0.5, coords[0, 1]+0.5, r"$\hat{w}_1$", fontsize=20)
	
	plt.scatter(coords[1, 0], coords[1, 1], marker='o', c='k', s=120, zorder=2)
	plt.text(coords[1, 0]+0.5, coords[1, 1]+0.5, r"$\hat{w}_\text{joint}$", fontsize=20)

	plt.scatter(coords[2, 0], coords[2, 1], marker='o', c='k', s=120, zorder=2)
	plt.text(coords[2, 0]+0.5, coords[2, 1]+0.5, r"$\hat{w}_2$", fontsize=20)


	
	plt.margins(0.0)
	plt.yticks(fontsize=18)
	plt.xticks(fontsize=18)
	colorbar.ax.tick_params(labelsize=18)
	plt.savefig('loss_plane.png', dpi=300, bbox_inches='tight')
	plt.close()

