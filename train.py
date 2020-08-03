import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from models import MLP,GatedMLP,Resnet20
from data_utils import get_rotated_mnist_tasks, get_multitask_rotated_mnist, get_rotated_mnist, get_subset_rotated_mnist
from utils import save_model, load_model, get_norm_distance, get_cosine_similarity
from utils import  plot, flatten_params, assign_weights, get_xy, contour_plot, get_random_string, assign_grads
import uuid
from pathlib import Path

NUM_TASKS = 3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRIAL_ID = 'RnmwZ' #get_random_string(5)
EXP_DIR = './checkpoints/{}'.format(TRIAL_ID)
BATCH_SIZE = 64
EPOCHS = 15

def train_single_epoch(net, optimizer, loader, criterion, task_id=None):
	net.train()
	#task_indicator = [0.0 for i in range(NUM_TASKS)]
	#task_indicator[task_id-1] = 1.0

	for batch_idx, (data, target) in enumerate(loader):
		data = data.to(DEVICE).view(-1, 784)
		target = target.to(DEVICE)

		# print(data.shape)
		# task_indicator_data = torch.tensor([task_indicator for i in range(target.shape[0])])
		# data = torch.cat((data, task_indicator_data), 1)
		
		# target_task_indicator = torch.tensor([(task_id - 1) for i in range(target.shape[0])], dtype=torch.long)
		# target = torch.cat((target, target_task_indicator), 1)

		optimizer.zero_grad()
		pred = net(data, task_id)

		loss = criterion(pred, target)
		loss.backward()
		optimizer.step()
	return net


def eval_single_epoch(net, loader, criterion, task_id=None):
	
	net.eval()
	test_loss = 0
	correct = 0

	count = 0
	with torch.no_grad():
		for data, target in loader:
			data = data.to(DEVICE).view(-1, 784)
			target = target.to(DEVICE)
			count += len(data)


			output = net(data, task_id)
			test_loss += criterion(output, target).item()
			pred = output.data.max(1, keepdim=True)[1]
			# if target.shape[0] < 256:
				# print(pred)
			correct += pred.eq(target.data.view_as(pred)).sum()
	test_loss /= count
	correct = correct.to('cpu')
	# print(correct.item())
	avg_acc = 100.0 * float(correct.numpy()) / count
	return {'accuracy': avg_acc, 'loss': test_loss}


def setup_experiment():
	Path(EXP_DIR).mkdir(parents=True, exist_ok=True)
	init_model = MLP(100, 10)
	save_model(init_model, '{}/init.pth'.format(EXP_DIR))

def get_stage_loaders(stage, train=True):
	idx = 0 if train == True else 1
	if stage == 't1':
		return get_rotated_mnist(1, BATCH_SIZE)[idx]
	elif stage == 't2':
		return get_rotated_mnist(2, BATCH_SIZE)[idx]
	elif stage == 't12':
		return get_multitask_rotated_mnist(2, BATCH_SIZE, 25000)[idx]
	else:
		raise Exception("unknown stage")

def train_stage(stage):
	stage_dependencies = {'t1': 'init', 't2': 't1', 't12': 't1'}
	joint_training = True if stage == 't12' else False

	model = load_model('{}/{}.pth'.format(EXP_DIR,stage_dependencies[stage]))
	train_loader = get_stage_loaders(stage)
	criterion = nn.CrossEntropyLoss().to(DEVICE)
	optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.8)

	for epoch in range(EPOCHS):
		model = train_single_epoch(model, optimizer, train_loader, criterion)
	save_model(model, '{}/{}.pth'.format(EXP_DIR, stage))

def eval_stage(stage):
	criterion = nn.CrossEntropyLoss().to(DEVICE)
	eval_dependencies = {'t1': ['t1'], 't2': ['t1', 't2'], 't12': ['t1', 't2']}
	model = load_model('{}/{}.pth'.format(EXP_DIR, stage))
	metrics = {}
	for eval_stage in eval_dependencies[stage]:
		eval_loader = get_stage_loaders(eval_stage)
		metrics[eval_stage] = eval_single_epoch(model, eval_loader, criterion)
	return metrics


def compare_distances(stage_1, stage_2):
	m1 = load_model('{}/{}.pth'.format(EXP_DIR, stage_1))
	m2 = load_model('{}/{}.pth'.format(EXP_DIR, stage_2))
	m1.eval()
	m2.eval()
	return get_norm_distance(m1, m2).numpy()

def train_stages():
	for stage in ['t1', 't2', 't12']:
		print(" --------------- stage {} ------------".format(stage))
		stage_model = train_stage(stage)
		eval_stage_result = eval_stage(stage)
		print(eval_stage_result)

def get_stats_after_train_stages():
	for stage_pairs in [['t1', 't2'], ['t1', 't12'], ['t2', 't12']]:
		print('dist {} and {} => {}'.format(stage_pairs[0], stage_pairs[1], compare_distances(stage_pairs[0], stage_pairs[1])))

def check_mode_connectivity(stage_1, stage_2):
	print('mode connectivity {} and {}'.format(stage_1, stage_2))
	m1 = load_model('{}/{}.pth'.format(EXP_DIR, stage_1))
	m2 = load_model('{}/{}.pth'.format(EXP_DIR, stage_2))

	m1 = flatten_params(m1, numpy_output=True)
	m2 = flatten_params(m2, numpy_output=True)
	metrics = []
	criterion = nn.CrossEntropyLoss().to(DEVICE)
	eval_loader = get_stage_loaders(stage_1, train=False)
	m = load_model('{}/{}.pth'.format(EXP_DIR, 'init'))
	for t in np.arange(0.0, 1.01, 0.1):
		cur_weight = m1 + (m2 - m1) * t
		m = assign_weights(m, cur_weight)
		eval_metric = eval_single_epoch(m, eval_loader, criterion)
		metrics.append([t, eval_metric['loss'], eval_metric['accuracy']])
		print(t, eval_metric)
	return metrics

def get_line_loss(stage, w, loader):
	m1 = load_model('{}/{}.pth'.format(EXP_DIR, stage))
	m1 = flatten_params(m1, numpy_output=True)
	m = load_model('{}/{}.pth'.format(EXP_DIR, 'init'))
	# loader = get_multitask_rotated_mnist(2, BATCH_SIZE, 200)[0]
	total_loss = 0
	accum_grad = None
	for t in np.arange(0.0, 1.01, 0.1):
		grads = []
		cur_weight = m1 + (w - m1) * t
		m = assign_weights(m, cur_weight)
		current_loss = get_clf_loss(m, loader)
		current_loss.backward()
		for param in m.parameters():
			grads.append(param.grad.view(-1))
		grads = torch.cat(grads)
		if accum_grad is None:
			accum_grad = grads
		else:
			accum_grad += grads
		# total_loss += current_loss
		# print(total_loss.item())
	# return total_loss
	return accum_grad


def get_clf_loss(net, loader):
	criterion = nn.CrossEntropyLoss().to(DEVICE)
	net.eval()
	test_loss = 0
	count = 0

	
	for data, target in loader:
			count += len(data)
			data = data.to(DEVICE).view(-1, 784)
			target = target.to(DEVICE)
			output = net(data)
			test_loss += criterion(output, target)
	# print("dataset size is => ", count)
	test_loss /= count
	return test_loss


def printw(w, count=30):
	print([round(x, 4) for x in w[:count]])

def experimental_approach():
	m1 = load_model('{}/{}.pth'.format(EXP_DIR, 't1'))
	m2 = load_model('{}/{}.pth'.format(EXP_DIR, 't2'))
	goal = load_model('{}/{}.pth'.format(EXP_DIR, 't12'))

	w1 = flatten_params(m1, numpy_output=True)
	w2 = flatten_params(m2, numpy_output=True)

	m = load_model('{}/{}.pth'.format(EXP_DIR, 'init'))
	m = assign_weights(m, (w1 + w2)/2.0)



	m.train()
	optimizer = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.8)
	criterion = nn.CrossEntropyLoss().to(DEVICE)
	# criterion = nn.MSELoss().to(DEVICE)
	loader = get_multitask_rotated_mnist(2, BATCH_SIZE, 200)[0]
	t1_loader = get_subset_rotated_mnist(1, BATCH_SIZE, 200)[0]
	t2_loader = get_subset_rotated_mnist(2, BATCH_SIZE, 200)[0]
	print(get_norm_distance(m, goal))
	for epoch in range(1, 50):
		# m = train_joint_model(m, m1, m2, loader)
		# m = train_single_epoch(m, optimizer, loader, criterion)
		optimizer.zero_grad()
		grads = get_line_loss('t1', flatten_params(m, numpy_output=True), t1_loader) \
			  + get_line_loss('t2', flatten_params(m, numpy_output=True), t2_loader)
		assign_grads(m, grads.numpy())
		# loss = get_clf_loss(m, loader)
		
		# loss.backward()
		optimizer.step()

		print('epoch {}, distance: {}'.format(epoch, get_norm_distance(m, goal)))

		# print('loss', loss.item())
		print('t1', eval_single_epoch(m, get_rotated_mnist(1, 256)[1], criterion))
		print('t2', eval_single_epoch(m, get_rotated_mnist(2, 256)[1], criterion))
		print('norm change', np.linalg.norm(flatten_params(m, numpy_output=True)-((w1+w2)/2.0)))
		print()
	# criterion = nn.CrossEntropyLoss().to(DEVICE)
	# print(eval_single_epoch(m, get_stage_loaders('t1', train=False), criterion))
	# print(eval_single_epoch(m, get_stage_loaders('t2', train=False), criterion))
	# print(get_norm_distance(m, goal).numpy())


def plot_loss():
	criterion = nn.CrossEntropyLoss().to(DEVICE)

	m1 = load_model('{}/{}.pth'.format(EXP_DIR, 't1'))
	m3 = load_model('{}/{}.pth'.format(EXP_DIR, 't2'))
	m2 = load_model('{}/{}.pth'.format(EXP_DIR, 't12'))

	w = [flatten_params(p, numpy_output=True) for p in [m1, m2, m3]]

	u = w[2] - w[0]
	dx = np.linalg.norm(u)
	u /= dx

	v = w[1] - w[0]
	v -= np.dot(u, v) * u
	dy = np.linalg.norm(v)
	v /= dy

	m = load_model('{}/{}.pth'.format(EXP_DIR, 'init'))
	m.eval()
	coords = np.stack(get_xy(p, w[0], u, v) for p in w)
	print("coords", coords)

	# w0_coord = get_xy(w[0], w[0], u, v)
	# w1_coord = get_xy(w[1], w[0], u, v)
	# w2_coor = get_xy(w[2], w[0], u, v)

	G = 10
	margin = 0.2
	alphas = np.linspace(0.0 - margin, 1.0 + margin, G)
	betas = np.linspace(0.0 - margin, 1.0 + margin, G)
	tr_loss = np.zeros((G, G))
	grid = np.zeros((G, G, 2))

	loader = get_multitask_rotated_mnist(2, BATCH_SIZE, 200)[0]

	for i, alpha in enumerate(alphas):
		for j, beta in enumerate(betas):
			p = w[0] + alpha * dx * u + beta * dy * v
			m = assign_weights(m, p)
			err = get_line_loss('t1', flatten_params(m, numpy_output=True), loader).item() + get_line_loss('t2', flatten_params(m, numpy_output=True), loader).item()#eval_single_epoch(m, val_loader, criterion, 1)['loss']
			c = get_xy(p, w[0], u, v)
			print(c)
			grid[i, j] = [alpha * dx, beta * dy]
			tr_loss[i, j] = err

	contour_plot(grid, tr_loss, coords, vmax=5.0, log_alpha=-5.0, N=7)


def main():
	# init and save
	#setup_experiment()	

	# convention:  init = zero init
	# convention:  t1   = task 1
	# convention:  t2   = task 2
	# convention:  t12  = task 1 & task 2 (multitask)
	# train_stages()
	# check_mode_connectivity('t2', 't12')
	experimental_approach()
	# plot_loss()

if __name__ == "__main__":
	main()