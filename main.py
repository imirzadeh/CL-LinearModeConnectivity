import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from models import MLP,GatedMLP,Resnet20
from data_utils import get_rotated_mnist_tasks, get_multitask_rotated_mnist
from utils import save_model, load_model, get_norm_distance, get_cosine_similarity
from utils import  plot, flatten_params, assign_weights, get_xy, contour_plot

DEVICE = 'cpu'
NUM_TASKS = 5

def train_single_epoch(net, optimizer, loader, criterion, task_id):
	"""
	Train the model for a single epoch
	
	:param net:
	:param optimizer:
	:param loader:
	:param criterion:
	:param task_id:
	:return:
	"""
	net.train()
	task_indicator = [0.0 for i in range(NUM_TASKS)]
	task_indicator[task_id-1] = 1.0

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


def eval_single_epoch(net, loader, criterion, task_id):
	"""
	Evaluate the model for single epoch
	
	:param net:
	:param loader:
	:param criterion:
	:param task_id:
	:return:
	"""
	net.eval()
	test_loss = 0
	correct = 0
	task_indicator = [0.0 for i in range(NUM_TASKS)]
	task_indicator[task_id-1] = 1.0

	with torch.no_grad():
		for data, target in loader:
			data = data.to(DEVICE).view(-1, 784)
			target = target.to(DEVICE)
			# task_indicator_data = torch.tensor([task_indicator for i in range(target.shape[0])])
			# data = torch.cat((data, task_indicator_data), 1)
		
			# target_task_indicator = torch.tensor([(task_id - 1) for i in range(target.shape[0])], dtype=torch.long)
			# target = torch.cat((target, target_task_indicator), 1)

			output = net(data, task_id)
			test_loss += criterion(output, target).item()
			pred = output.data.max(1, keepdim=True)[1]
			# if target.shape[0] < 256:
				# print(pred)
			correct += pred.eq(target.data.view_as(pred)).sum()
	test_loss /= len(loader.dataset)
	correct = correct.to('cpu')
	# print(correct.item())
	avg_acc = 100.0 * float(correct.numpy()) / len(loader.dataset)
	return {'accuracy': avg_acc, 'loss': test_loss}


def multitask_run(checkpoint):
	DIR = './checkpoints/{}/'.format(checkpoint)

	config = {'epochs': 10, 'lr': 0.05, 'tasks': 2, 'dropout': 0.1, 'batch-size': 32}
	# model = MLP(100, 10, config)
	model = load_model(DIR+'single-1.pth')
	train_loader, val_loader = get_multitask_rotated_mnist(2, batch_size=32, num_examples_per_task=5000)
	criterion = nn.CrossEntropyLoss().to(DEVICE)
	optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=0.8)

	for i in range(config['epochs']):
		train_single_epoch(model, optimizer, train_loader, criterion, 1)
		metrics = eval_single_epoch(model, val_loader, criterion, 1)
		print(metrics)
	save_model(model, DIR+'mtl.pth')


def run(checkpoint):
	policy = 'seq'
	config = {'epochs': 5, 'lr': 0.1, 'tasks': 5, 'dropout': 0.25, 'batch-size': 32}
	# model = MLP(100, 10, config)
	# save_model(model, './checkpoints/{}/init.pth'.format(checkpoint))
	DIR = './checkpoints/{}/'.format(checkpoint)
	model = load_model(DIR+'single-1.pth')

	tasks = get_rotated_mnist_tasks(config['tasks'], config['batch-size'])
	criterion = nn.CrossEntropyLoss().to(DEVICE)
	time = 0

	for i in range(1):
		print("***"*10 + "      iter {}     ".format(i+1) + "***"*10)
		for current_task_id in [2]:#range(1, config['tasks']+1):
			model = load_model(DIR+'single-1.pth')
			train_loader, val_loader = tasks[current_task_id]['train'], tasks[current_task_id]['test']
			for epoch in range(1, config['epochs']+1):
				optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=0.8)
				train_single_epoch(model, optimizer, train_loader, criterion, current_task_id)
				time += 1
				save_model(model, DIR+'{}-{}.pth'.format(policy, current_task_id))
			metrics = eval_single_epoch(model, val_loader, criterion, current_task_id)
			print(metrics)
				# for prev_task_id in range(1, current_task_id+1):
				# 	if epoch == config['epochs']:
				# 		model = model.to(DEVICE)
				# 		val_loader = tasks[prev_task_id]['test']
				# 		metrics = eval_single_epoch(model, val_loader, criterion, prev_task_id)
				# 		print("learning task:{}".format(current_task_id), "eval on task:{}".format(prev_task_id), '->',metrics)


def compare(checkpoint):
	DIR = './checkpoints/{}/'.format(checkpoint)
	mtl_model = load_model(DIR+'mtl.pth')
	mtl_model.eval()
	seq_models = [None, ]
	for i in range(1, 6):
		m = load_model(DIR+'seq-{}.pth'.format(i))
		m.eval()
		seq_models.append(m)

	for i in range(1, 6):
		norm_distance = get_norm_distance(seq_models[i], mtl_model)
		cosine_similarity = get_cosine_similarity(seq_models[i], mtl_model)
		print('||seq{} - mtl|| = {:.2f}'.format(i, norm_distance))
		print('cos(seq{} , mtl) = {:.3f}'.format(i, cosine_similarity))
		print()

	for i in range(1, 5):
		for j in range(i, 6):
			norm_distance = get_norm_distance(seq_models[i], seq_models[j])
			print('||seq{} - seq{}|| = {:.2f}'.format(i, j, norm_distance))
		print()


def plot_connection(checkpoint):
	# m1, m2, loader, criterion

	start_task = 1
	target_task = 'mtl'
	DIR = './checkpoints/{}/'.format(checkpoint)
	
	m1 = load_model(DIR+'single-{}.pth'.format(start_task))
	m2 = load_model(DIR+'{}.pth'.format(target_task))

	m = load_model(DIR+'single-{}.pth'.format(start_task))

	m1.eval()
	m2.eval()

	criterion = nn.CrossEntropyLoss().to(DEVICE)
	val_loader = get_rotated_mnist_tasks(5, 128)[1]['test']

	W1_diff = m2.W1.weight.data.numpy() - m1.W1.weight.data.numpy()
	b1_diff = m2.W1.bias.data.numpy() - m1.W1.bias.data.numpy()
	
	W2_diff = m2.W2.weight.data.numpy() - m1.W2.weight.data.numpy()
	b2_diff = m2.W2.bias.data.numpy() - m1.W2.bias.data.numpy()

	W3_diff = m2.W3.weight.data.numpy() - m1.W3.weight.data.numpy()
	b3_diff = m2.W3.bias.data.numpy() - m1.W3.bias.data.numpy()

	losses = []
	accs = []
	coeffs = np.arange(-0.05, 1.06, 0.05)
	for coeff in coeffs:
		with torch.no_grad():
			W1 = torch.from_numpy(m1.W1.weight.data.numpy() + coeff*W1_diff)
			m.W1.weight = nn.Parameter(W1)
			b1 = torch.from_numpy(m1.W1.bias.data.numpy() + coeff*b1_diff)
			m.W1.bias = nn.Parameter(b1)

			# if 0.999 < coeff < 1.001:
			# 	print(b1)
			# 	print(m2.W1.bias.data.numpy())

			W2 = torch.from_numpy(m1.W2.weight.data.numpy() + coeff*W2_diff)
			m.W2.weight = nn.Parameter(W2)
			b2 = torch.from_numpy(m1.W2.bias.data.numpy() + coeff*b2_diff)
			m.W2.bias = nn.Parameter(b2)

			W3 = torch.from_numpy(m1.W3.weight.data.numpy() + coeff*W3_diff)
			m.W3.weight = nn.Parameter(W3)
			b3 = torch.from_numpy(m1.W3.bias.data.numpy() + coeff*b3_diff)
			m.W3.bias = nn.Parameter(b3)

			metrics = eval_single_epoch(m, val_loader, criterion, 1)
			losses.append(metrics['loss'])
			accs.append(100.0-metrics['accuracy'])
			print(coeff, metrics)
	plot(coeffs, losses, 'Task {}'.format(start_task), '{}-{}-loss.png'.format(start_task, target_task))
	plot(coeffs, accs, 'Task {}'.format(start_task), '{}-{}-acc.png'.format(start_task, target_task))

def plot_loss(checkpoint):
	DIR = './checkpoints/{}/'.format(checkpoint)
	criterion = nn.CrossEntropyLoss().to(DEVICE)
	val_loader = get_rotated_mnist_tasks(5, 256)[2]['test']

	m1 = load_model(DIR+'single-{}.pth'.format(1))
	m2 = load_model(DIR+'{}.pth'.format('mtl'))
	m3 = load_model(DIR+'seq-{}.pth'.format(2))
	w = [flatten_params(p, numpy_output=True) for p in [m1, m2, m3]]

	u = w[2] - w[0]
	dx = np.linalg.norm(u)
	u /= dx

	v = w[1] - w[0]
	v -= np.dot(u, v) * u
	dy = np.linalg.norm(v)
	v /= dy

	m = load_model(DIR+'{}.pth'.format('init'))
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

	for i, alpha in enumerate(alphas):
		for j, beta in enumerate(betas):
			p = w[0] + alpha * dx * u + beta * dy * v
			m = assign_weights(m, p)
			err = eval_single_epoch(m, val_loader, criterion, 1)['loss']
			c = get_xy(p, w[0], u, v)
			print(c)
			grid[i, j] = [alpha * dx, beta * dy]
			tr_loss[i, j] = err

	contour_plot(grid, tr_loss, coords, vmax=5.0, log_alpha=-5.0, N=7)


def Bezier_curve(t, theta, w1, w2):
	return (1.0-t)**2 * w1 + 2*t*(1-t)*theta + t**2 * w2

def connect(chekpoint):
	DIR = './checkpoints/{}/'.format(checkpoint)
	criterion = nn.CrossEntropyLoss().to(DEVICE)
	tasks = get_rotated_mnist_tasks(5, 256)
	train_loader , val_loader = get_multitask_rotated_mnist(1, 16, 5000)

	m1 = load_model(DIR+'single-{}.pth'.format(1))
	m2 = load_model(DIR+'mtl.pth'.format(2))

	w1 = flatten_params(m1, numpy_output=True)
	w2 = flatten_params(m2, numpy_output=True)

	m = load_model(DIR+'single-{}.pth'.format(1))
	theta = flatten_params(m, numpy_output=True)
	optimizer = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.8)

	for epoch in range(5):
		s = 0
		print('*********** epoch {} ***********'.format(epoch+1))
		for t in np.arange(0.0, 1.01, 0.05):
			s += 1
			phi = Bezier_curve(t, theta, w1, w2)
			assign_weights(m, phi)
			train_single_epoch(m, optimizer, train_loader, criterion, 1)
			metrics = eval_single_epoch(m, val_loader, criterion, 1)
			print(round(t,2), metrics)




def eval(checkpoint):
	DIR = './checkpoints/{}/'.format(checkpoint)
	m = load_model(DIR+'{}.pth'.format('mtl'))
	criterion = nn.CrossEntropyLoss().to(DEVICE)
	val_loader = get_rotated_mnist_tasks(5, 128)[1]['test']
	print(eval_single_epoch(m, val_loader, criterion, 1))

if __name__ == "__main__":
	# run()
	# checkpoint = 1
	# run(checkpoint)
	# connect(checkpoint)
	# run(checkpoint)
	# plot_connection(checkpoint)
	# eval(checkpoint)
	# run(checkpoint)
	# multitask_run(checkpoint)
	# compare(checkpoint)
	# plot_loss(checkpoint)