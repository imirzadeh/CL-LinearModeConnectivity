import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from models import MLP,GatedMLP,Resnet20
from data_utils import get_rotated_mnist_tasks

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
		data = data.to(DEVICE)#.view(-1, 784)
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
			data = data.to(DEVICE)#.view(-1, 784)

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
	avg_acc = 100.0 * float(correct.numpy()) / len(loader.dataset)
	return {'accuracy': avg_acc, 'loss': test_loss}


def run():
	config = {'epochs': 3, 'lr': 0.1, 'tasks': 5, 'dropout': 0.25, 'batch-size': 32, 'num_condition_neurons': NUM_TASKS}
	# model = MLP(200, 10, config)
	# model = GatedMLP()
	model = Resnet20().to(DEVICE)
	tasks = get_rotated_mnist_tasks(config['tasks'], config['batch-size'])
	criterion = nn.CrossEntropyLoss().to(DEVICE)
	time = 0

	for i in range(1):
		print("***"*10 + "      iter {}     ".format(i+1) + "***"*10)
		for current_task_id in range(1, config['tasks']+1):
			train_loader = tasks[current_task_id]['train']
			for epoch in range(1, config['epochs']+1):
				optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=0.8)
				train_single_epoch(model, optimizer, train_loader, criterion, current_task_id)
				time += 1
				for prev_task_id in range(1, current_task_id+1):
					if epoch == config['epochs']:
						model = model.to(DEVICE)
						val_loader = tasks[prev_task_id]['test']
						metrics = eval_single_epoch(model, val_loader, criterion, prev_task_id)
						print("learning task:{}".format(current_task_id), "eval on task:{}".format(prev_task_id), '->',metrics)

if __name__ == "__main__":
	run()