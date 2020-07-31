import numpy as np
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader, Dataset, ConcatDataset
from torch.utils.data.sampler import Sampler, RandomSampler
import torchvision.transforms.functional as TorchVisionFunc
from torchvision.datasets import MNIST


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class RotationTransform:
	"""
	Rotation transforms for the images in `Rotation MNIST` dataset.
	"""
	def __init__(self, angle):
		self.angle = angle

	def __call__(self, x):
		return TorchVisionFunc.rotate(x, self.angle, fill=(0,))


def get_rotated_mnist_tasks(num_tasks, batch_size):
	"""
	Returns data loaders for all tasks of rotation MNIST dataset.
	:param num_tasks: number of tasks in the benchmark.
	:param batch_size:
	:return:
	"""
	datasets = {}
	for task_id in range(1, num_tasks+1):
		train_loader, test_loader = get_rotated_mnist(task_id, batch_size)
		datasets[task_id] = {'train': train_loader, 'test': test_loader}
	return datasets


def get_rotated_mnist(task_id, batch_size):
	"""
	Returns the dataset for a single task of Rotation MNIST dataset
	:param task_id:
	:param batch_size:
	:return:
	"""
	per_task_rotation = 30
	rotation_degree = (task_id - 1)*per_task_rotation

	transforms = torchvision.transforms.Compose([
		RotationTransform(rotation_degree),
		torchvision.transforms.ToTensor(),
		])

	train_loader = torch.utils.data.DataLoader(MNIST('./data/', train=True, download=True, transform=transforms), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(MNIST('./data/', train=False, download=True, transform=transforms),  batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

	return train_loader, test_loader


def get_multitask_rotated_mnist(num_tasks, batch_size, num_examples_per_task=50000):
	per_task_rotation = 30
	
	trains = []
	tests = []
	for i in range(1, num_tasks+1):
		rotation_degree = (i - 1)*per_task_rotation
		# rotation_degree -= (np.random.random()*per_task_rotation)

		transforms = torchvision.transforms.Compose([
		RotationTransform(rotation_degree),
		torchvision.transforms.ToTensor(),
		])
		train = MNIST('./data/', train=True, download=True, transform=transforms)
		test = MNIST('./data/', train=False, download=True, transform=transforms)

		trains.append(train)
		tests.append(test)

	train_datasets = ConcatDataset(trains)
	test_datasets = ConcatDataset(tests)


	if num_examples_per_task == 50000:
		sampler = None
	else:
		num_examples = num_examples_per_task * num_tasks
		sampler = torch.utils.data.RandomSampler(train_datasets, replacement=True, num_samples=num_examples)

	train_loader = torch.utils.data.DataLoader(train_datasets,  batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=4, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(test_datasets,  batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

	return train_loader, test_loader

# class FastMNIST(MNIST):
# 	def __init__(self, *args, **kwargs):
# 		super().__init__(*args, **kwargs)
# 		self.data = self.data.unsqueeze(1).float().div(255.0)
# 		self.data, self.targets = self.data.to('cpu'), self.targets.to('cpu')

# 	def __getitem__(self, index):
# 		"""
# 		Args:
# 			index (int): Index

# 		Returns:
# 			tuple: (image, target) where target is index of the target class.
# 		"""
# 		img, target = self.data[index], self.targets[index]
# 		if self.transform:
# 			img = self.transform(img)
# 		return img, target


# if __name__ == "__main__":
# 	train = get_rotated_mnist_tasks(5, 100)[3]['train']
# 	num_row = 5
# 	num_col = 5
   
# 	data, taget = next(iter(train))
# 	fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
# 	for i in range(25):
# 		ax = axes[i//num_row, i%num_col]
# 		ax.imshow(data[i].numpy()[0], cmap='gray')
# 		ax.set_title('Task: {}'.format(3))
# 	plt.tight_layout()
# 	plt.savefig('datasets.png', dpi=300)

