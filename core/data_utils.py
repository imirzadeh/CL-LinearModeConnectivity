import numpy as np
import random
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader, Dataset, ConcatDataset
from torch.utils.data.sampler import Sampler, RandomSampler
import torchvision.transforms.functional as TorchVisionFunc
from torchvision.datasets import MNIST


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# PER_TASK_ROATATION = 9

def fast_mnist_loader(loaders, device='cpu'):
    train_loader, eval_loader = loaders
    trains, evals = [], []
    for data, target in train_loader:
        data = data.to(device).view(-1, 784)
        target = target.to(device)
        trains.append([data, target, None])

    for data, target in eval_loader:
        data = data.to(device).view(-1, 784)
        target = target.to(device)
        evals.append([data, target, None])

    return trains, evals


def fast_cifar_loader(loaders, task_id, device='cpu'):
    train_loader, eval_loader = loaders
    trains, evals = [], []
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        trains.append([data, target, task_id])

    for data, target in eval_loader:
        data = data.to(device)
        target = target.to(device)
        evals.append([data, target, task_id])

    return trains, evals

class RotationTransform:
    """
    Rotation transforms for the images in `Rotation MNIST` dataset.
    """
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return TorchVisionFunc.rotate(x, self.angle, fill=(0,))


def get_rotated_mnist(task_id, batch_size, per_task_rotation):
    """
    Returns the dataset for a single task of Rotation MNIST dataset
    :param task_id:
    :param batch_size:
    :return:
    """
    # per_task_rotation = PER_TASK_ROATATION
    rotation_degree = (task_id - 1)*per_task_rotation

    transforms = torchvision.transforms.Compose([
        RotationTransform(rotation_degree),
        torchvision.transforms.ToTensor(),
        ])

    train_loader = torch.utils.data.DataLoader(MNIST('./data/', train=True, download=True, transform=transforms), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(MNIST('./data/', train=False, download=True, transform=transforms),  batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader


def get_subset_rotated_mnist(task_id, batch_size, num_examples, per_task_rotation):
    # per_task_rotation = PER_TASK_ROATATION

    trains = []
    tests = []
    for i in [task_id]:
        rotation_degree = (i - 1)*per_task_rotation
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

    # num_examples = num_examples_per_task * num_tasks
    sampler = torch.utils.data.RandomSampler(train_datasets, replacement=True, num_samples=num_examples)

    train_loader = torch.utils.data.DataLoader(train_datasets,  batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_datasets,  batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

    return train_loader, test_loader

def get_multitask_rotated_mnist(num_tasks, batch_size, num_examples, per_task_rotation):
    num_examples_per_task = num_examples//num_tasks

    trains = []
    tests = []
    all_mtl_data = {}

    for task in range(1, num_tasks+1):
        all_mtl_data[task] = {}
        train_loader, test_loader = fast_mnist_loader(get_subset_rotated_mnist(task, batch_size, num_examples_per_task, per_task_rotation))
        trains += train_loader
        tests += test_loader
        all_mtl_data[task]['train'] = random.sample(trains[:], len(trains))
        all_mtl_data[task]['val'] = tests[:]


    return all_mtl_data


def get_split_cifar100(task_id, batch_size, cifar_train, cifar_test):
    """
    Returns a single task of split CIFAR-100 dataset
    :param task_id:
    :param batch_size:
    :return:
    """

    start_class = (task_id-1)*5
    end_class = task_id * 5

    targets_train = torch.tensor(cifar_train.targets)
    target_train_idx = ((targets_train >= start_class) & (targets_train < end_class))
    
    targets_test = torch.tensor(cifar_test.targets)
    target_test_idx = ((targets_test >= start_class) & (targets_test < end_class))

    train_loader = torch.utils.data.DataLoader(torch.utils.data.dataset.Subset(cifar_train, np.where(target_train_idx==1)[0]), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.dataset.Subset(cifar_test, np.where(target_test_idx==1)[0]), batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def get_subset_split_cifar100(task_id, batch_size, cifar_train, num_examples):
    """
    Returns a single task of split CIFAR-100 dataset
    :param task_id:
    :param batch_size:
    :return:
    """

    start_class = (task_id-1)*5
    end_class = task_id * 5

    per_class_examples = num_examples//5

    targets_train = torch.tensor(cifar_train.targets)

    # target_train_idx = ((targets_train >= start_class) & (targets_train < end_class))
    # train_dataset = torch.utils.data.dataset.Subset(cifar_train, np.where(target_train_idx==1)[0])
    
    trains = []
    for class_number in range(start_class, end_class):
        target = (targets_train == class_number)
        class_train_idx = np.random.choice(np.where(target == 1)[0], per_class_examples, False)
        current_class_train_dataset = torch.utils.data.dataset.Subset(cifar_train, class_train_idx)
        trains.append(current_class_train_dataset)              

    trains = ConcatDataset(trains)
    train_loader = torch.utils.data.DataLoader(trains, batch_size=batch_size, shuffle=True)

    return train_loader, []



def get_multitask_cifar100_loaders(num_tasks, batch_size, num_examples):
    num_examples_per_task = num_examples//num_tasks
    trains = []
    tests = []
    all_mtl_data = {}
    cifar_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
    cifar_train = torchvision.datasets.CIFAR100('./data/', train=True, download=True, transform=cifar_transforms)

    for task in range(1, num_tasks+1):
        all_mtl_data[task] = {}
        train_loader, test_loader = fast_cifar_loader(get_subset_split_cifar100(task, batch_size, cifar_train, num_examples_per_task), task)
        trains += train_loader
        tests += test_loader
        all_mtl_data[task]['train'] = trains[:]
        all_mtl_data[task]['val'] = tests[:]
    return all_mtl_data


def get_all_loaders(dataset, num_tasks, bs_inter, bs_intra, num_examples, per_task_rotation=9):
    dataset = dataset.lower()
    loaders = {'sequential': {},  'multitask':  {}, 'subset': {}, 'full-multitask': {}}

    print('loading multitask {}'.format(dataset))
    if 'cifar' in dataset:
        loaders['multitask'] = get_multitask_cifar100_loaders(num_tasks, bs_inter, num_examples)
        loaders['full-multitask'] = get_multitask_cifar100_loaders(num_tasks, bs_inter, num_tasks*5*500)
    elif 'rot' in dataset and 'mnist' in dataset:
        loaders['multitask'] = get_multitask_rotated_mnist(num_tasks, bs_inter, num_examples, per_task_rotation)
        loaders['full-multitask'] = get_multitask_rotated_mnist(num_tasks, bs_inter, num_tasks*10*2000, per_task_rotation)

    else:
        raise Exception("{} not implemented!".format(dataset))

    # cache cifar
    if 'cifar' in dataset:
        cifar_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
        cifar_train = torchvision.datasets.CIFAR100('./data/', train=True, download=True, transform=cifar_transforms)
        cifar_test = torchvision.datasets.CIFAR100('./data/', train=False, download=True, transform=cifar_transforms)

    # Load sequential tasks
    for task in range(1, num_tasks+1):
        loaders['sequential'][task], loaders['subset'][task] = {}, {}
        print("loading {} for task {}".format(dataset, task))
        if 'rot' in dataset and 'mnist' in dataset:
            seq_loader_train , seq_loader_val = fast_mnist_loader(get_rotated_mnist(task, bs_intra, per_task_rotation), 'cpu')
            sub_loader_train , _ = fast_mnist_loader(get_subset_rotated_mnist(task, bs_inter, num_examples, per_task_rotation),'cpu')

        elif 'cifar' in dataset:
            seq_loader_train , seq_loader_val = fast_cifar_loader(get_split_cifar100(task, bs_intra, cifar_train, cifar_test), task, 'cpu')
            sub_loader_train , _ = fast_cifar_loader(get_subset_split_cifar100(task, bs_inter, cifar_train, 5*num_examples), task, 'cpu')
        loaders['sequential'][task]['train'], loaders['sequential'][task]['val'] = seq_loader_train, seq_loader_val
        loaders['subset'][task]['train'] = sub_loader_train
    return loaders

# if __name__ == "__main__":
#     loaders = get_all_loaders('cifar', 5, 16, 16, 25, 10)
#     print(loaders['sequential'].keys(), loaders['subset'].keys(), loaders['multitask'].keys())
#     for task in range(1, 6):
#         count = 0
#         print('testing task {}'.format(task))
#         for data, target in loaders[task]['train']:
#             count += len(data)
#         print("count >> ", count)

    # train = get_rotated_mnist_tasks(5, 100)[3]['train']
    # num_row = 5
    # num_col = 5
   
#   data, taget = next(iter(train))
#   fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
#   for i in range(25):
#       ax = axes[i//num_row, i%num_col]
#       ax.imshow(data[i].numpy()[0], cmap='gray')
#       ax.set_title('Task: {}'.format(3))
#   plt.tight_layout()
#   plt.savefig('datasets.png', dpi=300)

