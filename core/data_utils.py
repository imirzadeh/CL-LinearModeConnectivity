import numpy as np
import random
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader, Dataset, ConcatDataset
from torch.utils.data.sampler import Sampler, RandomSampler
import torchvision.transforms.functional as TorchVisionFunc
from torchvision.datasets import MNIST, FashionMNIST
from torch.utils.data.dataloader import default_collate

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# PER_TASK_ROATATION = 9

# global permutation map for this run
permute_map = {k:np.random.RandomState().permutation(784) for k in range(2, 51)}
permute_map[1] = np.array(range(784))


FASHION_MNIST_NOISE_STD = 0.00
FASHION_MNIST_CORRUPT_PROB = 0.30

class FashionMNISTCorrupt(FashionMNIST):
  """Fashion dataset, with support for randomly corrupt labels.
  Code borrowed from: https://github.com/pluskid/fitting-random-labels/blob/master/cifar10_data.py
  Params
  ------
  corrupt_prob: float
    Default 0.0. The probability of a label being replaced with
    random label.
  num_classes: int
    Default 10. The number of classes in the dataset.
  """
  def __init__(self, corrupt_prob=FASHION_MNIST_CORRUPT_PROB, num_classes=10, **kwargs):
    super(FashionMNISTCorrupt, self).__init__(**kwargs)
    self.n_classes = num_classes
    if corrupt_prob > 0:
      self.corrupt_labels(corrupt_prob)

  def corrupt_labels(self, corrupt_prob):
    labels = np.array(self.targets)
    np.random.seed(123456)
    mask = np.random.rand(len(labels)) <= corrupt_prob
    rnd_labels = np.random.choice(self.n_classes, mask.sum())
    labels[mask] = rnd_labels
    # we need to explicitly cast the labels from npy.int64 to
    # builtin int type, otherwise pytorch will fail...
    labels = [int(x) for x in labels]

    self.targets = labels



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


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def get_mnist(fashion, batch_size, noise_std=0.0):
    dataset_class = FashionMNISTCorrupt if fashion is True else torchvision.datasets.MNIST

    if fashion is True and noise_std > 0.0:
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.5,), (0.5,)),
                                                    AddGaussianNoise(noise_std, 1.0)])
    else:
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.5,), (0.5,))])

    # transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
    mnist_train = dataset_class(root='./data/', train=True, download=True, transform=transform)
    mnist_test  = dataset_class(root='./data/', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(mnist_test,  batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader


def get_subset_mnist(fashion, batch_size, num_examples):
    trains = []
    tests = []
    
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),  torchvision.transforms.Normalize((0.5,), (0.5,))])

    dataset_class = FashionMNISTCorrupt if fashion is True else torchvision.datasets.MNIST

    mnist_train = dataset_class(root='./data/', train=True, download=True, transform=transform)
    mnist_test  = dataset_class(root='./data/', train=False, download=True, transform=transform)
    idx = mnist_train.targets < 10
    mnist_train.targets = mnist_train.targets[idx]
    mnist_train.data = mnist_train.data[idx.numpy().astype(np.bool)]
    # num_examples = num_examples_per_task * num_tasks
    sampler = torch.utils.data.RandomSampler(mnist_train, replacement=True, num_samples=num_examples)

    train_loader = torch.utils.data.DataLoader(mnist_train,  batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(mnist_test,  batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

    return train_loader, test_loader

def get_multitask_generic_mnist(batch_size):
    # num_examples_per_task = num_examples//num_tasks

    trains = []
    tests = []
    all_mtl_data = {}

    for task in range(1, 3):
        all_mtl_data[task] = {}
        # fashion = True if task == 2 else False
        # train_loader, test_loader = fast_mnist_loader(get_mnist(fashion, batch_size))
        if task == 1:
            # Task 1 => Incomplete MNIST
            train_loader, test_loader = fast_mnist_loader(get_subset_mnist(False, batch_size, 10000))
        else:
            # task 2 => Fashion MNIST
            train_loader, test_loader = fast_mnist_loader(get_mnist(True, batch_size, FASHION_MNIST_NOISE_STD))

        trains += train_loader
        tests += test_loader
        all_mtl_data[task]['train'] = random.sample(trains[:], len(trains))
        all_mtl_data[task]['val'] = tests[:]
    return all_mtl_data


def get_permuted_mnist(task_id, batch_size):
    idx_permute = permute_map[task_id]
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(lambda x: x.view(-1)[idx_permute] ),
                ])
    mnist_train = torchvision.datasets.MNIST('./data/', train=True, download=True, transform=transforms)
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=False, download=True, transform=transforms),  batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader

def get_subset_permuted_mnist(task_id, batch_size, num_examples):
    trains = []
    tests = []
    for i in [task_id]:
        idx_permute = permute_map[task_id]
        transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(lambda x: x.view(-1)[idx_permute] ),
                ])

        train = MNIST('./data/', train=True, download=True, transform=transforms)
        test = MNIST('./data/', train=False, download=True, transform=transforms)

        trains.append(train)
        tests.append(test)

    train_datasets = ConcatDataset(trains)
    test_datasets = ConcatDataset(tests)

    # num_examples = num_examples_per_task * num_tasks
    sampler = torch.utils.data.RandomSampler(train_datasets, replacement=True, num_samples=num_examples)

    train_loader = torch.utils.data.DataLoader(train_datasets,  batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_datasets,  batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

    return train_loader, test_loader


def get_multitask_permuted_mnist(num_tasks, batch_size, num_examples):
    num_examples_per_task = num_examples//num_tasks

    trains = []
    tests = []
    all_mtl_data = {}

    for task in range(1, num_tasks+1):
        all_mtl_data[task] = {}
        train_loader, test_loader = fast_mnist_loader(get_subset_permuted_mnist(task, batch_size, num_examples_per_task))
        trains += train_loader
        tests += test_loader
        all_mtl_data[task]['train'] = random.sample(trains[:], len(trains))
        all_mtl_data[task]['val'] = tests[:]
    return all_mtl_data


class RotationTransform:
    """
    Rotation transforms for the images in `Rotation MNIST` dataset.
    """
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return TorchVisionFunc.rotate(x, self.angle, fill=(0,))


def get_rotated_mnist(task_id, batch_size, per_task_rotation):
    rotation_degree = (task_id - 1)*per_task_rotation

    transforms = torchvision.transforms.Compose([
        RotationTransform(rotation_degree),
        torchvision.transforms.ToTensor(),
        ])

    train_loader = torch.utils.data.DataLoader(MNIST('./data/', train=True, download=True, transform=transforms), batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(MNIST('./data/', train=False, download=True, transform=transforms),  batch_size=1024, shuffle=False, num_workers=8, pin_memory=True)

    return train_loader, test_loader


def get_subset_rotated_mnist(task_id, batch_size, num_examples, per_task_rotation):
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
    test_loader = torch.utils.data.DataLoader(test_datasets,  batch_size=1024, shuffle=True, num_workers=4, pin_memory=True)

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
        all_mtl_data[task]['train'] = random.sample(trains[:], len(trains))#trains[:]
        all_mtl_data[task]['val'] = tests[:]
    return all_mtl_data


def get_val_loaders_mnist(num_tasks):
    loaders = {}
    for task in range(1, num_tasks+1):
        rotation_degree = (task - 1)*9.0

        transforms = torchvision.transforms.Compose([ RotationTransform(rotation_degree), torchvision.transforms.ToTensor(),])
        test_dataset = MNIST('./data/', train=False, download=True, transform=transforms)
        test_loader = torch.utils.data.DataLoader(test_dataset,  batch_size=256, shuffle=True, num_workers=4)
        loaders[task], _ = fast_mnist_loader([test_loader, []])
    return loaders

def get_val_loaders_cifar(num_tasks):
    loaders = {}
    cifar_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
    cifar_test = torchvision.datasets.CIFAR100('./data/', train=False, download=True, transform=cifar_transforms)
    
    for task_id in range(1, num_tasks+1):
        start_class = (task_id-1)*5
        end_class = task_id * 5
        
        targets_test = torch.tensor(cifar_test.targets)
        target_test_idx = ((targets_test >= start_class) & (targets_test < end_class))

        test_loader = torch.utils.data.DataLoader(torch.utils.data.dataset.Subset(cifar_test, np.where(target_test_idx==1)[0]), batch_size=128, shuffle=True)
        loaders[task_id], _ = fast_cifar_loader([test_loader, []], task_id)
    return loaders


def get_all_loaders(dataset, num_tasks, bs_inter, bs_intra, num_examples, per_task_rotation=9):
    dataset = dataset.lower()
    loaders = {'sequential': {},  'multitask':  {}, 'subset': {}, 'full-multitask': {}}

    print('loading multitask {}'.format(dataset))

    if 'cifar' in dataset:
        loaders['multitask'] = get_multitask_cifar100_loaders(num_tasks, bs_inter, num_examples)
        # loaders['full-multitask'] = get_multitask_cifar100_loaders(num_tasks, bs_inter, num_tasks*5*500)
    elif 'fashion' in dataset:
        loaders['multitask'] = []
        # loaders['full-multitask'] = get_multitask_generic_mnist(bs_inter)

    elif 'rot' in dataset and 'mnist' in dataset:
        loaders['multitask'] = get_multitask_rotated_mnist(num_tasks, bs_inter, num_examples, per_task_rotation)
        # loaders['full-multitask'] = get_multitask_rotated_mnist(num_tasks, bs_inter, num_tasks*10*2000, per_task_rotation)
    elif 'perm' in dataset and 'mnist' in dataset:
        loaders['multitask'] = get_multitask_permuted_mnist(num_tasks, bs_inter, num_examples)
        # loaders['full-multitask'] = get_multitask_permuted_mnist(num_tasks, bs_inter, num_tasks*10*2000)
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
        if 'fashion' in dataset and 'mnist' in dataset:
            assert task <= 2
            fashion = True if task == 2 else False
            seq_loader_train , seq_loader_val = fast_mnist_loader(get_mnist(fashion, bs_intra, FASHION_MNIST_NOISE_STD), 'cpu')
            sub_loader_train , _ = [], []
        elif 'rot' in dataset and 'mnist' in dataset:
            seq_loader_train , seq_loader_val = fast_mnist_loader(get_rotated_mnist(task, bs_intra, per_task_rotation), 'cpu')
            sub_loader_train , _ = fast_mnist_loader(get_subset_rotated_mnist(task, bs_inter, 2*num_examples, per_task_rotation),'cpu')
        elif 'perm' in dataset and 'mnist' in dataset:
            seq_loader_train , seq_loader_val = fast_mnist_loader(get_permuted_mnist(task, bs_intra), 'cpu')
            sub_loader_train , _ =  (get_subset_permuted_mnist(task, bs_inter, 10*num_examples),'cpu')
        elif 'cifar' in dataset:
            seq_loader_train , seq_loader_val = fast_cifar_loader(get_split_cifar100(task, bs_intra, cifar_train, cifar_test), task, 'cpu')
            sub_loader_train , _ = fast_cifar_loader(get_subset_split_cifar100(task, bs_inter, cifar_train, 5*num_examples), task, 'cpu')
        
        loaders['sequential'][task]['train'], loaders['sequential'][task]['val'] = seq_loader_train, seq_loader_val
        loaders['subset'][task]['train'] = sub_loader_train

    return loaders



# if __name__ == "__main__":
    # train = get_multitask_generic_mnist(32)[2]['train']
    # print(len(train), len(train[0][0]), len(train[0][1]))
    # loaders = get_mnist(fashion=True, batch_size=32)
    # loaders = get_all_loaders('mnist-fashion', 2, 32, 32, 100)
#     loaders = get_val_loaders_mnist(5)
#     print(loaders.keys())
#     print(len(loaders[1]))
#     loaders = get_all_loaders('cifar', 5, 16, 16, 25, 10)
#     print(loaders['sequential'].keys(), loaders['subset'].keys(), loaders['multitask'].keys())
#     for task in range(1, 6):
#         count = 0
#         print('testing task {}'.format(task))
#         for data, target in loaders[task]['train']:
#             count += len(data)
#         print("count >> ", count)

    # train = get_rotated_mnist_tasks(5, 100)[3]['train']
    # num_row = 10
    # num_col = 2
   
    # # data, taget = next(iter(train))
    # data = train
    # fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
    # for i in range(25):
    #     ax = axes[i//num_row, i%num_col]
    #     ax.imshow(data[i].numpy().reshape(28, 28), cmap='gray')
    #     # ax.set_title('Task: {}'.format(3))
    # plt.tight_layout()
    # plt.savefig('datasets.png', dpi=300)

