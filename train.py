from comet_ml import Experiment
import os
import nni
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from models import MLP,GatedMLP,Resnet20
from data_utils import get_multitask_rotated_mnist, get_rotated_mnist, get_subset_rotated_mnist, fast_mnist_loader
from utils import save_model, load_model, get_norm_distance, get_cosine_similarity
from utils import  plot, flatten_params, assign_weights, get_xy, contour_plot, get_random_string, assign_grads
import uuid
from pathlib import Path

HIDDENS = 200
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRIAL_ID =  os.environ.get('NNI_TRIAL_JOB_ID', get_random_string(5))
EXP_DIR = './checkpoints/{}'.format(TRIAL_ID)

# config = {'num_tasks': 5, 'per_task_rotation': 10, 'trial': TRIAL_ID,\
#         'memory_size': 200,  'num_lmc_samples': 20, 'lcm_init': 0.4,
#         'lr_inter': 1.25, 'epochs_inter': 3, 'bs_inter': 32, \
#         'lr_intra': 0.1, 'epochs_intra': 3,  'bs_intra': 64,
#        }

config = {'num_tasks': 20, 'per_task_rotation': 9, 'trial': TRIAL_ID,\
          'memory_size': 200, 'num_lmc_samples': 25, 'lcm_init': 0.05,
          'lr_inter': 0.1, 'epochs_inter': 15, 'bs_inter': 32, \
          'lr_intra': 0.1, 'epochs_intra': 5,  'bs_intra': 32,
         }

# config = nni.get_next_parameter()
config['trial'] = TRIAL_ID

experiment = Experiment(api_key="1UNrcJdirU9MEY0RC3UCU7eAg", \
                        project_name="explore-rotmnist-20-tasks", \
                        workspace="cl-modeconnectivity", disabled=True)

def train_single_epoch(net, optimizer, loader):
    net = net.to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    net.train()
    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(DEVICE).view(-1, 784)
        target = target.to(DEVICE)
        optimizer.zero_grad()
        pred = net(data)

        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
    return net


def eval_single_epoch(net, loader):
    net = net.to(DEVICE)
    net.eval()
    test_loss = 0
    correct = 0
    count = 0 # because of sampler
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    with torch.no_grad():
        for data, target in loader:
            data = data.to(DEVICE).view(-1, 784)
            target = target.to(DEVICE)
            count += len(data)
            output = net(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= count
    correct = correct.to('cpu')
    avg_acc = 100.0 * float(correct.numpy()) / count
    return {'accuracy': avg_acc, 'loss': test_loss}


def setup_experiment():
    Path(EXP_DIR).mkdir(parents=True, exist_ok=True)
    init_model = MLP(HIDDENS, 10)
    save_model(init_model, '{}/init.pth'.format(EXP_DIR))
    experiment.log_parameters(config)


def get_all_loaders():
    """
    Create all required dataset loaders for the CL experience
    """
    loaders = {'sequential': {}, 'multitask': get_multitask_rotated_mnist(config['num_tasks'], config['bs_intra'], int(config['memory_size']/config['num_tasks']),  config['per_task_rotation']), 'subset': {}}
    for task in range(1, config['num_tasks']+1):
        print("loading data for task {}".format(task))
        loaders['sequential'][task], loaders['subset'][task] = {}, {}

        seq_loader_train , seq_loader_val = fast_mnist_loader(get_rotated_mnist(task, config['bs_intra'], config['per_task_rotation']), 'cpu')
        # mtl_loader_train , mtl_loader_val = fast_mnist_loader(get_multitask_rotated_mnist(task, config['bs_intra'], int(config['memory_size']/task), config['per_task_rotation']), 'cpu')
        sub_loader_train , _ = fast_mnist_loader(get_subset_rotated_mnist(task, config['bs_inter'], 5*int(config['memory_size']), config['per_task_rotation']),'cpu')
        
        # loaders['multitask'][task]['train'], loaders['multitask'][task]['val'] = mtl_loader_train, mtl_loader_val
        loaders['sequential'][task]['train'], loaders['sequential'][task]['val'] = seq_loader_train, seq_loader_val
        loaders['subset'][task]['train'] = sub_loader_train
    return loaders

def train_task_sequentially(task, config):
    prev_model_name = 'init' if task == 1 else 't_{}_seq'.format(str(task-1))
    prev_model_path = '{}/{}.pth'.format(EXP_DIR, prev_model_name)
    model = load_model(prev_model_path).to(DEVICE)
    train_loader = loaders['sequential'][task]['train']
    
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr_intra'], momentum=0.8)

    for epoch in range(config['epochs_intra']):
        model = train_single_epoch(model, optimizer, train_loader)

    save_model(model, '{}/t_{}_seq.pth'.format(EXP_DIR, task))
    if task == 1:
        save_model(model, '{}/t_{}_lcm.pth'.format(EXP_DIR, task))
    return model

def get_line_loss(start_w, w, loader):

    m = load_model('{}/{}.pth'.format(EXP_DIR, 'init')).to(DEVICE)
    total_loss = 0
    accum_grad = None
    for t in np.arange(0.0, 1.01, 1.0/float(config['num_lmc_samples'])):
        grads = []
        cur_weight = start_w + (w - start_w) * t
        m = assign_weights(m, cur_weight).to(DEVICE)
        current_loss = get_clf_loss(m, loader)
        current_loss.backward()
        for param in m.parameters():
            grads.append(param.grad.view(-1))
        grads = torch.cat(grads)
        if accum_grad is None:
            accum_grad = grads
        else:
            accum_grad += grads
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
    test_loss /= count
    return test_loss


def train_task_lmc(task, config):
    assert task > 1
    model_prev = load_model('{}/t_{}_lcm.pth'.format(EXP_DIR, task-1)).to(DEVICE)
    model_curr = load_model('{}/t_{}_seq.pth'.format(EXP_DIR, task)).to(DEVICE)

    w_prev = flatten_params(model_prev, numpy_output=True)
    w_curr = flatten_params(model_curr, numpy_output=True)

    model_lmc = load_model('{}/{}.pth'.format(EXP_DIR, 'init')).to(DEVICE)
    model_lmc = assign_weights(model_lmc, w_prev + config['lcm_init']*(w_curr-w_prev)).to(DEVICE)
    optimizer = torch.optim.SGD(model_lmc.parameters(), lr=config['lr_inter'], momentum=0.8)


    loader_prev = loaders['multitask'][task]['train']
    loader_curr = loaders['subset'][task]['train']
    factor = 2 if task == config['num_tasks'] else 1
    for epoch in range(factor*config['epochs_inter']): 
        model_lmc.train()
        optimizer.zero_grad()

        grads = get_line_loss(w_prev, flatten_params(model_lmc, numpy_output=True), loader_prev) \
              + get_line_loss(w_curr, flatten_params(model_lmc, numpy_output=True), loader_curr)
        model_lmc = assign_grads(model_lmc, grads).to(DEVICE) # NOTE: it has loss.backward() within of itself
        optimizer.step()
        for prev_task in range(1, task+1):
            print("LMC Debug >> epoch {} >> metric {} >> {}".format(epoch+1, prev_task, eval_single_epoch(model_lmc, loaders['sequential'][prev_task]['val'])))
        print()
    save_model(model_lmc, '{}/t_{}_lcm.pth'.format(EXP_DIR, task))
    return model_lmc


def log_comet_metric(exp, name, val, step):
    exp.log_metric(name=name, value=val, step=step)



def main():
    print(TRIAL_ID)
    # init and save
    setup_experiment()  

    # convention:  init      =>  initialization
    # convention:  t_i_seq   =>  task i (sequential)
    # convention:  t_i_mtl   => task 1 ... i (multitask)
    # convention:  t_i_lcm   => task 1 ... i (Linear Mode Connectivity)
    for task in range(1, config['num_tasks']+1):
        print('---- Task {} (seq) ----'.format(task))
        seq_model = train_task_sequentially(task, config)
        for prev_task in range(1, task+1):
            metrics = eval_single_epoch(seq_model, loaders['sequential'][prev_task]['val'])
            print(prev_task, metrics)
            log_comet_metric(experiment, 't_{}_seq_acc'.format(prev_task), metrics['accuracy'], task)
            log_comet_metric(experiment, 't_{}_seq_loss'.format(prev_task), round(metrics['loss'], 5), task)
        
        if task > 1:
            print('---- Task {} (lcm) ----'.format(task))
            lmc_model = train_task_lmc(task, config)
            for prev_task in range(1, task+1):
                metrics = eval_single_epoch(lmc_model, loaders['sequential'][prev_task]['val'])
                print(prev_task, metrics)
                log_comet_metric(experiment, 't_{}_lmc_acc'.format(prev_task), metrics['accuracy'], task)
                log_comet_metric(experiment, 't_{}_lmc_loss'.format(prev_task), round(metrics['loss'], 5), task)
        print()
    experiment.log_asset_folder(EXP_DIR)
    experiment.end()

loaders = get_all_loaders()

if __name__ == "__main__":
    main()
