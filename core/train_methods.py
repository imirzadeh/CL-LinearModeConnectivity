import os
import torch
import numpy as np
import torch.nn as nn
from .utils import DEVICE, save_model,load_model
from .utils import flatten_params, assign_weights, assign_grads
from .mode_connectivity import get_line_loss, bezier_path_opt
# from core.mode_connectivity import 


def train_single_epoch(net, optimizer, loader):
    net = net.to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    net.train()
    for batch_idx, (data, target, task_id) in enumerate(loader):
        data = data.to(DEVICE)#.view(-1, 784)
        target = target.to(DEVICE)
        optimizer.zero_grad()
        pred = net(data, task_id)

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
        for data, target, task_id in loader:
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            count += len(target)
            output = net(data, task_id)
            test_loss += criterion(output, target).item()*len(target)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= count
    correct = correct.to('cpu')
    avg_acc = 100.0 * float(correct.numpy()) / count
    return {'accuracy': avg_acc, 'loss': test_loss}

def train_task_sequentially(task, train_loader, config):
    EXP_DIR = config['exp_dir']
    current_lr = max(0.001, config['seq_lr'] * (config['lr_decay'])**(task-1))
    prev_model_name = 'init' if task == 1 else 't_{}_seq'.format(str(task-1))
    prev_model_path = '{}/{}.pth'.format(EXP_DIR, prev_model_name)
    model = load_model(prev_model_path).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=current_lr, momentum=config['momentum'])

    # train_loader = loaders['sequential'][task]['train']
    
    #optimizer = torch.optim.SGD(model.parameters(), lr=config['seq_lr'], momentum=0.8)
    for epoch in range(config['seq_epochs']):
        model = train_single_epoch(model, optimizer, train_loader)
    return model


def train_task_LMC_offline(task, loaders, config):
    assert task >= 2
    EXP_DIR = config['exp_dir']
    model_prev = load_model('{}/t_{}_lmc.pth'.format(EXP_DIR, task-1)).to(DEVICE)
    model_curr = load_model('{}/t_{}_seq.pth'.format(EXP_DIR, task)).to(DEVICE)

    model_lmc = load_model('{}/{}.pth'.format(EXP_DIR, 'init')).to(DEVICE)

    w_prev = flatten_params(model_prev, numpy_output=True)
    w_curr = flatten_params(model_curr, numpy_output=True)

    model_lmc = assign_weights(model_lmc, w_prev + config['lcm_init_position']*(w_curr-w_prev)).to(DEVICE)

    loader_prev = loaders['multitask'][task]['train']
    loader_curr = loaders['subset'][task]['train']

    optimizer = torch.optim.SGD(model_lmc.parameters(), lr=config['lmc_lr'], momentum=config['momentum'])
    factor = 1 #if task != config['num_tasks'] else 2
    for epoch in range(factor*config['lmc_epochs']):
        model_lmc.train()
        optimizer.zero_grad()
        # grads = get_line_loss(w_prev, flatten_params(model_lmc), loader_prev, config) \
        #       + get_line_loss(w_curr, flatten_params(model_lmc), loader_curr, config)
        grads = bezier_path_opt(w_prev, w_curr, flatten_params(model_lmc), loader_prev + loader_curr, config)
        model_lmc = assign_grads(model_lmc, grads).to(DEVICE) # NOTE: it has loss.backward() within of itself
        optimizer.step()
        for prev_task in range(1, task+1):
            print("LMC Debug >> epoch {} >> metric {} >> {}".format(epoch+1, prev_task, eval_single_epoch(model_lmc, loaders['sequential'][prev_task]['val'])))
        print()
    return model_lmc

def train_task_MTL(task, train_loader, config, val_loader):
    assert task >= 2
    EXP_DIR = config['exp_dir']
    if task == 2 and config['mtl_start_from_other_init'] == True:
        model = load_model('{}/init_2.pth'.format(EXP_DIR)).to(DEVICE)
        print("WARNING >> MTL is loading not a shared init checkpoint!")
    else:
        model = load_model('{}/t_{}_mtl.pth'.format(EXP_DIR, task-1)).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr_mtl'], momentum=0.8)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    for epoch in range(config['epochs_mtl']):
        model = train_single_epoch(model, optimizer, train_loader)
        print("DEBUG >> ", eval_single_epoch(model, val_loader))
        # scheduler.step()
    return model

'''
def train_task_lmc(task, config):
    assert task > 1
    model_prev = load_model('{}/t_{}_lmc.pth'.format(EXP_DIR, task-1)).to(DEVICE)
    model_curr = load_model('{}/t_{}_seq.pth'.format(EXP_DIR, task)).to(DEVICE)

    w_prev = flatten_params(model_prev, numpy_output=True)
    w_curr = flatten_params(model_curr, numpy_output=True)

    model_lmc = load_model('{}/{}.pth'.format(EXP_DIR, 'init')).to(DEVICE)
    model_lmc = assign_weights(model_lmc, w_prev + config['lcm_init']*(w_curr-w_prev)).to(DEVICE)
    # optimizer = torch.optim.SGD(model_lmc.parameters(), lr=config['lr_inter'], momentum=0.8)


    loader_prev = loaders['multitask'][task]['train']
    loader_curr = loaders['subset'][task]['train']
    optimizer = torch.optim.SGD(model_lmc.parameters(), lr=config['lr_inter'], momentum=0.8)
    factor = 1 if task == config['num_tasks'] else 1
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
    save_model(model_lmc, '{}/t_{}_lmc.pth'.format(EXP_DIR, task))
    return model_lmc
'''
