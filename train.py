from comet_ml import Experiment
import os
import nni
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from models import MLP, ResNet18
from data_utils import get_all_loaders
from utils import save_model,load_model, get_norm_distance, get_cosine_similarity, plot_interpolation
from utils import  flatten_params, assign_weights, get_xy, contour_plot, get_random_string, assign_grads
from utils import plot_multi_interpolation
import uuid
from pathlib import Path

DATASET = 'cifar'
HIDDENS = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRIAL_ID =  os.environ.get('NNI_TRIAL_JOB_ID', get_random_string(5))
EXP_DIR = './checkpoints/{}'.format(TRIAL_ID)


config = {'num_tasks': 3, 'per_task_rotation': 10, 'trial': TRIAL_ID,\
          'memory_size': 500, 'num_lmc_samples': 10, 'lcm_init': 0.5,
          'lr_inter': 0.01, 'epochs_inter': 3, 'bs_inter': 32, 
          'lr_intra': 0.01, 'epochs_intra': 25,  'bs_intra': 32,
          'lr_mtl':0.05, 'epochs_mtl': 42,
         }


#config = nni.get_next_parameter()
config['trial'] = TRIAL_ID

experiment = Experiment(api_key="1UNrcJdirU9MEY0RC3UCU7eAg", \
                        project_name="explore-mc-cifar", \
                        workspace="cl-modeconnectivity", disabled=False)

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
            data = data.to(DEVICE)#.view(-1, 784)
            target = target.to(DEVICE)
            count += len(data)
            output = net(data, task_id)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= count
    correct = correct.to('cpu')
    avg_acc = 100.0 * float(correct.numpy()) / count
    return {'accuracy': avg_acc, 'loss': test_loss}


def setup_experiment():
    Path(EXP_DIR).mkdir(parents=True, exist_ok=True)
    init_model = ResNet18()#MLP(HIDDENS, 10)
    save_model(init_model, '{}/init.pth'.format(EXP_DIR))
    experiment.log_parameters(config)


def train_task_sequentially(task, config):
    prev_model_name = 'init' if task == 1 else 't_{}_seq'.format(str(task-1))
    prev_model_path = '{}/{}.pth'.format(EXP_DIR, prev_model_name)
    model = load_model(prev_model_path).to(DEVICE)
    train_loader = loaders['sequential'][task]['train']
    
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr_intra'], momentum=0.8)
    factor = 1 if task != 1 else 1
    for epoch in range(factor*config['epochs_intra']):
        model = train_single_epoch(model, optimizer, train_loader)

    save_model(model, '{}/t_{}_seq.pth'.format(EXP_DIR, task))
    if task == 1:
        save_model(model, '{}/t_{}_lcm.pth'.format(EXP_DIR, task))
        save_model(model, '{}/t_{}_mtl.pth'.format(EXP_DIR, task))
    return model

def train_task_MTL(task, config):
    assert task >= 2
    model = load_model('{}/t_{}_mtl.pth'.format(EXP_DIR, task-1)).to(DEVICE)
    train_loader = loaders['full-multitask'][task]['train']
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr_mtl'], momentum=0.8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    for epoch in range(config['epochs_mtl']):
        model = train_single_epoch(model, optimizer, train_loader)
        scheduler.step()
    save_model(model, '{}/t_{}_mtl.pth'.format(EXP_DIR, task))
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

        for name, param in m.named_parameters():
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

    for data, target, task_id in loader:
            count += len(data)
            data = data.to(DEVICE)#.view(-1, 784)
            target = target.to(DEVICE)
            output = net(data, task_id)
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
    save_model(model_lmc, '{}/t_{}_lcm.pth'.format(EXP_DIR, task))
    return model_lmc


def log_comet_metric(exp, name, val, step):
    exp.log_metric(name=name, value=val, step=step)


def check_mode_connectivity(w1, w2, eval_loader):
    net = load_model('{}/{}.pth'.format(EXP_DIR, 'init')).to(DEVICE)
    loss_history, acc_history, ts = [], [], []
    for t in np.arange(0.0, 1.01, 0.025):
        ts.append(t)
        net = assign_weights(net, w1 + t*(w2-w1)).to(DEVICE)
        metrics = eval_single_epoch(net, eval_loader)
        loss_history.append(metrics['loss'])
        acc_history.append(metrics['accuracy'])
    return loss_history, acc_history, ts


def plot_loss_plane(w, eval_loader, path, w_labels=[]):

    u = w[2] - w[0]
    dx = np.linalg.norm(u)
    u /= dx

    v = w[1] - w[0]
    v -= np.dot(u, v) * u
    dy = np.linalg.norm(v)
    v /= dy

    m = load_model('{}/{}.pth'.format(EXP_DIR, 'init')).to(DEVICE)
    m.eval()
    coords = np.stack(get_xy(p, w[0], u, v) for p in w)
    print("coords", coords)

    G = 10
    margin = 0.25
    alphas = np.linspace(0.0 - margin, 1.0 + margin, G)
    betas = np.linspace(0.0 - margin, 1.0 + margin, G)
    tr_loss = np.zeros((G, G))
    grid = np.zeros((G, G, 2))

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            p = w[0] + alpha * dx * u + beta * dy * v
            m = assign_weights(m, p).to(DEVICE)
            err = eval_single_epoch(m, eval_loader)['loss']
            c = get_xy(p, w[0], u, v)
            #print(c)
            grid[i, j] = [alpha * dx, beta * dy]
            tr_loss[i, j] = err

    contour_plot(grid, tr_loss, coords, log_alpha=-5.0, N=10, path=path, w_labels=w_labels)

def plot_mode_connections():
    # w_labels = [r"$\hat{w}_1$", r"$\hat{w}_\text{lmc}$", r"$\hat{w}_2$"]

    seq_1 = flatten_params(load_model('{}/t_{}_seq.pth'.format(EXP_DIR, 1)).to(DEVICE))
    seq_2 = flatten_params(load_model('{}/t_{}_seq.pth'.format(EXP_DIR, 2)).to(DEVICE))
    seq_3 = flatten_params(load_model('{}/t_{}_seq.pth'.format(EXP_DIR, 3)).to(DEVICE))

    mtl_2 = flatten_params(load_model('{}/t_{}_mtl.pth'.format(EXP_DIR, 2)).to(DEVICE))
    mtl_3 = flatten_params(load_model('{}/t_{}_mtl.pth'.format(EXP_DIR, 3)).to(DEVICE))

    lmc_2 = flatten_params(load_model('{}/t_{}_lcm.pth'.format(EXP_DIR, 2)).to(DEVICE))
    lmc_3 = flatten_params(load_model('{}/t_{}_lcm.pth'.format(EXP_DIR, 3)).to(DEVICE))

    #------------------------- task 1 ----------------------
    eval_loader = loaders['sequential'][1]['val']
    loss, accs, ts = check_mode_connectivity(seq_1, mtl_2, eval_loader)
    plot_interpolation(ts, accs, 'seq 1 to mtl 2', path=EXP_DIR+'/seq1_mtl2_accs.png')
    plot_interpolation(ts, loss, 'seq 1 to mtl 2', path=EXP_DIR+'/seq1_mtl2_loss.png')
    w_labels = [r"$\hat{w}_1$", r"$w^*_{2}$", r"$\hat{w}_2$"]
    plot_loss_plane([seq_1, mtl_2, seq_2], eval_loader, path=EXP_DIR+'/task1_surface_mtl2_2.png', w_labels=w_labels)

    loss, accs, ts = check_mode_connectivity(seq_1, mtl_3, eval_loader)
    plot_interpolation(ts, accs, 'seq 1 to mtl 3', path=EXP_DIR+'/seq1_mtl3_accs.png')
    plot_interpolation(ts, loss, 'seq 1 to mtl 3', path=EXP_DIR+'/seq1_mtl3_loss.png')
    w_labels = [r"$\hat{w}_1$", r"$w^*_{3}$", r"$\hat{w}_2$"]
    plot_loss_plane([seq_1, mtl_3, seq_2], eval_loader, path=EXP_DIR+'/task1_surface_mtl3_2.png', w_labels=w_labels)

    w_labels = [r"$\hat{w}_1$", r"$w^*_{3}$", r"$\hat{w}_3$"]
    plot_loss_plane([seq_1, mtl_3, seq_3], eval_loader, path=EXP_DIR+'/task1_surface_mtl3_3.png', w_labels=w_labels)

    loss_1, _, ts =  check_mode_connectivity(seq_1, mtl_2, eval_loader)
    loss_2, _, ts = check_mode_connectivity(seq_1, seq_2, eval_loader)
    plot_multi_interpolation(ts, [loss_1, loss_2], [r"$\hat{w}_1 ~ \rightarrow ~ w^*_{2} $", r"$\hat{w}_1 ~ \rightarrow ~ \hat{w}_2$"], path=EXP_DIR+'/task1_compare_paths.png')
    # loss, accs, ts = check_mode_connectivity(seq_1, lmc_2, eval_loader)
    # plot_interpolation(ts, accs, 'seq 1 to lmc 2', path=EXP_DIR+'/seq1_lmc2_accs.png')
    # plot_interpolation(ts, loss, 'seq 1 to lmc 2', path=EXP_DIR+'/seq1_lmc2_loss.png')
    # plot_loss_plane([seq_1, lmc_2, seq_2], eval_loader, path=EXP_DIR+'/task1_surface_lmc2.png', w_labels=w_labels)

    # loss, accs, ts = check_mode_connectivity(seq_1, lmc_3, eval_loader)
    # plot_interpolation(ts, accs, 'seq 1 to lmc 3', path=EXP_DIR+'/seq1_lmc3_accs.png')
    # plot_interpolation(ts, loss, 'seq 1 to lmc 3', path=EXP_DIR+'/seq1_lmc3_loss.png')
    # plot_loss_plane([seq_1, lmc_3, seq_2], eval_loader, path=EXP_DIR+'/task1_surface_lmc3.png', w_labels=w_labels)

    #------------------------- task 2 ----------------------
    eval_loader = loaders['sequential'][2]['val']
    loss, accs, ts = check_mode_connectivity(seq_2, mtl_2, eval_loader)
    plot_interpolation(ts, accs, 'seq 2 to mtl 2', path=EXP_DIR+'/seq2_mtl2_accs.png')
    plot_interpolation(ts, loss, 'seq 2 to mtl 2', path=EXP_DIR+'/seq2_mtl2_loss.png')
    w_labels = [r"$\hat{w}_1$", r"$w^*_{2}$", r"$\hat{w}_2$"]
    plot_loss_plane([seq_1, mtl_2, seq_2], eval_loader, path=EXP_DIR+'/task2_surface_mtl2_2.png', w_labels=w_labels)

    loss, accs, ts = check_mode_connectivity(seq_2, mtl_3, eval_loader)
    plot_interpolation(ts, accs, 'seq 2 to mtl 3', path=EXP_DIR+'/seq2_mtl3_accs.png')
    plot_interpolation(ts, loss, 'seq 2 to mtl 3', path=EXP_DIR+'/seq2_mtl3_loss.png')
    w_labels = [r"$\hat{w}_3$", r"$w^*_{3}$", r"$\hat{w}_2$"]
    plot_loss_plane([seq_3, mtl_3, seq_2], eval_loader, path=EXP_DIR+'/task2_surface_mtl3_2.png', w_labels=w_labels)

    w_labels = [r"$\hat{w}_1$", r"$w^*_{3}$", r"$\hat{w}_3$"]
    plot_loss_plane([seq_1, mtl_3, seq_3], eval_loader, path=EXP_DIR+'/task2_surface_mtl3_3.png', w_labels=w_labels)

    loss_1, _, ts =  check_mode_connectivity(seq_2, mtl_2, eval_loader)
    loss_2, _, ts = check_mode_connectivity(seq_2, seq_1, eval_loader)
    plot_multi_interpolation(ts, [loss_1, loss_2], [r"$\hat{w}_2 ~\rightarrow ~ w^*_{2} $", r"$\hat{w}_2 ~ \rightarrow ~ \hat{w}_1$"], path=EXP_DIR+'/task2_compare_paths.png')

    #------------------------- task 2 ----------------------
    eval_loader = loaders['sequential'][3]['val']
    loss, accs, ts = check_mode_connectivity(seq_3, mtl_3, eval_loader)
    plot_interpolation(ts, accs, 'seq 3 to mtl 3', path=EXP_DIR+'/seq3_mtl3_accs.png')
    plot_interpolation(ts, loss, 'seq 3 to mtl 3', path=EXP_DIR+'/seq3_mtl3_loss.png')

    w_labels = [r"$\hat{w}_3$", r"$w^*_{3}$", r"$\hat{w}_1$"]
    plot_loss_plane([seq_3, mtl_2, seq_1], eval_loader, path=EXP_DIR+'/task3_surface_mtl3_1.png', w_labels=w_labels)

    w_labels = [r"$\hat{w}_3$", r"$w^*_{3}$", r"$\hat{w}_2$"]
    plot_loss_plane([seq_3, mtl_2, seq_1], eval_loader, path=EXP_DIR+'/task3_surface_mtl3_2.png', w_labels=w_labels)

    loss_1, _, ts =  check_mode_connectivity(seq_3, mtl_3, eval_loader)
    loss_2, _, ts = check_mode_connectivity(seq_3, seq_1, eval_loader)
    loss_3, _, ts = check_mode_connectivity(seq_3, seq_2, eval_loader)

    plot_multi_interpolation(ts, [loss_1, loss_2, loss_3], [r"$\hat{w}_3 ~ \rightarrow ~  w^*_{3} $", r"$\hat{w}_3 ~ \rightarrow ~ \hat{w}_1$", r"$\hat{w}_3 ~ \rightarrow ~ \hat{w}_2$"], path=EXP_DIR+'/task3_compare_paths.png')



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
            if task == 1:
                log_comet_metric(experiment, 'avg_acc', metrics['accuracy'],task)
                log_comet_metric(experiment, 'avg_loss', metrics['loss'],task)

        if task > 1:
            accs, losses = [], []
            accs_mtl, losses_mtl = [], []
            
            print('---- Task {} (mtl) ----'.format(task))
            mtl_model = train_task_MTL(task, config)

            print('---- Task {} (lmc) ----'.format(task))
            lmc_model = train_task_lmc(task, config)
            for prev_task in range(1, task+1):
                metrics = eval_single_epoch(lmc_model, loaders['sequential'][prev_task]['val'])
                metrics_mtl =  eval_single_epoch(mtl_model, loaders['sequential'][prev_task]['val'])

                accs.append(metrics['accuracy'])
                accs_mtl.append(metrics_mtl['accuracy'])

                losses.append(metrics['loss'])
                losses_mtl.append(metrics_mtl['loss'])
                
                print('LMC >> ', prev_task, metrics)
                print('MTL >> ', prev_task, metrics_mtl)
                log_comet_metric(experiment, 't_{}_lmc_acc'.format(prev_task), metrics['accuracy'], task)
                log_comet_metric(experiment, 't_{}_lmc_loss'.format(prev_task), round(metrics['loss'], 5), task)
                log_comet_metric(experiment, 't_{}_mtl_acc'.format(prev_task), metrics_mtl['accuracy'], task)
                log_comet_metric(experiment, 't_{}_mtl_loss'.format(prev_task), round(metrics_mtl['loss'], 5), task)
            log_comet_metric(experiment, 'avg_acc_lmc', np.mean(accs),task)
            log_comet_metric(experiment, 'avg_loss_lmc', np.mean(losses),task)
            log_comet_metric(experiment, 'avg_acc_mtl', np.mean(accs_mtl),task)
            log_comet_metric(experiment, 'avg_loss_mtl', np.mean(losses_mtl),task)
    
        print()

    plot_mode_connections()
    #plot_loss_landscapes()

    experiment.log_asset_folder(EXP_DIR)
    experiment.end()

loaders = get_all_loaders('cifar', config['num_tasks'], config['bs_inter'], config['bs_intra'], config['memory_size'], config.get('per_task_rotation'))

if __name__ == "__main__":
    main()
