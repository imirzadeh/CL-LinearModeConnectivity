from comet_ml import Experiment
import os
import torch
import numpy as np
import pandas as pd
from core.train_methods import train_task_sequentially, train_task_MTL, eval_single_epoch
from core.utils import save_np_arrays, setup_experiment, log_comet_metric, get_random_string
from core.utils import save_task_model_by_policy, load_task_model_by_policy, flatten_params
from core.utils import assign_weights
from core.data_utils import get_all_loaders
from core.mode_connectivity import calculate_mode_connectivity
from core.visualization import plot_contour, get_xy
from core.visualization import plot_single_interpolation, plot_multi_interpolations

DATASET = 'rot-mnist'
HIDDENS = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRIAL_ID =  os.environ.get('NNI_TRIAL_JOB_ID', get_random_string(5))
EXP_DIR = './checkpoints/{}'.format(TRIAL_ID)


config = {'num_tasks': 2, 'per_task_rotation': 45, 'trial': TRIAL_ID,\
          'memory_size': 2000, 'num_lmc_samples': 10, 'lcm_init': 0.5,
          'lr_inter': 0.05, 'epochs_inter': 5, 'bs_inter': 64, 
          'lr_intra': 0.05, 'epochs_intra': 5,  'bs_intra': 64,
          'lr_mtl':0.05, 'epochs_mtl': 5, 'exp_dir': EXP_DIR,
          'mtl_start_from_init': False,
          'dataset': DATASET, 'mlp_hiddens': HIDDENS, 'device': DEVICE,
         }

#config = nni.get_next_parameter()
config['trial'] = TRIAL_ID
experiment = Experiment(api_key="1UNrcJdirU9MEY0RC3UCU7eAg", \
                        project_name="explore-refactor-code", \
                        workspace="cl-modeconnectivity", disabled=True)

loaders = get_all_loaders(config['dataset'], config['num_tasks'],\
                         config['bs_inter'], config['bs_intra'],\
                         config['memory_size'], config.get('per_task_rotation'))


def plot_loss_plane(w, eval_loader, path, w_labels, config):
    u = w[2] - w[0]
    dx = np.linalg.norm(u)
    u /= dx

    v = w[1] - w[0]
    v -= np.dot(u, v) * u
    dy = np.linalg.norm(v)
    v /= dy

    m = load_task_model_by_policy(0, 'init', config['exp_dir'])
    m.eval()
    coords = np.stack(get_xy(p, w[0], u, v) for p in w)
    # print("coords", coords)

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

    contour = {'grid': grid, 'values': tr_loss, 'coords': coords}
    save_np_arrays(contour, path=path)
    plot_contour(grid, tr_loss, coords, log_alpha=-5.0, N=7, path=path, w_labels=w_labels)
    return contour

def plot_mode_connections(t1, p1, t2, p2, eval_task, eval_loader, config):
    w1 = flatten_params(load_task_model_by_policy(t1, p1, config['exp_dir']))
    w2 = flatten_params(load_task_model_by_policy(t2, p2, config['exp_dir']))
    loss, acc, ts = calculate_mode_connectivity(w1, w2, eval_loader, config)
    save_path = '{}/mc_{}_{}_to_{}_{}_on_{}'.format(config['exp_dir'],p1, t1, p2, t2, eval_task)
    res = {'loss': loss, 'acc': acc, 'ts': ts}
    print("DEBUG >>", loss[::5])
    save_np_arrays(res, path=save_path)
    # plot_single_interpolation(x=ts, y=loss, path=save_path)
    return res

def plot_graphs(num_tasks, config):

    # load models
    models = {'seq': {}, 'mtl': {}}
    for t in range(1, num_tasks+1):
        models['seq'][t] = flatten_params(load_task_model_by_policy(t, 'seq', config['exp_dir']))
        if t >= 2:
            models['mtl'][t] = flatten_params(load_task_model_by_policy(t, 'mtl', config['exp_dir']))

    # plot mode connectivity
    r1 = plot_mode_connections(1, 'seq', 2, 'seq', 1, loaders['sequential'][1]['val'], config)
    r2 = plot_mode_connections(1, 'seq', 2, 'mtl', 1, loaders['sequential'][1]['val'], config)
    ylabels = [r"$\hat{w}_1 ~ \rightarrow ~ w^*_{2} $", r"$\hat{w}_1 ~ \rightarrow ~ \hat{w}_2$"]
    plot_multi_interpolations(x=r1['ts'], ys=[r1['loss'], r2['loss']], y_labels=ylabels, path='{}/mc_on_{}'.format(config['exp_dir'], 1))

    path = '{}/surface_{}_{}_{}_{}_{}_{}_on_{}'.format(config['exp_dir'], 'seq', 1, 'mtl', 2, 'seq', 2,  1)
    labels = [r"$\hat{w}_1$", r"$w^*_{2}$", r"$\hat{w}_2$"]
    plot_loss_plane([models['seq'][1], models['mtl'][2], models['seq'][2]], loaders['sequential'][1]['val'], path, labels, config)

    path = '{}/surface_{}_{}_{}_{}_{}_{}_on_{}'.format(config['exp_dir'], 'seq', 1, 'mtl', 2, 'seq', 2,  2)
    labels = [r"$\hat{w}_1$", r"$w^*_{2}$", r"$\hat{w}_2$"]
    plot_loss_plane([models['seq'][1], models['mtl'][2], models['seq'][2]], loaders['sequential'][2]['val'], path, labels, config)


def main():
    print('Started the trial >>', TRIAL_ID, 'for experiment 1')
    # init and save
    setup_experiment(experiment, config)  

    # convention:  init      =>  initialization
    # convention:  t_i_seq   =>  task i (sequential)
    # convention:  t_i_mtl   => task 1 ... i (multitask)
    # convention:  t_i_lcm   => task 1 ... i (Linear Mode Connectivity)
    for task in range(1, config['num_tasks']+1):
        print('---- Task {} (seq) ----'.format(task))
        seq_model = train_task_sequentially(task, loaders['sequential'][task]['train'], config)
        save_task_model_by_policy(seq_model, task, 'seq', config['exp_dir'])
        for prev_task in range(1, task+1):
            metrics = eval_single_epoch(seq_model, loaders['sequential'][prev_task]['val'])
            print(prev_task, metrics)
            log_comet_metric(experiment, 't_{}_seq_acc'.format(prev_task), metrics['accuracy'], task)
            log_comet_metric(experiment, 't_{}_seq_loss'.format(prev_task), round(metrics['loss'], 5), task)
            if task == 1:
                log_comet_metric(experiment, 'avg_acc', metrics['accuracy'],task)
                log_comet_metric(experiment, 'avg_loss', metrics['loss'],task)

        if task > 1:
            accs_mtl, losses_mtl = [], []
            
            print('---- Task {} (mtl) ----'.format(task))
            mtl_model = train_task_MTL(task, loaders['full-multitask'][task]['train'], config, loaders['sequential'][1]['val'])

            save_task_model_by_policy(mtl_model, task, 'mtl', config['exp_dir'])
            
            for prev_task in range(1, task+1):
                metrics_mtl =  eval_single_epoch(mtl_model, loaders['sequential'][prev_task]['val'])
                accs_mtl.append(metrics_mtl['accuracy'])
                losses_mtl.append(metrics_mtl['loss'])
                
                print('MTL >> ', prev_task, metrics_mtl)
                log_comet_metric(experiment, 't_{}_mtl_acc'.format(prev_task), metrics_mtl['accuracy'], task)
                log_comet_metric(experiment, 't_{}_mtl_loss'.format(prev_task), round(metrics_mtl['loss'], 5), task)
            log_comet_metric(experiment, 'avg_acc_mtl', np.mean(accs_mtl),task)
            log_comet_metric(experiment, 'avg_loss_mtl', np.mean(losses_mtl),task)
        print()

    plot_graphs(2, config)
    #plot_loss_landscapes()

    experiment.log_asset_folder(EXP_DIR)
    experiment.end()

if __name__ == "__main__":
    main()
