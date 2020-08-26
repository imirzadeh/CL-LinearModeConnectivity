from comet_ml import Experiment
import os
import torch
import numpy as np
import pandas as pd
from core.data_utils import get_all_loaders
from core.cka_utils import calculate_CKA
from core.train_methods import train_task_sequentially, train_task_LMC_offline, eval_single_epoch
from core.utils import save_np_arrays, setup_experiment, log_comet_metric, get_random_string
from core.utils import save_task_model_by_policy, load_task_model_by_policy, flatten_params
from core.utils import assign_weights, get_norm_distance, ContinualMeter
from core.visualization import plot_contour, get_xy, plot_heat_map, plot_l2_map, plot_accs
from core.visualization import plot_single_interpolation, plot_multi_interpolations

DATASET = 'rot-mnist'
HIDDENS = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRIAL_ID =  os.environ.get('NNI_TRIAL_JOB_ID', get_random_string(5))
EXP_DIR = './checkpoints/{}'.format(TRIAL_ID)


# config = {
#          # ---COMMON----
#          'num_tasks': 20, 'per_task_rotation': 9, 'trial': TRIAL_ID, 'exp_dir': EXP_DIR,\
#          'memory_size': 200, 'dataset': DATASET, 'device': DEVICE, 'momentum': 0.8,\
#          'mlp_hiddens': HIDDENS, 'dropout': 0.2, 'lr_decay': 0.8, 'stable_sgd': False,\

#           # ----Seq Model-----
#           'seq_lr': 0.1, 'seq_batch_size': 64, 'seq_epochs': 1,\

#           # ------LMC models------
#           'lmc_policy': 'offline', 'lmc_interpolation': 'linear',\
#           'lmc_lr': 0.01, 'lmc_batch_size': 64, 'lcm_init_position': 0.1,\
#           'lmc_line_samples': 5, 'lmc_epochs': 1,   
#          }

seq_meter = ContinualMeter('seq_accs', config['num_tasks'])
lmc_meter = ContinualMeter('lmc_accs', config['num_tasks'])

config = nni.get_next_parameter()
config['per_task_rotation'] = 9
config['mlp_hiddens'] = HIDDENS
config['trial'] = TRIAL_ID
config['dataset'] = DATASET
config['device'] = DEVICE
config['exp_dir'] = EXP_DIR
config['lmc_policy'] = 'offline'
config['lmc_interpolation'] =  linear


experiment = Experiment(api_key="1UNrcJdirU9MEY0RC3UCU7eAg", \
                        project_name="lmc-offline-rot-20", \
                        workspace="cl-modeconnectivity", disabled=False)

loaders = get_all_loaders(config['dataset'], config['num_tasks'],\
                         config['lmc_batch_size'], config['seq_batch_size'],\
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

    G = 15
    margin = 0.2
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
    plot_contour(grid, tr_loss, coords, log_alpha=-5.0, N=7, path=path, w_labels=w_labels, dataset=config['dataset'])
    return contour

def get_mode_connections(p1, t1, p2, t2, eval_task, config):
    w1 = flatten_params(load_task_model_by_policy(t1, p1, config['exp_dir']))
    w2 = flatten_params(load_task_model_by_policy(t2, p2, config['exp_dir']))
    loss, acc, ts = calculate_mode_connectivity(w1, w2, loaders['sequential'][eval_task]['val'], config)
    save_path = '{}/mc_{}_{}_to_{}_{}_on_{}'.format(config['exp_dir'],p1, t1, p2, t2, eval_task)
    res = {'loss': loss, 'acc': acc, 'ts': ts}
    save_np_arrays(res, path=save_path)
    return res

def plot_mode_connections_for_minima(p1, t1, config, max_task=None):
    seq_cons, mtl_cons = [], []
    seq_labels, mtl_labels = [], []
    segments = []
    if max_task is None:
        max_task = config['num_tasks']
    for t2 in range(t1+1, max_task+1):
        seq_con = get_mode_connections(p1, t1, 'seq', t2, t1, config)
        mtl_con = get_mode_connections(p1, t1, 'mtl', t2, t1, config)
        segments = seq_con['ts']
        seq_labels.append(r"$\hat{{w}}_{} \rightarrow \hat{{w}}_{}$".format(t1, t2))
        mtl_labels.append(r"$\hat{{w}}_{} \rightarrow w^*_{}$".format(t1, t2))
        seq_cons.append(seq_con['loss'])
        mtl_cons.append(mtl_con['loss'])
    # print("DEBUG MC >> len(labels)=", len(seq_cons+mtl_cons))
    save_path = path='{}/mc_on_{}_max_{}'.format(config['exp_dir'], t1, max_task)
    plot_multi_interpolations(x=segments, ys=seq_cons + mtl_cons ,y_labels=seq_labels+mtl_labels, path=save_path)


def plot_graphs(config):
    # load models
    models = {'seq': {}, 'mtl': {}, 'lmc': {}}
    for t in range(1, config['num_tasks']+1):
        models['seq'][t] = flatten_params(load_task_model_by_policy(t, 'seq', config['exp_dir']))
        if t >= 2:
            # models['mtl'][t] = flatten_params(load_task_model_by_policy(t, 'mtl', config['exp_dir']))
            models['lmc'][t] = flatten_params(load_task_model_by_policy(t, 'lmc', config['exp_dir']))
    acc_fig_path = "{}/accs".format(config['exp_dir'])
    plot_accs(config['num_tasks'], seq_meter.data, lmc_meter.data, acc_fig_path)

    # --- task 1 ---
    # plot_mode_connections_for_minima('seq', 1, config)
    # plot_mode_connections_for_minima('seq', 1, config, 2)
    # plot_mode_connections_for_minima('seq', 1, config, 3)
    # plot_mode_connections_for_minima('seq', 2, config)
    # plot_mode_connections_for_minima('seq', 2, config, 3)

    # path = '{}/surface_{}_{}_{}_{}_{}_{}_on_{}'.format(config['exp_dir'], 'seq', 1, 'lmc', 2, 'seq', 2,  1)
    # labels = [r"$\hat{w}_1$", r"$\bar{w}_{2}$", r"$\hat{w}_2$"]
    # plot_loss_plane([models['seq'][1], models['lmc'][2], models['seq'][2]], loaders['sequential'][1]['val'], path, labels, config)

    # path = '{}/surface_{}_{}_{}_{}_{}_{}_on_{}'.format(config['exp_dir'], 'seq', 1, 'lmc', 2, 'seq', 2,  2)
    # labels = [r"$\hat{w}_1$", r"$\bar{w}_{2}$", r"$\hat{w}_2$"]
    # plot_loss_plane([models['seq'][1], models['lmc'][2], models['seq'][2]], loaders['sequential'][2]['val'], path, labels, config)

    # path = '{}/surface_{}_{}_{}_{}_{}_{}_on_{}'.format(config['exp_dir'], 'seq', 1, 'lmc', 3, 'seq', 3,  1)
    # labels = [r"$\hat{w}_1$", r"$\bar{w}_{3}$", r"$\hat{w}_3$"]
    # plot_loss_plane([models['seq'][1], models['lmc'][3], models['seq'][3]], loaders['sequential'][1]['val'], path, labels, config)

    # path = '{}/surface_{}_{}_{}_{}_{}_{}_on_{}'.format(config['exp_dir'], 'seq', 1, 'lmc', 3, 'seq', 3,  3)
    # labels = [r"$\hat{w}_1$", r"$W\bar{w}{3}$", r"$\hat{w}_3$"]
    # plot_loss_plane([models['seq'][1], models['lmc'][3], models['seq'][3]], loaders['sequential'][3]['val'], path, labels, config)

    # path = '{}/surface_{}_{}_{}_{}_{}_{}_on_{}'.format(config['exp_dir'], 'seq', 2, 'lmc', 3, 'seq', 3,  2)
    # labels = [r"$\hat{w}_2$", r"$W\bar{w}{3}$", r"$\hat{w}_3$"]
    # plot_loss_plane([models['seq'][2], models['lmc'][3], models['seq'][3]], loaders['sequential'][2]['val'], path, labels, config)

def calculate_mode_connectivity(w1, w2, eval_loader, config):
    net = load_model('{}/{}.pth'.format(config['exp_dir'], 'init')).to(DEVICE)
    loss_history, acc_history, ts = [], [], []
    for t in np.arange(0.0, 1.01, 0.025):
        ts.append(t)
        net = assign_weights(net, w1 + t*(w2-w1)).to(DEVICE)
        metrics = eval_single_epoch(net, eval_loader)
        loss_history.append(metrics['loss'])
        acc_history.append(metrics['accuracy'])
    return loss_history, acc_history, ts


def main():
    print('Started the trial >>', TRIAL_ID, 'for experiment 1')
    # init and save
    setup_experiment(experiment, config)

    # convention:  init      =>  initialization
    # convention:  t_i_seq   =>  task i (sequential)
    # convention:  t_i_mtl   => task 1 ... i (multitask)
    # convention:  t_i_lmc   => task 1 ... i (Linear Mode Connectivity)


    for task in range(1, config['num_tasks']+1):
        print('---- Task {} (seq) ----'.format(task))
        seq_model = train_task_sequentially(task, loaders['sequential'][task]['train'], config)
        save_task_model_by_policy(seq_model, task, 'seq', config['exp_dir'])
        for prev_task in range(1, task+1):
            metrics = eval_single_epoch(seq_model, loaders['sequential'][prev_task]['val'])
            seq_meter.update(task, prev_task, metrics['accuracy'])
            print(prev_task, metrics)
            log_comet_metric(experiment, 't_{}_seq_acc'.format(prev_task), metrics['accuracy'], task)
            log_comet_metric(experiment, 't_{}_seq_loss'.format(prev_task), round(metrics['loss'], 5), task)
            if task == 1:
                log_comet_metric(experiment, 'avg_acc', metrics['accuracy'],task)
                log_comet_metric(experiment, 'avg_loss', metrics['loss'],task)

        if task > 1:
            accs_lmc, losses_lmc = [], []

            print('---- Task {} (lmc) ----'.format(task))
            lmc_model = train_task_LMC_offline(task, loaders, config)

            save_task_model_by_policy(lmc_model, task, 'lmc', config['exp_dir'])

            for prev_task in range(1, task+1):
                metrics_lmc =  eval_single_epoch(lmc_model, loaders['sequential'][prev_task]['val'])
                accs_lmc.append(metrics_lmc['accuracy'])
                losses_lmc.append(metrics_lmc['loss'])
                lmc_meter.update(task, prev_task, metrics['accuracy'])

                print('LMC >> ', prev_task, metrics_lmc)
                log_comet_metric(experiment, 't_{}_lmc_acc'.format(prev_task), metrics_lmc['accuracy'], task)
                log_comet_metric(experiment, 't_{}_lmc_loss'.format(prev_task), round(metrics_lmc['loss'], 5), task)
            log_comet_metric(experiment, 'avg_acc_lmc', np.mean(accs_lmc),task)
            log_comet_metric(experiment, 'avg_loss_lmc', np.mean(losses_lmc),task)
        print()

    seq_meter.save(config)
    lmc_meter.save(config)

    # plot_graphs(config)

    experiment.log_asset_folder(config['exp_dir'])
    experiment.end()

if __name__ == "__main__":
    main()
