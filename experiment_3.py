from comet_ml import Experiment
import os
import torch
import numpy as np
import pandas as pd
from core.data_utils import get_all_loaders
from core.cka_utils import calculate_CKA
from core.train_methods import train_task_sequentially, train_task_MTL, eval_single_epoch
from core.utils import save_np_arrays, setup_experiment, log_comet_metric, get_random_string
from core.utils import save_task_model_by_policy, load_task_model_by_policy, flatten_params
from core.utils import assign_weights, get_norm_distance, ContinualMeter, load_model
from core.visualization import plot_contour, get_xy, plot_heat_map, plot_l2_map, plot_accs
from core.visualization import plot_single_interpolation, plot_multi_interpolations
from core.utils import get_model_grads, get_model_eigenspectrum

DATASET = 'rot-mnist'
HIDDENS = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRIAL_ID =  os.environ.get('NNI_TRIAL_JOB_ID', get_random_string(5))
EXP_DIR = './checkpoints/{}'.format(TRIAL_ID)



config = {
         # ---COMMON----
         'num_tasks': 2, 'per_task_rotation': 45, 'trial': TRIAL_ID, 'exp_dir': EXP_DIR,\
         'memory_size': 200, 'dataset': DATASET, 'device': DEVICE, 'momentum': 0.8,\
         'mlp_hiddens': HIDDENS, 'dropout': 0.0, 'lr_decay': 0.99, 'stable_sgd': False,\

          # ----Seq Model-----
          'seq_lr': 0.05, 'seq_batch_size': 64, 'seq_epochs': 15,\
          'lr_mtl': 0.05, 'epochs_mtl': 5, 'mtl_start_from_other_init': False,\

          # ------LMC models------
          'lmc_policy': 'offline', 'lmc_interpolation': 'linear',\
          'lmc_lr': 0.1, 'lmc_batch_size': 64, 'lcm_init_position': 0.5,\
          'lmc_line_samples': 5, 'lmc_epochs': 1,   
         }

seq_meter = ContinualMeter('seq_accs', config['num_tasks'])
mtl_meter = ContinualMeter('mtl_accs', config['num_tasks'])
#config = nni.get_next_parameter()
config['trial'] = TRIAL_ID
experiment = Experiment(api_key="1UNrcJdirU9MEY0RC3UCU7eAg", \
                        project_name="explore-eigens", \
                        workspace="cl-modeconnectivity", disabled=True)
    
loaders = get_all_loaders(config['dataset'], config['num_tasks'],\
                         config['seq_batch_size'], config['seq_batch_size'],\
                         config['memory_size'], config.get('per_task_rotation'))


def compute_direction_cosines(grads, eigenvecs):
    cosines = []
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    for v in eigenvecs:
        cosines.append(torch.abs(cos(-1.0*grads.cpu(), torch.from_numpy(v))).item())
    return cosines

def main():
    print('Started the trial >>', TRIAL_ID, 'for experiment 1')
    # init and save9
    setup_experiment(experiment, config)

    # convention:  init      =>  initialization
    # convention:  t_i_seq   =>  task i (sequential)
    # convention:  t_i_mtl   => task 1 ... i (multitask)
    # convention:  t_i_lcm   => task 1 ... i (Linear Mode Connectivity)

    eigen_spectrum = {1: {}, 2: {}}

    for task in range(1, config['num_tasks']+1):
        print('---- Task {} (seq) ----'.format(task))
        seq_model = train_task_sequentially(task, loaders['sequential'][task]['train'], config)
        
        eigenvals, eigenvecs = get_model_eigenspectrum(seq_model, loaders['sequential'][task]['val'])
        eigen_spectrum[task]['eigenvals'], eigen_spectrum[task]['eigenvecs']= eigenvals, eigenvecs
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
            accs_mtl, losses_mtl = [], []

            print('---- Task {} (mtl) ----'.format(task))
            mtl_model = train_task_MTL(task, loaders['full-multitask'][task]['train'], config, loaders['sequential'][1]['val'])
            grads_t1 = get_model_grads(mtl_model, loaders['sequential'][1]['val'])
            # grads_t2 = get_model_grads(mtl_model, loaders['sequential'][2]['val'])
            grads_t3 = get_model_grads(load_task_model_by_policy(1, 'seq', config['exp_dir']), False).to(DEVICE), loaders['sequential'][2]['val'])

            seq_1 = flatten_params(load_task_model_by_policy(1, 'seq', config['exp_dir']), False).cpu()
            seq_2 = flatten_params(load_task_model_by_policy(2, 'seq', config['exp_dir']), False).cpu()

            cosines_t1 = compute_direction_cosines(grads_t1, eigen_spectrum[1]['eigenvecs'])
            # cosines_t2 = compute_direction_cosines(grads_t2, eigen_spectrum[2]['eigenvecs'])
            cosines_t3 = compute_direction_cosines(grads_t3, eigen_spectrum[1]['eigenvecs'])

            cosine_d1 = compute_direction_cosines((flatten_params(mtl_model, False).cpu()-seq_1), eigen_spectrum[1]['eigenvecs'])
            cosine_d2 = compute_direction_cosines(seq_2-seq_1, eigen_spectrum[1]['eigenvecs'])
            print("cos 1 >> ", cosines_t1)
            # print("cos 2 >> ", cosines_t2)
            print("cos 3 >> ", cosines_t3)

            print("dir 1 >>", cosine_d1)
            print("dir 2 >>", cosine_d2)

            save_task_model_by_policy(mtl_model, task, 'mtl', config['exp_dir'])

            for prev_task in range(1, task+1):
                metrics_mtl =  eval_single_epoch(mtl_model, loaders['sequential'][prev_task]['val'])
                accs_mtl.append(metrics_mtl['accuracy'])
                losses_mtl.append(metrics_mtl['loss'])
                mtl_meter.update(task, prev_task, metrics['accuracy'])

                print('MTL >> ', prev_task, metrics_mtl)
                log_comet_metric(experiment, 't_{}_mtl_acc'.format(prev_task), metrics_mtl['accuracy'], task)
                log_comet_metric(experiment, 't_{}_mtl_loss'.format(prev_task), round(metrics_mtl['loss'], 5), task)
            log_comet_metric(experiment, 'avg_acc_mtl', np.mean(accs_mtl),task)
            log_comet_metric(experiment, 'avg_loss_mtl', np.mean(losses_mtl),task)
        print()

    seq_meter.save(config)
    mtl_meter.save(config)


    experiment.log_asset_folder(config['exp_dir'])
    experiment.end()

if __name__ == "__main__":
    main()
