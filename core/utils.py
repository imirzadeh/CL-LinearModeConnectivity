import os
import torch
import numpy as np
import torch.nn as nn
import random
import string
from pathlib import Path
from core.models import MLP, ResNet18
from core.hessian_eigenthings import compute_hessian_eigenthings

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def log_comet_metric(exp, name, val, step):
    exp.log_metric(name=name, value=val, step=step)

def get_random_string(length):
    # Random string with the combination of lower and upper case
    letters = string.ascii_letters
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def save_model(model, path):
    torch.save(model.cpu(), path)

def load_model(path):
    model = torch.load(path)
    return model

def save_task_model_by_policy(model, task, policy, exp_dir):
    # task = 0 is the initilization
    if task == 0 or policy == 'init':
        save_model(model, '{}/init.pth'.format(exp_dir))

    # the first task model is the same for all 
    if task == 1:
        save_model(model, '{}/t_{}_seq.pth'.format(exp_dir, task))
        save_model(model, '{}/t_{}_lmc.pth'.format(exp_dir, task))
        save_model(model, '{}/t_{}_mtl.pth'.format(exp_dir, task))
    else:
        save_model(model, '{}/t_{}_{}.pth'.format(exp_dir, task, policy))


def load_task_model_by_policy(task, policy, exp_dir):
    if task == 0 or policy == 'init':
        return load_model('{}/init.pth'.format(exp_dir))
    return load_model('{}/t_{}_{}.pth'.format(exp_dir, task, policy))


def flatten_params(m, numpy_output=True):
    total_params = []
    for param in m.parameters():
            total_params.append(param.view(-1))
    total_params = torch.cat(total_params)
    if numpy_output:
        return total_params.cpu().detach().numpy()
    return total_params

def assign_weights(m, weights):
    state_dict = m.state_dict(keep_vars=True)
    index = 0
    with torch.no_grad():
        for param in state_dict.keys():
            if 'running_mean' in param or 'running_var' in param or 'num_batches_tracked' in param:
                continue
            # print(param, index)
            param_count = state_dict[param].numel()
            param_shape = state_dict[param].shape
            state_dict[param] =  nn.Parameter(torch.from_numpy(weights[index:index+param_count].reshape(param_shape)))
            index += param_count
    m.load_state_dict(state_dict)
    return m

def assign_grads(m, grads):
    state_dict = m.state_dict(keep_vars=True)
    index = 0
    for param in state_dict.keys():
        if 'running_mean' in param or 'running_var' in param or 'num_batches_tracked' in param:
                continue
        param_count = state_dict[param].numel()
        param_shape = state_dict[param].shape
        state_dict[param].grad =  grads[index:index+param_count].view(param_shape).clone()
        index += param_count
    m.load_state_dict(state_dict)
    return m

def get_norm_distance(m1, m2):
    m1 = flatten_params(m1, numpy_output=False)
    m2 = flatten_params(m2, numpy_output=False)
    return torch.norm(m1-m2, 2).item()


def get_cosine_similarity(m1, m2):
    m1 = flatten_params(m1)
    m2 = flatten_params(m2)
    cosine = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    return cosine(m1, m2)


def save_np_arrays(data, path):
    np.savez(file=path, **data)


def setup_experiment(experiment, config):
    Path(config['exp_dir']).mkdir(parents=True, exist_ok=True)
    init_model = ResNet18() if 'cifar' in config['dataset'] else MLP(config)
    save_model(init_model, '{}/init.pth'.format(config['exp_dir']))

    init_model_2 = ResNet18() if 'cifar' in config['dataset'] else MLP(config)
    save_model(init_model, '{}/init_2.pth'.format(config['exp_dir']))
    experiment.log_parameters(config)


class ContinualMeter:
    def __init__(self, name, n_tasks):
        self.name = name
        self.data = np.zeros((n_tasks, n_tasks))

    def update(self, current_task, target_task, metric):
        self.data[current_task-1][target_task-1] = round(metric, 3)

    def save(self, config):
        path = '{}/{}.csv'.format(config['exp_dir'], self.name)
        np.savetxt(path, self.data, delimiter=",")


def get_latex_str_for_minima(policy, task):
    if policy == 'seq':
        return r"$\hat{{w}}_{{{}}}".format(task)
    elif policy == 'lmc':
        return r"$\bar{{w}}_{{{}}}".format(task)
    elif policy == 'mtl':
        return r"$w^*_{{{}}}".format(task)
    else:
        raise Exception("unknown policy")

def get_latex_str_for_path(p1, t1, p2, t2):
    start = get_latex_str_for_minima(p1, t1)
    end = get_latex_str_for_minima(p2, t2)
    return start + r" \rightarrow " + end


def get_model_grads(model, loader):
    grads = []
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    count = 0
    test_loss = 0
    for data, target, task_id in loader:
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            count += len(target)
            output = model(data, task_id)
            curr_loss = criterion(output, target)
            curr_loss.backward()
    for param in model.parameters():
            grads.append(param.grad.view(-1))
    grads = torch.cat(grads)
    # print("Norm grad >> ", torch.norm(grads))
    return grads

def get_model_eigenspectrum(model, loader):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    use_gpu = True if DEVICE != 'cpu' else False

    new_loader = []
    for i in range(5):
        new_loader.append([loader[i][0], loader[i][1]])

    est_eigenvals, est_eigenvecs = compute_hessian_eigenthings(
        model,
        new_loader,
        criterion,
        num_eigenthings=50,
        power_iter_steps=500,
        power_iter_err_threshold=1e-5,
        momentum=0,
        use_gpu=use_gpu,
    )
    #key = 'task-{}-epoch-{}'.format(task_id, time-1)
    #hessian_eig_db[key] = est_eigenvals
    #EXPERIMENT_DIRECTORY+"/{}-vec.npy".format(key)
    #np.save(eig_vec_dir, est_eigenvecs)
    #return hessian_eig_db
    # print(" [[[[ Eigenvec shape] ]]]  >>> ".format(est_eigenvecs[0].shape))
    return est_eigenvals, est_eigenvecs


# if __name__ == "__main__":
#     m = MLP(10, 10)#ResNet18() #MLP(10, 10)

#     for n, p in m.named_parameters():
#         print(n, p.grad)
#     flats = flatten_params(m, numpy_output=True)

#     m = assign_weights_old(m , flats)
#     print("**"*10)
#     for n, p in m.named_parameters():
#         print(n, p.grad)
    # print(flats.shape)
    # dic = m.state_dict()
    # count = 0
    # for p in dic.keys():
    #     # print(p, count)
    #     count += dic[p].numel()

    # count = 0
    # print('**'*20)
    # for n, p in m.named_parameters():
    #     print(n, count)
    #     count += p.numel()
    # print(count)

    # m = assign_weights(m, p)

#     # model = MLP(100, 10)
#     init = load_model('./checkpoints/YOGby/init.pth')
#     t1 = load_model('./checkpoints/YOGby/t_1_seq.pth')
#     w1 = flatten_params(init, True)
#     w2 = flatten_params(t1, True)

#     m = assign_grads(init, torch.tensor((w1+w2)/3.0))
#     for n, p in m.named_parameters():
#         print(n, p.grad)

    # w3 = flatten_params(m, True)

    # for check in [1, 1000, 12000, 25000, -1]:
    #     print((w1[check] + w2[check])/2.0, w3[check])
