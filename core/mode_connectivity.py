import torch
import torch.nn as nn
import numpy as np
from .utils import DEVICE, load_model, assign_weights, flatten_params
# from .train_methods import eval_single_epoch

def get_line_loss(start_w, w, loader, config):
    interpolation  = None
    if 'line' in config['lmc_interpolation'] or 'integral' in config['lmc_interpolation']:
        interpolation = 'linear'
    elif 'stochastic' in config['lmc_interpolation']:
        interpolation = 'stochastic'
    else:
        raise Exception("non-implemented interpolation")

    m = load_model('{}/{}.pth'.format(config['exp_dir'], 'init')).to(DEVICE)
    total_loss = 0
    accum_grad = None
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    print("DEBUG >> LMC interpolation is >> {}".format(interpolation))
    if interpolation == 'linear':
        for t in np.arange(0.0, 1.01, 1.0/float(config['lmc_line_samples'])):
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

    elif interpolation == 'stochastic':
        for data, target, task_id in loader:
                grads = []
                t = np.random.uniform()
                cur_weight = start_w + (w - start_w) * t
                m = assign_weights(m, cur_weight).to(DEVICE)
                m.eval()
                data = data.to(DEVICE)#.view(-1, 784)
                target = target.to(DEVICE)
                output = m(data, task_id)
                current_loss = criterion(output, target)
                current_loss.backward()
                for name, param in m.named_parameters():
                    grads.append(param.grad.view(-1))
                grads = torch.cat(grads)
                if accum_grad is None:
                    accum_grad = grads
                else:
                    accum_grad += grads
        return accum_grad

    else:
        return None

def get_clf_loss(net, loader):
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    net.eval()
    test_loss = 0
    count = 0

    for data, target, task_id in loader:
            count += len(target)
            data = data.to(DEVICE)#.view(-1, 784)
            target = target.to(DEVICE)
            output = net(data, task_id)
            test_loss += criterion(output, target)#*len(target)
    test_loss /= count
    return test_loss

def bezier_path_opt(w_1, w_2, theta, config):
    accum_grad = None
    for t in np.arange(0.0, 1.01, 1.0/float(config['lmc_line_samples'])):
            grads = []
            # cur_weight = start_w + (w - start_w) * t
            cur_weight = ((1-t)**2)*w_1 + 2*t*(1-t)*theta + (t**2)*w_2
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

