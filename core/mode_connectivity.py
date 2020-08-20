import torch
import torch.nn as nn
import numpy as np
from .utils import DEVICE, load_model, assign_weights, flatten_params
from .train_methods import eval_single_epoch

def get_line_loss(start_w, w, loader, config):
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