import os
import errno
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml


__all__ = ['mkdir', 'savefig', 'load_yaml', 'write_yaml', 'get_all_loses', 'plot_hist']


def mkdir(path):
    """make dir if not exist"""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def savefig(fname, dpi=None):
    dpi = 150 if dpi == None else dpi
    plt.savefig(fname, dpi=dpi, format='png')


def load_yaml(filepath):
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    return data


def write_yaml(data, filepath):
    with open(filepath, 'w') as f:
        yaml.dump(data, f)


def get_all_losses(dataloader, model, criterion, device):
    model.eval()

    losses = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            ### Forward
            outputs = model(inputs)

            ### Evaluate
            loss = criterion(outputs, targets)
            losses.append(loss.cpu().numpy())

    losses = np.concatenate(losses)
    return losses


def plot_hist(values, names, save_file):
    plt.figure()
    bins = np.histogram(np.hstack(values), bins=50)[1]
    for val, name in zip(values, names):
        plt.hist(val, bins=bins, alpha=0.5, label=name)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(save_file, dpi=150, format='png')
    plt.close()