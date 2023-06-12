from __future__ import print_function, absolute_import
import numpy as np
import torch
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import metrics


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    try:
        _, pred = output.topk(maxk, 1, True, True)
    except:
        _, pred = output.topk(2, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = torch.reshape(correct[:k], (-1,)).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
