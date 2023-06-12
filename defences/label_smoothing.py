import os
import sys
import argparse
import time
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_DIR, '../'))

import utils.models as models
from utils.helper import mkdir, load_yaml, write_yaml, plot_hist
from utils.datasets.loader import DatasetLoader
from utils.trainer import BaseTrainer


#############################################################################################################
# get and save the arguments
#############################################################################################################
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model architecture', default='resnet18')
    parser.add_argument('--dataset', type=str, help='dataset name', default='CIFAR10')
    parser.add_argument('--random_seed', '-s', type=int, default=123, help='random seed')
    parser.add_argument('--nb_epochs', '-ep', type=int, help='number of epochs')
    
    parser.add_argument('--epsilon', type=float, default=0.1, help='epsilon for label smoothing')
    
    return parser


def check_args(parser):
    """Check and store the arguments as well as set up the save_dir"""
    args = parser.parse_args()

    ## set up save_dir
    save_dir = os.path.join(FILE_DIR, '../results', args.model, args.dataset, f'seed{args.random_seed}', 'defences', 'label_smoothing')
    mkdir(save_dir)

    ## load configs and store the parameters
    default_file = os.path.join(FILE_DIR, 'configs', args.dataset, args.model, 'default.yml')
    if not os.path.exists(default_file): default_file = os.path.join(FILE_DIR, 'configs', 'default.yml')
    default_configs = load_yaml(default_file)
    parser.set_defaults(**default_configs)
    args = parser.parse_args()
    if None in [z[1] for z in args._get_kwargs()]:
        raise Exception(f'No argument can have the value None: {args}')

    write_yaml(vars(args), os.path.join(save_dir, 'params.yml'))

    ## store this script
    shutil.copy(os.path.realpath(__file__), save_dir)

    return args, save_dir


################################################################################
# helper functions
################################################################################
class LabelSmoothingCrossEntropy(nn.Module):
    """
    epsilon * 1/K \sum_{i=1}^K -\log p(yi|x) + (1-epsilon)* \sum_{i=1}^K -ti \log p(yi|x)
    =  epsilon * KL (u || p(y|x)) + const + (1-epsilon)* CE(p(y|x), target)
    """
    def __init__(self, epsilon :float =.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return self.epsilon * 1/n * loss + (1 - self.epsilon) * nll

    @staticmethod
    def reduce_loss(loss, reduction='mean'):
        return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LSTrainer(BaseTrainer):
    def set_criterion(self):
        """Set the criterion to be label smoothing"""
        self.crossentropy_noreduce = nn.CrossEntropyLoss(reduction='none')
        self.criterion = LabelSmoothingCrossEntropy(epsilon=self.args.epsilon)
        self.crossentropy = nn.CrossEntropyLoss()


#############################################################################################################
# main function
#############################################################################################################
def main():
    args, save_dir = check_args(parse_arguments())

    model_architecture = models.__dict__[args.model] # function alias that takes the number of classes as argument
    
    datasetloader = DatasetLoader(args.dataset, args.random_seed, args.batchsize)
    nb_classes = datasetloader.nb_classes

    ### Set up trainer and model
    trainer = LSTrainer(args, save_dir, datasetloader)
    model = model_architecture(nb_classes)
    model = torch.nn.DataParallel(model)
    model.to(trainer.device)
    torch.backends.cudnn.benchmark = True
    print('Total params: %.2f' % (sum(p.numel() for p in model.parameters())))
    optimizer = optim.Adam(model.parameters())

    logger = trainer.logger

    ### Training
    for epoch in range(args.nb_epochs):
        t0 = time.time()
        train_loss, train_acc, train_acc5 = trainer.train(model, optimizer)
        test_loss, test_acc, test_acc5 = trainer.test(model)
        epoch_time = time.time() - t0
        logger.append([train_loss, test_loss, train_acc, test_acc, train_acc5, test_acc5, epoch_time])
        print(f'Epoch {epoch}: {train_acc = :.3f} | {test_acc = :.3f} | time = {epoch_time:.3f}s')

    ### Save model
    torch.save(model, os.path.join(save_dir, 'model.pt'))

    ### Visualize
    trainer.logger_plot()
    train_losses, test_losses = trainer.get_loss_distributions(model)
    plot_hist([train_losses, test_losses], ['train', 'val'], os.path.join(save_dir, 'hist_ep%d.png' % epoch))


if __name__ == '__main__':
    main()