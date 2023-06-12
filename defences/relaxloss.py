import os
import sys
import argparse
import time
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from functools import partial
from tqdm import tqdm

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_DIR, '../'))

import utils.models as models
from utils.helper import mkdir, load_yaml, write_yaml, plot_hist
from utils.eval import accuracy
from utils.logger import AverageMeter
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

    parser.add_argument('--alpha', type=float, help='the desired loss level')
    parser.add_argument('--upper', type=float, help='upper confidence level')

    return parser

def check_args(parser):
    """Check and store the arguments as well as set up the save_dir"""
    args = parser.parse_args()

    ## set up save_dir
    save_dir = os.path.join(FILE_DIR, '../results', args.model, args.dataset, f'seed{args.random_seed}', 'defences', 'relaxloss')
    mkdir(save_dir)

    ## load configs and store the parameters
    default_file = os.path.join(FILE_DIR, 'configs', args.dataset, args.model, 'default.yml')
    if not os.path.exists(default_file): default_file = os.path.join(FILE_DIR, 'configs', 'default.yml')
    default_configs = load_yaml(default_file)

    try:
        default_configs['alpha'] = default_configs['relaxloss_alpha']
        default_configs['upper'] = default_configs['relaxloss_upper']
        del default_configs['relaxloss_alpha'], default_configs['relaxloss_upper']
    except:
        raise NotImplementedError(f'alpha and upper parameters need to be defined in the file {default_file} as relaxloss_alpha and relaxloss_upper')

    parser.set_defaults(**default_configs)
    args = parser.parse_args()
    if None in [z[1] for z in args._get_kwargs()]:
        raise Exception(f'No argument can have the value None: {args}')

    write_yaml(vars(args), os.path.join(save_dir, 'params.yml'))

    ## store this script
    shutil.copy(os.path.realpath(__file__), save_dir)

    return args, save_dir


#############################################################################################################
# helper functions
#############################################################################################################
def CrossEntropy_soft(input, target, reduction='mean'):
    '''
    cross entropy loss on soft labels
    :param input:
    :param target:
    :param reduction:
    :return:
    '''
    logprobs = F.log_softmax(input, dim=1)
    losses = -(target * logprobs)
    if reduction == 'mean':
        return losses.sum() / input.shape[0]
    elif reduction == 'sum':
        return losses.sum()
    elif reduction == 'none':
        return losses.sum(-1)
    
def one_hot_embedding(y, num_classes=10, dtype=torch.cuda.FloatTensor):
    '''
    apply one hot encoding on labels
    :param y: class label
    :param num_classes: number of classes
    :param dtype: data type
    :return:
    '''
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes).type(dtype)
    return zeros.scatter(scatter_dim, y_tensor, 1)
    

class RelaxLossTrainer(BaseTrainer):
    def __init__(self, the_args, save_dir, datasetloader :DatasetLoader):
        super().__init__(the_args, save_dir, datasetloader)
        self.nb_classes = datasetloader.nb_classes

    def set_criterion(self):
        """Set up the relaxloss training criterion"""
        self.crossentropy_noreduce = nn.CrossEntropyLoss(reduction='none')
        self.crossentropy_soft = partial(CrossEntropy_soft, reduction='none')
        self.crossentropy = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.alpha = self.args.alpha
        self.upper = self.args.upper

    def train(self, model, optimizer, epoch):
        model.train()

        losses = AverageMeter()
        losses_ce = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        batch_time = AverageMeter()
        dataload_time = AverageMeter()
        time_stamp = time.time()

        for batch_idx, (inputs, labels) in (pbar := tqdm(enumerate(self.trainloader))):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            ### Record the data loading time
            dataload_time.update(time.time() - time_stamp)

            ### Output
            outputs = model(inputs)
            loss_ce_full = self.crossentropy_noreduce(outputs, labels)
            loss_ce = torch.mean(loss_ce_full)

            if epoch % 2 == 0:  # gradient ascent/ normal gradient descent
                loss = (loss_ce - self.alpha).abs()
            else:
                if loss_ce > self.alpha:  # normal gradient descent
                    loss = loss_ce
                else:  # posterior flattening
                    pred = torch.argmax(outputs, dim=1)
                    correct = torch.eq(pred, labels).float()
                    confidence_target = self.softmax(outputs)[torch.arange(labels.size(0)), labels]
                    confidence_target = torch.clamp(confidence_target, min=0., max=self.upper)
                    confidence_else = (1.0 - confidence_target) / (self.nb_classes - 1)
                    onehot = one_hot_embedding(labels, num_classes=self.nb_classes)
                    soft_targets = onehot * confidence_target.unsqueeze(-1).repeat(1, self.nb_classes) \
                                   + (1 - onehot) * confidence_else.unsqueeze(-1).repeat(1, self.nb_classes)
                    loss = (1 - correct) * self.crossentropy_soft(outputs, soft_targets) - 1. * loss_ce_full
                    loss = torch.mean(loss)

            ### Record accuracy and loss
            prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            losses_ce.update(loss_ce.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            ### Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### Record the total time for processing the batch
            batch_time.update(time.time() - time_stamp)
            time_stamp = time.time()

            ### Progress bar
            pbar.set_description(
                '  > train ({batch:3}/{size:3}) loss: {loss:.4f} CEloss: {CEloss:.4f} | top1: {top1: .4f} | top5: {top5: .4f} | data_time: {data_time:.3f}s | batch_time: {bt:.3f}s |'.format(
                batch = batch_idx + 1,
                size = len(self.trainloader),
                loss = losses.avg,
                CEloss = losses_ce.avg,
                top1 = top1.avg,
                top5 = top5.avg,
                data_time = dataload_time.avg,
                bt = batch_time.avg,
            ))

        return (losses_ce.avg, top1.avg, top5.avg)


#############################################################################################################
# main function
#############################################################################################################
def main():
    args, save_dir = check_args(parse_arguments())

    model_architecture = models.__dict__[args.model] # function alias that takes the number of classes as argument
    
    datasetloader = DatasetLoader(args.dataset, args.random_seed, args.batchsize)
    nb_classes = datasetloader.nb_classes

    ### Set up trainer and model
    trainer = RelaxLossTrainer(args, save_dir, datasetloader)
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
        train_loss, train_acc, train_acc5 = trainer.train(model, optimizer, epoch)
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
