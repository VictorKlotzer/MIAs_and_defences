import os
import sys
import argparse
import time
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_DIR, '../'))

import utils.models as models
from utils.helper import mkdir, load_yaml, write_yaml, plot_hist, get_all_losses
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

    parser.add_argument('--temperatures_distrib', '-Tdist', type=str, help='temperatures distribution (for beta we use Beta(.5, .5))', default='unif',
                        choices=['unif', 'beta', 'norm', 'constant'])
    parser.add_argument('--temperatures_min', '-Tmin', type=float, help='temperatures minimum', default=.2)
    parser.add_argument('--temperatures_max', '-Tmax', type=float, help='temperatures maximum', default=10.0)
    parser.add_argument('--temperatures_mean', '-Tmean', type=float, help='temperatures mean (if Tdist is norm)', default=10.0)
    parser.add_argument('--temperatures_std', '-Tstd', type=float, help='temperatures std (if Tdist is norm)', default=1.0)
    
    return parser


def check_args(parser):
    """Check and store the arguments as well as set up the save_dir"""
    args = parser.parse_args()

    ## load configs and store the parameters
    default_file = os.path.join(FILE_DIR, 'configs', args.dataset, args.model, 'default.yml')
    if not os.path.exists(default_file): default_file = os.path.join(FILE_DIR, 'configs', 'default.yml')
    default_configs = load_yaml(default_file)
    parser.set_defaults(**default_configs)
    args = parser.parse_args()
    if None in [z[1] for z in args._get_kwargs()]:
        raise Exception(f'No argument can have the value None: {args}')
    

    ## set up save_dir
    Tdist = args.temperatures_distrib
    Tmin, Tmax = args.temperatures_min, args.temperatures_max
    Tmean, Tstd = args.temperatures_mean, args.temperatures_std

    Tname = f'{Tdist}'
    if Tdist == 'norm':
        Tname += f'-{Tmean}-{Tstd}'
    elif Tdist == 'constant':
        assert Tmin == Tmax, 'For constant temperature distribution, give the same value to -Tmin and -Tmax'
        Tname += f'_{Tmin}'
    elif Tdist in ['unif', 'beta']:
        Tname += f'_{Tmin}-{Tmax}'
    else:
        raise NotImplementedError

    save_dir = os.path.join(FILE_DIR, '../results', args.model, args.dataset, f'seed{args.random_seed}', 'defences', 'rtt', Tname)
    mkdir(save_dir)

    write_yaml(vars(args), os.path.join(save_dir, 'params.yml'))

    ## store this script
    shutil.copy(os.path.realpath(__file__), save_dir)

    return args, save_dir



#############################################################################################################
# helper functions
#############################################################################################################
class Dataset_with_temperatures(torch.utils.data.Dataset):
    def __init__(self, original_dataset :torch.utils.data.Dataset, temperatures):
        self.original_dataset = original_dataset
        self.temperatures = temperatures

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, label = self.original_dataset.__getitem__(idx)
        temp = self.temperatures[idx]
        return image, temp, label
    

class RTTTrainer(BaseTrainer):
    """Trainer for RTT defence"""
    def __init__(self, the_args, save_dir, datasetloader :DatasetLoader):
        self.args = the_args
        self.save_dir = save_dir
        self.set_cuda_device()
        self.set_seed()
        self.set_logger()
        self.set_criterion()

        trainset = datasetloader.tr_set
        temperatures = self.get_temperatures(train_size=len(trainset))
        dataloader = datasetloader.get_dataloader()
        self.trainloader = dataloader(
            Dataset_with_temperatures(trainset, temperatures),
            suffle=True
        )
        self.trainloader_without_temperatures = datasetloader.get_train_loader()
        self.testloader = datasetloader.get_test_loader()

    def set_criterion(self):
        """Set up criterion"""
        self.criterion = nn.NLLLoss()
        self.crossentropy = nn.CrossEntropyLoss()
        self.crossentropy_noreduce = nn.CrossEntropyLoss(reduction='none')

    def get_temperatures(self, train_size):
        """Get the temperatures for the training dataset"""
        Tdist = self.args.temperatures_distrib
        Tmin, Tmax = self.args.temperatures_min, self.args.temperatures_max
        Tmean, Tstd = self.args.temperatures_mean, self.args.temperatures_std

        # Retrieve the temperatures if they have already been drawn, otherwise generate them
        file_path = os.path.join(self.save_dir, f'temperatures.npy')
        if os.path.exists(file_path):
            temperatures = np.load(file_path)
        else:
            if Tdist == 'unif':
                temperatures = np.random.rand(train_size, 1) * (Tmax - Tmin) + Tmin
            elif Tdist == 'beta':
                temperatures = np.random.beta(.5, .5, (train_size, 1)) * (Tmax - Tmin) + Tmin
            elif Tdist == 'norm':
                temperatures = np.random.normal(Tmean, Tstd, size=(train_size, 1))
            elif Tdist == 'constant':
                temperatures = np.zeros((train_size, 1)) + Tmin
            np.save(file_path, temperatures)
        
        return temperatures


    @staticmethod
    def __change_temperature_log(logprobit, T=1):
        if not (isinstance(T, int) or isinstance(T, float)):
            assert T.shape[0] == len(logprobit) and T.shape[1] == 1, 'T should have be two dimensional and have the same length as probit'

        res = logprobit * (1/T)
        log_sum_exp = torch.log(torch.exp(res).sum(axis=-1, keepdims=True))
        return res - log_sum_exp


    def train(self, model, optimizer):
        """Train"""
        model.train()
        criterion = self.criterion
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        batch_time = AverageMeter()
        dataload_time = AverageMeter()
        time_stamp = time.time()

        for batch_idx, (inputs, input_temperatures, labels) in (pbar := tqdm(enumerate(self.trainloader))):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            ### Record the data loading time
            dataload_time.update(time.time() - time_stamp)

            ### Output
            outputs = model(inputs)
            logprobits = torch.log_softmax(outputs, axis=1)
            logprobits = RTTTrainer.__change_temperature_log(logprobits, T=input_temperatures)
            loss = criterion(logprobits, labels)

            ### Record accuracy and loss
            prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
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
                '  > train ({batch:3}/{size:3}) loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f} | data_time: {data_time:.3f}s | batch_time: {bt:.3f}s |'.format(
                batch = batch_idx + 1,
                size = len(self.trainloader),
                loss = losses.avg,
                top1 = top1.avg,
                top5 = top5.avg,
                data_time = dataload_time.avg,
                bt = batch_time.avg,
            ))
 
        return (losses.avg, top1.avg, top5.avg)
    

    def get_loss_distributions(self, model):
        """ Obtain the member and nonmember loss distributions"""
        train_losses = get_all_losses(self.trainloader_without_temperatures, model, self.crossentropy_noreduce, self.device)
        test_losses = get_all_losses(self.testloader, model, self.crossentropy_noreduce, self.device)
        return train_losses, test_losses


#############################################################################################################
# main function
#############################################################################################################
def main():
    args, save_dir = check_args(parse_arguments())

    model_architecture = models.__dict__[args.model] # function alias that takes the number of classes as argument
    
    datasetloader = DatasetLoader(args.dataset, args.random_seed, args.batchsize)
    nb_classes = datasetloader.nb_classes

    ### Set up trainer and model
    trainer = RTTTrainer(args, save_dir, datasetloader)
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
