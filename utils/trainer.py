import os
import time
import torch
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm 


from .helper import savefig, get_all_losses
from .eval import accuracy
from .logger import AverageMeter, Logger
from .datasets.loader import DatasetLoader

__all__ = ['BaseTrainer']


class BaseTrainer():
    """ Base training procedure """

    def __init__(self, the_args, save_dir, datasetloader :DatasetLoader):
        """The function to initialize this class."""
        self.args = the_args
        self.save_dir = save_dir
        self.set_cuda_device()
        self.set_seed()
        self.set_logger()
        self.set_criterion()
        self.trainloader = datasetloader.get_train_loader()
        self.testloader  = datasetloader.get_test_loader()

    def set_cuda_device(self):
        """The function to set CUDA device."""
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

    def set_seed(self):
        """Set random seed"""
        random.seed(self.args.random_seed)
        torch.manual_seed(self.args.random_seed)
        np.random.seed(self.args.random_seed)
        if self.use_cuda:
            torch.cuda.manual_seed_all(self.args.random_seed)

    def set_logger(self):
        """Set up logger"""
        title = self.args.dataset
        self.start_epoch = 0
        logger = Logger(os.path.join(self.save_dir, 'log.txt'), title=title)
        logger.set_names(['Train Loss', 'Val Loss', 'Train Acc', 'Val Acc', 'Train Acc 5', 'Val Acc 5', 'Time'])
        self.logger = logger

    def set_criterion(self):
        """Set up criterion"""
        self.criterion = nn.CrossEntropyLoss()
        self.crossentropy = nn.CrossEntropyLoss()
        self.crossentropy_noreduce = nn.CrossEntropyLoss(reduction='none')

    def train(self, model, optimizer, *args):
        """Train"""
        model.train()
        criterion = self.criterion
        losses = AverageMeter()
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
            loss = criterion(outputs, labels)

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

    def test(self, model):
        """Test"""
        model.eval()
        criterion = self.crossentropy
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        batch_time = AverageMeter()
        dataload_time = AverageMeter()
        time_stamp = time.time()

        with torch.no_grad():
            for batch_idx, (inputs, labels) in (pbar := tqdm(enumerate(self.testloader))):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                ### Record the data loading time
                dataload_time.update(time.time() - time_stamp)

                ### Forward
                outputs = model(inputs)

                ### Evaluate
                loss = criterion(outputs, labels)
                prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

                ### Record the total time for processing the batch
                batch_time.update(time.time() - time_stamp)
                time_stamp = time.time()

                ### Progress bar
                pbar.set_description(
                    '  > test  ({batch:3}/{size:3}) loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f} | data_time: {data_time:.3f}s | batch_time: {bt:.3f}s |'.format(
                    batch = batch_idx + 1,
                    size = len(self.testloader),
                    loss = losses.avg,
                    top1 = top1.avg,
                    top5 = top5.avg,
                    data_time = dataload_time.avg,
                    bt = batch_time.avg,
                ))
        return (losses.avg, top1.avg, top5.avg)

    def get_loss_distributions(self, model):
        """ Obtain the member and nonmember loss distributions"""
        train_losses = get_all_losses(self.trainloader, model, self.crossentropy_noreduce, self.device)
        test_losses = get_all_losses(self.testloader, model, self.crossentropy_noreduce, self.device)
        return train_losses, test_losses

    def logger_plot(self):
        """ Visualize the training progress"""
        self.logger.plot(['Train Loss', 'Val Loss'])
        savefig(os.path.join(self.save_dir, 'loss.png'))

        self.logger.plot(['Train Acc', 'Val Acc'])
        savefig(os.path.join(self.save_dir, 'acc.png'))