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
    
    return parser


def check_args(parser):
    """Check and store the arguments as well as set up the save_dir"""
    args = parser.parse_args()

    ## set up save_dir
    save_dir = os.path.join(FILE_DIR, '../results', args.model, args.dataset, f'seed{args.random_seed}', 'defences', 'ttltt')
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

class Dataset_with_testTargets(torch.utils.data.Dataset):
    def __init__(self, original_dataset :torch.utils.data.Dataset, new_targets):
        self.original_dataset = original_dataset
        self.new_targets = new_targets

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, label = self.original_dataset.__getitem__(idx)
        target = self.new_targets[idx]
        return image, label, target
    

class TTLTTTrainer(BaseTrainer):
    """ Target Training data looks Like Target Testing data trainer """

    def __init__(self, the_args, save_dir, datasetloader: DatasetLoader):
        self.args = the_args
        self.save_dir = save_dir
        self.set_cuda_device()
        self.set_seed()
        self.set_logger()
        self.set_criterion()

        trainset = datasetloader.tr_set
        targets = self.get_targets(
            train_labels = trainset.targets,
            data_loader = datasetloader.get_shadow_test_loader()
        )
        dataloader = datasetloader.get_dataloader()

        self.trainloader = dataloader(
            Dataset_with_testTargets(trainset, targets),
            suffle=True
        )
        self.trainloader_without_new_targets = datasetloader.get_train_loader()
        self.testloader = datasetloader.get_test_loader()
    

    def get_targets(self, train_labels, data_loader):
        """Get the new targets for the training dataset"""

        # Retrieve the targets if they have already been drawn, otherwise generate them
        file_path = os.path.join(self.save_dir, f'targets.npy')
        if os.path.exists(file_path):
            targets = np.load(file_path)
        else:
            vanilla_model = torch.load(os.path.join(self.save_dir, '..', 'vanilla', 'model.pt')).to(self.device)
            vanilla_model.eval()

            possible_targets = []
            possible_labels = []
            with torch.no_grad():
                for fea, lab in data_loader:
                    target = torch.softmax(vanilla_model(fea), dim=1)
                    possible_targets.append( target.cpu().numpy() )
                    possible_labels.append( lab.cpu().numpy() )

            possible_targets = np.concatenate(possible_targets, axis=0)
            possible_labels = np.concatenate(possible_labels, axis=0)

            # Remove misclassified targets
            mask = possible_targets.argmax(axis=1) == possible_labels
            possible_targets = possible_targets[mask]
            possible_labels  = possible_labels[mask]

            idx = np.zeros(shape=len(train_labels), dtype=int) -1
            for lab in set(possible_labels):
                possible_idx = np.arange(0, len(possible_labels))[possible_labels == lab]
                idx[train_labels == lab] = np.random.choice(possible_idx, size=sum(train_labels == lab))
           
            targets = possible_targets[idx]
            np.save(file_path, targets)
        
        return targets


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

        for batch_idx, (inputs, labels, targets) in (pbar := tqdm(enumerate(self.trainloader))):
            inputs, labels, targets = inputs.to(self.device), labels.to(self.device), targets.to(self.device)

            ### Record the data loading time
            dataload_time.update(time.time() - time_stamp)

            ### Output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

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
        train_losses = get_all_losses(self.trainloader_without_new_targets, model, self.crossentropy_noreduce, self.device)
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
    trainer = TTLTTTrainer(args, save_dir, datasetloader)
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