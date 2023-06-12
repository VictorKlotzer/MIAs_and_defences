import os
import sys
import argparse
import random
import shutil
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_DIR, '../'))

import utils.models as models
from utils.helper import mkdir, load_yaml, write_yaml, plot_hist, savefig
from utils.eval import accuracy
from utils.logger import AverageMeter, Logger
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
    
    parser.add_argument('--lr_attack', type=float, default=0.001, help='attack learning rate')
    parser.add_argument('--alpha', type=float, default=1.0, help='weight for the adversarial loss')
    parser.add_argument('--attack_steps', type=int, default=5, help='attacker update steps per one target step')
    
    return parser

def check_args(parser):
    """Check and store the arguments as well as set up the save_dir"""
    args = parser.parse_args()

    ## set up save_dir
    save_dir = os.path.join(FILE_DIR, '../results', args.model, args.dataset, f'seed{args.random_seed}', 'defences', 'advreg')
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


#############################################################################################################
# helper functions
#############################################################################################################
class Attack(nn.Module):
    def __init__(self, input_dim, num_classes=1, hiddens=[100]):
        super(Attack, self).__init__()
        self.layers = []
        for i in range(len(hiddens)):
            if i == 0:
                layer = nn.Linear(input_dim, hiddens[i])
            else:
                layer = nn.Linear(hiddens[i - 1], hiddens[i])
            self.layers.append(layer)
        self.last_layer = nn.Linear(hiddens[-1], num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        output = x
        for layer in self.layers:
            output = self.relu(layer(output))
        output = self.last_layer(output)
        output = self.sigmoid(output)
        return output


def accuracy_binary(output, target):
    """Computes the accuracy for binary classification"""
    batch_size = target.size(0)

    pred = output.view(-1) >= 0.5
    truth = target.view(-1) >= 0.5
    acc = pred.eq(truth).float().sum(0).mul_(100.0 / batch_size)
    return acc


class AdvRegTrainer(BaseTrainer):
    def __init__(self, the_args, save_dir, datasetloader: DatasetLoader):
        super().__init__(the_args, save_dir, datasetloader)

        self.nb_classes = datasetloader.nb_classes
        self.alpha = self.args.alpha
        self.attack_member_loader = self.trainloader
        self.attack_nonmember_loader = datasetloader.get_shadow_train_loader()

    def set_logger(self):
        """Set up logger"""
        title = self.args.dataset + '_' + self.args.model
        self.start_epoch = 0
        logger = Logger(os.path.join(self.save_dir, 'log.txt'), title=title)
        logger.set_names(['Train Loss', 'Val Loss', 'Train Acc', 'Val Acc', 'Train Acc 5', 'Val Acc 5',
                          'Attack Train Loss', 'Attack Train Acc', 'Attack Test Acc', 'Time'])
        self.logger = logger

    def set_criterion(self):
        self.crossentropy_noreduce = nn.CrossEntropyLoss(reduction='none')
        self.crossentropy = nn.CrossEntropyLoss()
        self.criterion = self.crossentropy
        self.attack_criterion = nn.MSELoss()

    def attack_input_transform(self, x, y):
        """Transform the input to attack model"""
        out_x = x
        out_x, _ = torch.sort(out_x, dim=1)
        one_hot = torch.from_numpy((np.zeros((y.size(0), self.nb_classes)) - 1)).cuda().type(
            torch.cuda.FloatTensor) 
        out_y = one_hot.scatter_(1, y.type(torch.cuda.LongTensor).view([-1, 1]).data, 1)
        return out_x, out_y

    def logger_plot(self):
        """ Visualize the training progress"""
        self.logger.plot(['Train Loss', 'Val Loss'])
        savefig(os.path.join(self.save_dir, 'loss.png'))

        self.logger.plot(['Train Acc', 'Val Acc'])
        savefig(os.path.join(self.save_dir, 'acc.png'))

        self.logger.plot(['Attack Train Acc', 'Attack Test Acc'])
        savefig(os.path.join(self.save_dir, 'attack_acc.png'))

    def train_privately(self, target_model, attack_model, target_optimizer, num_batches=10_000):
        """ Target model should minimize the CE while making the attacker's output close to 0.5"""
        target_model.train()
        attack_model.eval()

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        batch_time = AverageMeter()
        dataload_time = AverageMeter()
        time_stamp = time.time()

        max_batches = min(num_batches, len(self.attack_member_loader))

        for batch_idx, (inputs, labels) in (pbar := tqdm(enumerate(self.trainloader))):
            if batch_idx >= num_batches:
                break
            dataload_time.update(time.time() - time_stamp)

            ### Forward and compute loss
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = target_model(inputs)
            inference_input_x, inference_input_y = self.attack_input_transform(outputs, labels)
            inference_output = attack_model(inference_input_x, inference_input_y)
            loss = self.criterion(outputs, labels) + ((self.alpha) * (torch.mean((inference_output)) - 0.5))

            ### Record accuracy and loss
            prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size()[0])
            top1.update(prec1.item(), inputs.size()[0])
            top5.update(prec5.item(), inputs.size()[0])

            ### Optimization
            target_optimizer.zero_grad()
            loss.backward()
            target_optimizer.step()

            ### Record the total time for processing the batch
            batch_time.update(time.time() - time_stamp)
            time_stamp = time.time()
            
            ### Progress bar
            pbar.set_description(
                '  > train defender ({batch:3}/{size:3}) loss: {loss:.4f} | top1: {top1: .4f} | data_time: {data_time:.3f}s | batch_time: {bt:.3f}s |'.format(
                batch = batch_idx + 1,
                size = max_batches,
                loss = losses.avg,
                top1 = top1.avg,
                data_time = dataload_time.avg,
                bt = batch_time.avg,
            ))

        return (losses.avg, top1.avg, top5.avg)

    def train_attack(self, target_model, attack_model, attack_optimizer, num_batches=100_000):
        """ Train pseudo attacker"""
        target_model.eval()
        attack_model.train()

        losses = AverageMeter()
        top1 = AverageMeter()
        batch_time = AverageMeter()
        dataload_time = AverageMeter()
        time_stamp = time.time()

        max_batches = min(num_batches, len(self.attack_member_loader))

        for batch_idx, (member, nonmember) in (pbar := tqdm(enumerate(zip(self.attack_member_loader, self.attack_nonmember_loader)))):
            if batch_idx >= num_batches:
                break
            dataload_time.update(time.time() - time_stamp)

            inputs_member, labels_member = member
            inputs_member, labels_member = inputs_member.to(self.device), labels_member.to(self.device)
            inputs_nonmember, labels_nonmember = nonmember
            inputs_nonmember, labels_nonmember = inputs_nonmember.to(self.device), labels_nonmember.to(self.device)
            
            outputs_member_x, outputs_member_y = self.attack_input_transform(target_model(inputs_member),
                                                                             labels_member)
            outputs_nonmember_x, outputs_nonmember_y = self.attack_input_transform(target_model(inputs_nonmember),
                                                                                   labels_nonmember)
            
            attack_input_x = torch.cat((outputs_member_x, outputs_nonmember_x))
            attack_input_y = torch.cat((outputs_member_y, outputs_nonmember_y))
            attack_labels = np.zeros((inputs_member.size()[0] + inputs_nonmember.size()[0]))
            attack_labels[:inputs_member.size()[0]] = 1.  # member=1
            attack_labels[inputs_member.size()[0]:] = 0.  # nonmember=0

            indices = np.arange(len(attack_input_x))
            np.random.shuffle(indices)
            attack_input_x = attack_input_x[indices]
            attack_input_y = attack_input_y[indices]
            attack_labels = attack_labels[indices]
            is_member_labels = torch.from_numpy(attack_labels).type(torch.FloatTensor).to(self.device)
            attack_output = attack_model(attack_input_x, attack_input_y).view(-1)

            ### Record accuracy and loss
            loss_attack = self.attack_criterion(attack_output, is_member_labels)
            prec1 = accuracy_binary(attack_output.data, is_member_labels.data)
            losses.update(loss_attack.item(), len(attack_output))
            top1.update(prec1.item(), len(attack_output))

            ### Optimization
            attack_optimizer.zero_grad()
            loss_attack.backward()
            attack_optimizer.step()

            batch_time.update(time.time() - time_stamp)
            time_stamp = time.time()

            ### Progress bar
            pbar.set_description(
                '  > train attacker ({batch:3}/{size:3}) loss: {loss:.4f} | top1: {top1: .4f} | data_time: {data_time:.3f}s | batch_time: {bt:.3f}s |'.format(
                batch = batch_idx + 1,
                size = max_batches,
                loss = losses.avg,
                top1 = top1.avg,
                data_time = dataload_time.avg,
                bt = batch_time.avg,
            ))

        return (losses.avg, top1.avg)

    def test_attack(self, target_model, attack_model):
        """ Test pseudo attack model"""
        target_model.eval()
        attack_model.eval()

        losses = AverageMeter()
        top1 = AverageMeter()
        batch_time = AverageMeter()
        dataload_time = AverageMeter()

        time_stamp = time.time()

        with torch.no_grad():
            for batch_idx, (member, nonmember) in (pbar := tqdm(enumerate(zip(self.attack_member_loader, self.attack_nonmember_loader)))):
                inputs_member, labels_member = member
                inputs_member, labels_member = inputs_member.to(self.device), labels_member.to(self.device)
                inputs_nonmember, labels_nonmember = nonmember
                inputs_nonmember, labels_nonmember = inputs_nonmember.to(self.device), labels_nonmember.to(self.device)

                ### Record the data loading time
                dataload_time.update(time.time() - time_stamp)

                ### Forward
                outputs_member_x, outputs_member_y = self.attack_input_transform(target_model(inputs_member), labels_member)
                outputs_nonmember_x, outputs_nonmember_y = self.attack_input_transform(target_model(inputs_nonmember), labels_nonmember)
                attack_input_x = torch.cat((outputs_member_x, outputs_nonmember_x))
                attack_input_y = torch.cat((outputs_member_y, outputs_nonmember_y))
                attack_labels = np.zeros((inputs_member.size()[0] + inputs_nonmember.size()[0]))
                attack_labels[:inputs_member.size()[0]] = 1.  # member=1
                attack_labels[inputs_member.size()[0]:] = 0.  # nonmember=0
                is_member_labels = torch.from_numpy(attack_labels).type(torch.FloatTensor).to(self.device)
                attack_output = attack_model(attack_input_x, attack_input_y).view(-1)

                ### Evaluate
                loss_attack = self.attack_criterion(attack_output, is_member_labels)
                prec1 = accuracy_binary(attack_output.data, is_member_labels.data)
                losses.update(loss_attack.item(), len(attack_output))
                top1.update(prec1.item(), len(attack_output))

                ### Record the total time for processing the batch
                batch_time.update(time.time() - time_stamp)
                time_stamp = time.time()

                ### Progress bar
                pbar.set_description(
                    '  > test attacker  ({batch:3}/{size:3}) loss: {loss:.4f} | top1: {top1: .4f} | data_time: {data_time:.3f}s | batch_time: {bt:.3f}s |'.format(
                    batch = batch_idx + 1,
                    size = len(self.attack_member_loader),
                    loss = losses.avg,
                    top1 = top1.avg,
                    data_time = dataload_time.avg,
                    bt = batch_time.avg,
                ))

        return (losses.avg, top1.avg)


########################################################################
### main function
########################################################################
def main():
    args, save_dir = check_args(parse_arguments())

    model_architecture = models.__dict__[args.model] # function alias that takes the number of classes as argument
    
    datasetloader = DatasetLoader(args.dataset, args.random_seed, args.batchsize)
    nb_classes = datasetloader.nb_classes

    ### Set up trainer and model
    trainer = AdvRegTrainer(args, save_dir, datasetloader)
    model = model_architecture(nb_classes)
    model = torch.nn.DataParallel(model)
    model.to(trainer.device)
    torch.backends.cudnn.benchmark = True
    print('Total params: %.2f' % (sum(p.numel() for p in model.parameters())))
    optimizer = optim.Adam(model.parameters())

    ### Set up attack model
    attack_model = Attack(input_dim=nb_classes)
    attack_model = torch.nn.DataParallel(attack_model)
    attack_model = attack_model.to(trainer.device)
    attack_optimizer = optim.Adam(attack_model.parameters(), lr=args.lr_attack)

    ### Set up logger
    logger = trainer.logger

    ### Training
    for epoch in range(args.nb_epochs):
        t0 = time.time()
        if epoch < 3:
            train_loss, train_acc, train_acc5 = trainer.train(model, optimizer)
            attack_loss, attack_acc = 0, 0
            for _ in range(args.attack_steps):
                attack_loss, attack_acc = trainer.train_attack(model, attack_model, attack_optimizer)
            test_loss, test_acc, test_acc5 = trainer.test(model)
            attack_test_loss, attack_test_acc = trainer.test_attack(model, attack_model)

        else:
            attack_loss, attack_acc = trainer.train_attack(model, attack_model, attack_optimizer)
            train_loss, train_acc, train_acc5 = trainer.train_privately(model, attack_model, optimizer)
            test_loss, test_acc, test_acc5 = trainer.test(model)
            attack_test_loss, attack_test_acc = trainer.test_attack(model, attack_model)
        
        epoch_time = time.time() - t0
        logger.append([train_loss, test_loss, train_acc, test_acc, train_acc5, test_acc5,
                       attack_loss, attack_acc, attack_test_acc, epoch_time])
        print(f'Epoch {epoch}: {train_acc = :.3f} | {test_acc = :.3f} | time = {epoch_time:.3f}s')


    ### Save model
    torch.save(model, os.path.join(save_dir, 'model.pt'))

    ### Visualize
    trainer.logger_plot()
    train_losses, test_losses = trainer.get_loss_distributions(model)
    plot_hist([train_losses, test_losses], ['train', 'val'], os.path.join(save_dir, 'hist_ep%d.png' % epoch))



if __name__ == '__main__':
    main()
