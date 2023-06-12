import os
import sys
import argparse
import shutil
import numpy as np
import pandas as pd
import random
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Subset

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_DIR, '../'))

import utils.models as models
from utils.helper import mkdir, load_yaml, write_yaml, plot_hist
from utils.datasets.loader import DatasetLoader
from utils.trainer import BaseTrainer

"""
In case LiRA training did not finish, the same command can be used to continue the training :) 
"""


#############################################################################################################
# get and save the arguments
#############################################################################################################
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model architecture', default='resnet18')
    parser.add_argument('--dataset', type=str, help='dataset name', default='CIFAR10')
    parser.add_argument('--random_seed', '-s', type=int, default=123, help='random seed')
    parser.add_argument('--nb_epochs', '-ep', type=int, help='number of epochs')
    parser.add_argument('--nb_shadows', type=int, default=64, help='number of shadow models to use per challenger')
    parser.add_argument('--nb_challengers', type=int, default=5_000, help='number of challengers (members and non-members) to consider')
    
    return parser


def check_args(parser):
    '''check and store the arguments as well as set up the save_dir'''
    args = parser.parse_args()

    ## set up save_dir
    save_dir = os.path.join(FILE_DIR, '../results', args.model, args.dataset, f'seed{args.random_seed}', 'attacks', 'lira')
    mkdir(save_dir)
    mkdir(os.path.join(save_dir, 'shadows'))

    ## load configs and store the parameters
    default_file = os.path.join(FILE_DIR, '../defences', 'configs', args.dataset, args.model, 'default.yml')
    if not os.path.exists(default_file): default_file = os.path.join(FILE_DIR, 'configs', 'default.yml')
    default_configs = load_yaml(default_file)
    parser.set_defaults(**default_configs)
    args = parser.parse_args()
    if None in [z[1] for z in args._get_kwargs()]:
        raise Exception(f'No argument can have the value None: {args}')
    
    params_file_path = os.path.join(save_dir, 'params.yml')
    if os.path.exists(params_file_path):
        prev_params = load_yaml(params_file_path)
        if prev_params != vars(args):
            raise Exception(f'To run LiRA with other parameters than {prev_params}, delete lira folder of current random seed or choose another random seed')
    else:
        write_yaml(vars(args), params_file_path)

    ## store this script
    shutil.copy(os.path.realpath(__file__), save_dir)

    return args, save_dir


#############################################################################################################
# helper functions
#############################################################################################################
class TrainerForLiRA(BaseTrainer):
    """ Trainer for one shadow model """

    def __init__(self, the_args, save_dir, datasetloader :DatasetLoader, full_dataset, idx_train_data):
        """The function to initialize this class."""
        self.args = the_args
        self.save_dir = save_dir
        self.set_cuda_device()
        self.set_seed()
        self.set_logger()
        self.set_criterion()
        dataloader = datasetloader.get_dataloader()
        self.trainloader = dataloader(Subset(full_dataset, indices=idx_train_data), suffle=True)
        self.testloader  = datasetloader.get_shadow_test_loader()

   
class LiRA():
    """ LiRA attack for a given model and on a fixed subset of challengers """
    
    log_epsilon = 1e-30
    
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir
        self.set_cuda_device()
        self.set_seed()
        training_progress_file_path = os.path.join(self.save_dir, 'shadows', 'training_progress.yml')
        if os.path.exists(training_progress_file_path):
            self.nb_trained_shadows = load_yaml(training_progress_file_path)['nb_trained_shadows']
        else:
            write_yaml({'nb_trained_shadows':0}, training_progress_file_path)
            self.nb_trained_shadows = 0
        self.set_dataset()

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

    def set_dataset(self):
        self.datasetloader = DatasetLoader(self.args.dataset, self.args.random_seed, self.args.batchsize)
        train_data = self.datasetloader.s_tr_set # shadow train data
        chal_data  = self.datasetloader.chal_set # challengers
        self.full_data = ConcatDataset([train_data, chal_data])

        n = len(train_data)
        m = len(chal_data) # number of available challengers

        self.idx_train_data  = np.arange(n).astype(int)

        idx_challengers = np.arange(n, n + m).astype(int)
        nb_wanted_challengers = self.args.nb_challengers # number of wanted challengers
        self.idx_challengers = np.concatenate([
            idx_challengers[:m//2][:nb_wanted_challengers//2], # members
            idx_challengers[m//2:][:nb_wanted_challengers//2], # non-members
        ])
        
        # Set IN and OUT shadow training datasets
        if self.nb_trained_shadows == 0:
            train_data_per_shadow = len(self.datasetloader.tr_set) - nb_wanted_challengers//2
            shadow_datasets_idx, IN_shadow_models = self.create_shadow_datasets_idx(train_data_per_shadow)
            self.shadow_datasets_idx = shadow_datasets_idx

            np.save(os.path.join(self.save_dir, f'shadow_datasets_idx__nb_challengers={nb_wanted_challengers}.npy'), np.stack(shadow_datasets_idx))
            IN_shadow_models.to_pickle(os.path.join(self.save_dir, f'IN_shadow_models__nb_challengers={nb_wanted_challengers}.pkl'))
        
        elif self.nb_trained_shadows <= self.args.nb_shadows:
            try:
                self.shadow_datasets_idx = list(np.load(os.path.join(self.save_dir, f'shadow_datasets_idx__nb_challengers={nb_wanted_challengers}.npy')))
            except FileExistsError:
                raise Exception(f'To run LiRA with another number ({nb_wanted_challengers}) of challengers, choose another random seed')

    def create_shadow_datasets_idx(self, train_data_per_shadow):
        K = self.args.nb_shadows                    # number of shadow models
        n_per_shadow = train_data_per_shadow        # number of train data used per shadow model
        m            = len(self.idx_challengers)    # number of challenger data points to check for membership

        shadow_datasets_idx = [] # list of length K
        IN_shadow_models_per_challenger = np.array([False] * m * K).reshape(m, -1) # size m*K

        i_challengers = np.arange(m)

        for k in range(K//2): # create K shadow datasets, 2 per loop
            np.random.shuffle(i_challengers)

            shadow_datasets_idx.append(np.concatenate((
                np.random.choice(self.idx_train_data, n_per_shadow, replace=False), # use a subsample of the complete available train data for the shadow models
                self.idx_challengers[i_challengers[:m//2]]
            )))
            IN_shadow_models_per_challenger[i_challengers[:m//2], k*2] = True

            shadow_datasets_idx.append(np.concatenate((
                np.random.choice(self.idx_train_data, n_per_shadow, replace=False), # use a subsample of the complete available train data for the shadow models
                self.idx_challengers[i_challengers[m//2:]]
            )))
            IN_shadow_models_per_challenger[i_challengers[m//2:], k*2+1] = True

        IN_shadow_models = pd.Series(list(IN_shadow_models_per_challenger), index=self.idx_challengers, name='challenger_idx')

        return shadow_datasets_idx, IN_shadow_models


    def train_shadows(self):
        if self.nb_trained_shadows == self.args.nb_shadows:
            print('Shadow models already trained')
        else:
            nb_classes = self.datasetloader.nb_classes

            for k in range(self.nb_trained_shadows, self.args.nb_shadows):
                print(f'\n## Shadow n°{k}')
                shadow_save_dir = os.path.join(self.save_dir, 'shadows', f'{k:03}')
                mkdir(shadow_save_dir)

                idx_shadow_trainset = self.shadow_datasets_idx[k]

                ### Set up trainer and model
                trainer = TrainerForLiRA(self.args, shadow_save_dir, self.datasetloader,
                                         self.full_data, idx_shadow_trainset)
                model = models.__dict__[self.args.model](nb_classes)
                model = torch.nn.DataParallel(model)
                model.to(trainer.device)
                torch.backends.cudnn.benchmark = True
                print('Total params: %.2f' % (sum(p.numel() for p in model.parameters())))
                optimizer = optim.Adam(model.parameters())

                logger = trainer.logger

                ### Training
                for epoch in range(self.args.nb_epochs):
                    t0 = time.time()
                    train_loss, train_acc, train_acc5 = trainer.train(model, optimizer)
                    test_loss, test_acc, test_acc5 = trainer.test(model)
                    epoch_time = time.time() - t0
                    logger.append([train_loss, test_loss, train_acc, test_acc, train_acc5, test_acc5, epoch_time])
                    print(f'Epoch {epoch}: {train_acc = :.3f} | {test_acc = :.3f} | time = {epoch_time:.3f}s')

                ### Save model
                torch.save(model, os.path.join(shadow_save_dir, 'model.pt'))

                ### Visualize
                trainer.logger_plot()
                train_losses, test_losses = trainer.get_loss_distributions(model)
                plot_hist([train_losses, test_losses], ['train', 'val'], os.path.join(shadow_save_dir, 'hist_ep%d.png' % epoch))
            
                write_yaml({'nb_trained_shadows':k+1}, os.path.join(self.save_dir, 'shadows', 'training_progress.yml'))


    def load_shadows(self):
        self.shadow_models = []

        for k in range(self.args.nb_shadows):
            shadow_path = os.path.join(self.save_dir, 'shadows', f'{k:03}', f'model.pt')
            shadow = torch.load(shadow_path).to(self.device)
            self.shadow_models.append(shadow)
    
    def get_challenger_features_and_labels(self):
        """The function to set the dataset parameters"""
        ## Retrieve challenger features and labels
        tmp_dataloader = DataLoader(Subset(self.full_data, indices=self.idx_challengers),
                                    batch_size=len(self.idx_challengers),
                                    generator=torch.Generator(device=self.device))
        for features, labels in iter(tmp_dataloader): # this loops just once because of the batch_size above
            return features, labels

    def compute_confs(self):
        """ Confidence scores in the IN/OUT shadow models """
        confs_shadow = [] # confidences scores given for the truth label for every instance for every shadow model
        m = len(self.idx_challengers)

        features, labels = self.get_challenger_features_and_labels()
        features = features.to(self.device)

        for shadow in self.shadow_models:
            shadow.eval()
            with torch.no_grad():
                pred_scores = shadow(features)
            
            pred_scores = torch.softmax(pred_scores, 1).cpu().numpy() # softmax rescaling

            before_logit = pred_scores[np.arange(len(pred_scores)), np.array(labels.cpu())][:, None] # take the y^th coordinate corresponding to the ground-truth
            sum_without_y_coord = pred_scores
            sum_without_y_coord[np.arange(len(sum_without_y_coord)), np.array(labels.cpu())] = 0
            sum_without_y_coord = np.sum(sum_without_y_coord, axis=1)[:, None]

            confs_shadow.append(
                np.log(before_logit +LiRA.log_epsilon) - np.log(sum_without_y_coord +LiRA.log_epsilon)
            )

        confs_shadow = np.concatenate(confs_shadow, axis=1) # size m*K

        IN_shadow_models = pd.read_pickle(os.path.join(self.save_dir, f'IN_shadow_models__nb_challengers={m}.pkl'))
        # Sanity check
        assert np.array_equiv(self.idx_challengers, IN_shadow_models.index), 'Should be equal xD'

        confs_IN = confs_shadow[np.stack(IN_shadow_models.values)].reshape(m, -1) # for every instance, the confidence values in the IN models
        confs_OUT = confs_shadow[~np.stack(IN_shadow_models.values)].reshape(m, -1) # for every instance, the confidence values in the OUT models

        mean_IN = confs_IN.mean(axis=1) # for every instance, the mean value of the confidence values in the IN models
        std_IN  = confs_IN.std(axis=1)
        mean_OUT = confs_OUT.mean(axis=1)
        std_OUT  = confs_OUT.std(axis=1)

        df_IN_OUT_confs = pd.DataFrame({
            'confs_IN' : pd.Series(list(confs_IN)),
            'confs_OUT' : pd.Series(list(confs_OUT)),
            'mean_IN' : mean_IN,
            'mean_OUT' : mean_OUT,
            'std_IN' : std_IN,
            'std_OUT' : std_OUT
        })
        df_IN_OUT_confs.index = self.idx_challengers

        df_IN_OUT_confs.to_pickle(os.path.join(self.save_dir, f'IN_OUT_confs__nb_challengers={m}.pkl'))

    


#############################################################################################################
# main function
#############################################################################################################
def main():
    args, save_dir = check_args(parse_arguments())

    lira = LiRA(args, save_dir)
    lira.train_shadows()
    lira.load_shadows()
    lira.compute_confs()


if __name__ == '__main__':
    main()
