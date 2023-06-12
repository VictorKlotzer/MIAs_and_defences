import os
import sys
import argparse
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_DIR, '../'))

from utils.helper import mkdir, load_yaml
from utils.datasets.loader import DatasetLoader

"""
This .py file is just saving the losses of the challengers when passing them through the LiRA shadow models
"""


#############################################################################################################
# get and save the arguments
#############################################################################################################
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model architecture', default='resnet18')
    parser.add_argument('--dataset', type=str, help='dataset name', default='CIFAR10')
    parser.add_argument('--random_seed', '-s', type=int, default=123, help='random seed')
    # parser.add_argument('--nb_shadows', type=int, default=64, help='number of shadow models to use per challenger')
    # parser.add_argument('--nb_challengers', type=int, default=5_000, help='number of challengers (members and non-members) to consider')

    parser.add_argument('--shadows_defence', type=str,
                        help='defence used for the shadow models (if rtt, need to add the subfolder for the temperature distribution: e.g. rtt/beta_0.2-10)',
                        default='')
                        # choices=['' "for vanilla", 'dpsgd', 'advreg', 'relaxloss', 'label_smoothing', 'rtt']
    
    return parser


def check_args(parser):
    '''check and store the arguments as well as set up the save_dir'''
    args = parser.parse_args()

    ## Set up save_dir
    save_dir = os.path.join(FILE_DIR, '../results', args.model, args.dataset, f'seed{args.random_seed}', 'attacks', f'mast_{args.shadows_defence}')
    mkdir(save_dir)
    shadows_dir = os.path.join(FILE_DIR, '../results', args.model, args.dataset, f'seed{args.random_seed}', 'attacks', f'lira_{args.shadows_defence}', 'shadows')

    ## Check that LiRA shadow models were trained
    if not os.path.exists(shadows_dir):
        raise NotADirectoryError('LiRA shadow models first needs to be trained')
    else:
        nb_trained_shadows = load_yaml(os.path.join(shadows_dir, 'training_progress.yml'))['nb_trained_shadows']
        nb_totrain_shadows = load_yaml(os.path.join(shadows_dir, '../params.yml'))['nb_shadows']
        if nb_trained_shadows != nb_totrain_shadows:
            raise Exception(f'Not all LiRA shadow models have been trained ({nb_trained_shadows} out of {nb_totrain_shadows}), you first need to complete their training')

    ## Copy the configuration from LiRA
    shutil.copy(os.path.join(shadows_dir, '../params.yml'), save_dir)

    configs = load_yaml(os.path.join(save_dir, 'params.yml'))
    parser.set_defaults(**configs)
    args = parser.parse_args()
    if None in [z[1] for z in args._get_kwargs()]:
        raise Exception(f'No argument can have the value None: {args}')
    
    print(f'Using {configs["nb_shadows"]} shadow models and {configs["nb_challengers"]:_} challengers')

    ## Store this script
    shutil.copy(os.path.realpath(__file__), save_dir)

    return args, save_dir, shadows_dir


#############################################################################################################
# helper functions
#############################################################################################################

class MAST():
    """ MAST attack using the shadow models trained for LiRA """
    
    def __init__(self, args, save_dir, shadows_dir):
        self.args = args
        self.save_dir = save_dir
        self.shadows_dir = shadows_dir
        self.set_cuda_device()

    def set_cuda_device(self):
        """The function to set CUDA device."""
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.device = torch.device("cuda" if self.use_cuda else "cpu")


    def load_shadows(self):
        self.shadow_models = []
        for k in range(self.args.nb_shadows):
            shadow_path = os.path.join(self.shadows_dir, f'{k:03}', f'model.pt')
            shadow = torch.load(shadow_path).to(self.device)
            self.shadow_models.append(shadow)
    
    def get_challenger_features_and_labels(self):
        """The function to set the dataset parameters"""
        datasetloader = DatasetLoader(self.args.dataset, self.args.random_seed, self.args.batchsize)
        challenger_dataset = datasetloader.chal_set

        nb_challengers = len(challenger_dataset)
        nb_wanted_challengers = self.args.nb_challengers # number of wanted challengers

        idx_challengers = np.arange(nb_challengers).astype(int)
        self.idx_challengers = np.concatenate([
            idx_challengers[:nb_challengers//2][:nb_wanted_challengers//2], # members
            idx_challengers[nb_challengers//2:][:nb_wanted_challengers//2]  # non-members
        ])
        
        ## Retrieve challenger features and labels
        tmp_dataloader = DataLoader(Subset(challenger_dataset, indices=self.idx_challengers),
                                    batch_size=nb_wanted_challengers,
                                    generator=torch.Generator(device=self.device))
        for features, labels in iter(tmp_dataloader): # this loops just once because of the batch_size above
            return features, labels

    def get_losses(self):
        """ Losses of the challengers in the IN/OUT shadow models """
        losses = [] # losses for every instance for every shadow model
        m = self.args.nb_challengers

        features, labels = self.get_challenger_features_and_labels()
        features = features.to(self.device)

        for shadow in self.shadow_models:
            shadow.eval()
            with torch.no_grad():
                outputs = shadow(features)
                losses.append( nn.CrossEntropyLoss(reduction='none')(outputs, labels).cpu().numpy()[:, None] )
            
        losses = np.concatenate(losses, axis=1) # size m*K

        IN_shadow_models = pd.read_pickle(os.path.join(self.shadows_dir, f'../IN_shadow_models__nb_challengers={m}.pkl'))
        # Sanity check
        assert np.array_equiv(self.idx_challengers, IN_shadow_models.index - IN_shadow_models.index[0]), 'Should be equal xD'

        losses_IN  = losses[np.stack(IN_shadow_models.values)].reshape(m, -1) # for every instance, the confidence values in the IN models
        losses_OUT = losses[~np.stack(IN_shadow_models.values)].reshape(m, -1) # for every instance, the confidence values in the OUT models

        mean_IN = losses_IN.mean(axis=1) # for every instance, the mean value of the confidence values in the IN models
        mean_OUT = losses_OUT.mean(axis=1)
        df_IN_OUT_losses = pd.DataFrame({
            'losses_IN' : pd.Series(list(losses_IN)),
            'losses_OUT' : pd.Series(list(losses_OUT)),
            'mean_IN' : mean_IN,
            'mean_OUT' : mean_OUT,
        })
        df_IN_OUT_losses.index = self.idx_challengers

        df_IN_OUT_losses.to_pickle(os.path.join(self.save_dir, f'IN_OUT_losses__nb_challengers={m}.pkl'))

    


#############################################################################################################
# main function
#############################################################################################################
def main():
    args, save_dir, shadows_dir = check_args(parse_arguments())

    mast = MAST(args, save_dir, shadows_dir)
    mast.load_shadows()
    mast.get_losses()


if __name__ == '__main__':
    main()