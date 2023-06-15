import os
import sys
import argparse
import shutil
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_DIR, '../'))

from utils.helper import mkdir, load_yaml, savefig
from utils.datasets.loader import DatasetLoader
from attacks.lira import LiRA

#############################################################################################################
# get and save the arguments
#############################################################################################################
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model architecture', default='resnet18')
    parser.add_argument('--dataset', type=str, help='dataset name', default='CIFAR10')
    parser.add_argument('--random_seed', '-s', type=int, help='random seed', default=123)
    parser.add_argument('--defence', type=str, help='defence name (if rtt, need to add the subfolder for the temperature distribution: e.g. rtt/beta_0.2-10)')
                        # choices=['vanilla', 'dpsgd', 'advreg', 'relaxloss', 'label_smoothing', 'rtt']
    parser.add_argument('--attacks', nargs='+', type=str, help='list of attacks to run on the target model (if multiple attacks, seperated with a blank space)',
                        choices=['all', 'entropy', 'Mentropy', 'MAST', 'LiRA',
                                 'MAST_label_smoothing', 'LiRA_label_smoothing'])    
    return parser


def check_args(parser):
    '''check and store the arguments as well as set up the save_dir'''
    args = parser.parse_args()

    ## set up save_dir
    defence_path = os.path.join(FILE_DIR, '../results', args.model, args.dataset, f'seed{args.random_seed}', 'defences', args.defence)
    if not os.path.exists(defence_path):
        raise FileExistsError(f'No defence at {defence_path}')
    save_dir = os.path.join(defence_path, 'attacks')
    mkdir(save_dir)
    attacks_dir = os.path.join(FILE_DIR, '../results', args.model, args.dataset, f'seed{args.random_seed}', 'attacks')

    ## store this script
    shutil.copy(os.path.realpath(__file__), save_dir)

    return args, save_dir, attacks_dir


#############################################################################################################
# helper functions
#############################################################################################################
class Attacks():
    attacks_name = {
        'entropy' : 'entropy (Song et al.)',
        'Mentropy' : 'Mentropy (Song et al.)',
        'MAST' : 'MAST (Sablayrolles et al.)',
        'LiRA' : 'LiRA (Carlini et al.)',
        'MAST_label_smoothing' : 'MAST_label_smoothing',
        'LiRA_label_smoothing' : 'LiRA_label_smoothing',
    }

    # Colors palette to choose from to keep the same color for a given attack 
    __possible_colors = list(matplotlib.colors.TABLEAU_COLORS.values())
    colors = {}
    for i, attack in enumerate(attacks_name.keys()):
        colors[attack] = __possible_colors[i % len(__possible_colors)]
    

    def __init__(self, args, save_dir, attacks_dir):
        self.args = args
        self.save_dir = save_dir
        self.attacks_dir = attacks_dir
        self.set_cuda_device()
        self.set_model_outputs()

    def set_cuda_device(self):
        """The function to set CUDA device."""
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.device = torch.device("cuda" if use_cuda else "cpu")

    def set_model_outputs(self):    
        target_model_dir = os.path.join(self.save_dir, '..')

        target_model = torch.load(os.path.join(target_model_dir, 'model.pt')).to(self.device)
        target_model.eval()
        
        batchsize = load_yaml(os.path.join(target_model_dir, 'params.yml'))['batchsize']
        datasetloader = DatasetLoader(self.args.dataset, self.args.random_seed, batchsize)

        challenger_loader = datasetloader.get_challengers_loader()
        
        features, labels, logits, losses = [], [], [], []
        with torch.no_grad():
            for fea, lab in challenger_loader:
                fea, lab = fea.to(self.device), lab.to(self.device)
                features.append( fea )
                labels.append( lab )
                outputs = target_model(fea)
                logits.append( outputs )
                losses.append( nn.CrossEntropyLoss(reduction='none')(outputs, lab) )
            
        features = torch.concat(features)
        labels   = torch.concat(labels)
        logits   = torch.concat(logits)
        losses   = torch.concat(losses)
        probits  = torch.softmax(logits, 1).cpu()
        
        nb_challengers = len(features)
        mem_or_nmem = np.concatenate([np.ones(shape=nb_challengers//2, dtype=bool), np.zeros(shape=nb_challengers//2, dtype=bool)])
        pred_labels = probits.argmax(dim=1)

        # Model outputs
        self.labels = labels.cpu().numpy()
        self.features = features.cpu().numpy()
        self.mem_or_nmem = mem_or_nmem
        self.pred_labels = pred_labels.numpy()
        self.logits = logits.cpu().numpy()
        self.losses = losses.cpu().numpy()
        self.probits = probits.numpy()

        self.__save_challengers_infos()

    def __save_challengers_infos(self):
        challengers_save_dir = os.path.join(self.save_dir, 'challengers')
        mkdir(challengers_save_dir)

        np.save( os.path.join(challengers_save_dir, 'labels.npy'), self.labels )
        np.save( os.path.join(challengers_save_dir, 'pred_labels.npy'), self.pred_labels )
        np.save( os.path.join(challengers_save_dir, 'losses.npy'), self.losses )
        np.save( os.path.join(challengers_save_dir, 'logits.npy'), self.logits )
        np.save( os.path.join(challengers_save_dir, 'probits.npy'), self.probits )
        np.save( os.path.join(challengers_save_dir, 'mem_or_nmem.npy'), self.mem_or_nmem )


    def plot_ROC_curves(self, xmin :float =None):
        print('Creating ROC curves...')
        attacks, list_fprs, list_tprs = self.get_fpr_tpr()

        self.save_metrics(attacks, list_fprs, list_tprs)
        
        plt.clf()
        fig, axs = plt.subplots(1, 2, dpi=150, figsize=(10, 4))
        for i, log_scale in enumerate([False, True]):
            ax = axs[i]
            for attack, fprs, tprs in zip(attacks, list_fprs, list_tprs):
                ax.plot(
                    fprs, tprs,
                    label=f'{Attacks.attacks_name[attack]} (auc={metrics.auc(fprs, tprs):.3f})',
                    color=Attacks.colors.get(attack, 'black')
                )
            ax.axline((0, 0), slope=1, c='#333', ls='--', lw=.5)
            ax.set_xlabel('FPR')
            ax.set_ylabel('TPR')

            if log_scale:
                ax.set_xscale('log')
                ax.set_yscale('log')
                if xmin is not None:
                    ax.set_xlim(xmin, 1)
                ax.set_title('log-ROC curve', fontsize=10)
            else:
                ax.set_title('ROC curve', fontsize=10)
        plt.suptitle(f'{self.args.defence}')
        plt.legend(title='Attacks:', loc='lower right', fontsize=9)
        savefig( os.path.join(self.save_dir, 'ROC.png') )


    def save_metrics(self, attacks, list_fprs, list_tprs):
        for attack, fprs, tprs in zip(attacks, list_fprs, list_tprs):
            file_name = os.path.join(self.save_dir, 'metrics', f'{attack}.txt')
            if os.path.exists(file_name):
                print(f'  Metrics already saved for {attack}')
            else:
                mkdir(os.path.join(self.save_dir, 'metrics'))
                m = pd.Series({
                        'AUC' : metrics.auc(fprs, tprs),
                        'TPR@0.1FPR' : np.interp(.1, fprs, tprs),
                        'TPR@0.01FPR' : np.interp(.01, fprs, tprs)
                    }, name=attack
                )
                m.to_csv(file_name, sep='\t')


    def get_fpr_tpr(self):
        """Return FPR and TPR for the attacks performed on the given target model"""
        attacks = self.args.attacks if not 'all' in self.args.attacks else list(Attacks.attacks_name.keys())

        list_fprs = []
        list_tprs = []

        scores = self.get_scores(attacks)
        for attack, membership_scores in scores.items():
            file_name = os.path.join(self.save_dir, 'fpr_tpr', f'{attack}.npy')
            if os.path.exists(file_name):
                print(f'  FPR/TPR already exists for {attack}')
                fprs_tprs = np.load(file_name)
                fprs, tprs = fprs_tprs[:, 0], fprs_tprs[:, 1]
            else:
                mkdir(os.path.join(self.save_dir, 'fpr_tpr'))
                fprs, tprs, _ = metrics.roc_curve(self.mem_or_nmem, membership_scores)
                np.save(file_name, np.concatenate([fprs[:, None], tprs[:, None]], axis=1))
            list_fprs.append(fprs)
            list_tprs.append(tprs)

        return scores.keys(), list_fprs, list_tprs


    def get_scores(self, attacks):
        """Membership scores for different attacks, the smaller the score of a challenger the more likely it is a member"""
        scores = {}
        if 'entropy' in attacks  : scores['entropy']  = self.__get_entropy_scores()
        if 'Mentropy' in attacks : scores['Mentropy'] = self.__get_Mentropy_scores()
        if 'MAST' in attacks     : scores['MAST']     = self.__get_mast_scores()
        if 'LiRA' in attacks     : scores['LiRA']     = self.__get_lira_scores()
        if 'MAST_label_smoothing' in attacks:
            s = self.__get_mast_scores(shadows_defence='_label_smoothing') # returns None if can't be computed
            if s is not None:
                scores['MAST_label_smoothing'] = s
        if 'LiRA_label_smoothing' in attacks:
            s = self.__get_lira_scores(shadows_defence='_label_smoothing') # returns None if can't be computed
            if s is not None:
                scores['LiRA_label_smoothing'] = s
        return scores
    
    def __get_entropy_scores(self):
        """compute the entropy of the prediction"""
        _log_value = lambda probits: -np.log(np.maximum(probits, 1e-30))

        return -np.sum(np.multiply(self.probits, _log_value(self.probits)), axis=1)
    
    def __get_Mentropy_scores(self):
        """-(1-f(x)_y) log(f(x)_y) - \sum_i f(x)_i log(1-f(x)_i)"""
        _log_value = lambda probits: -np.log(np.maximum(probits, 1e-30))

        log_probs = _log_value(self.probits)
        reverse_probs = 1 - self.probits
        log_reverse_probs = _log_value(reverse_probs)
        modified_probs = np.copy(self.probits)
        modified_probs[range(len(self.labels)), self.labels] = reverse_probs[range(len(self.labels)), self.labels]
        modified_log_probs = np.copy(log_reverse_probs)
        modified_log_probs[range(len(self.labels)), self.labels] = log_probs[range(len(self.labels)), self.labels]
        return -np.sum(np.multiply(modified_probs, modified_log_probs), axis=1)
    
    # def __get_LSattack_scores(self):
    #     """Attack specific for the label smoothing defence, computing KL divergence between expected ditribution and observed probit distribution"""
    #     kl_divergence = lambda p, q: np.sum(np.where(p != 0, p * np.log(p / q), 0))
    
    #     eps = .1
    #     nb_classes = len(set(self.labels))
    #     gt = eps * np.ones(nb_classes) / nb_classes
    #     gt_per_label = {}
    #     for label in set(self.labels):
    #         tmp_gt = gt.copy()
    #         tmp_gt[label] = 1 + eps * (1 - nb_classes) / nb_classes
    #         gt_per_label[label] = tmp_gt
        

    #     scores = []
    #     for l, p in zip(self.labels, self.probits):
    #         scores.append( kl_divergence(gt_per_label[l], p) )

    #     return -np.array(scores)
    
    def __get_mast_scores(self, shadows_defence =''):
        """Compute the MAST scores"""
        mast_dir = os.path.join(self.attacks_dir, f'mast{shadows_defence}')
        
        if os.path.exists(mast_dir): # can't compute scores if MAST was not trained in these settings
            nb_challengers = len(self.probits)
            df_IN_OUT_losses = pd.read_pickle(os.path.join(mast_dir, f'IN_OUT_losses__nb_challengers={nb_challengers}.pkl'))

            # Compute the statistic (for every instance)
            scores_lossesOUT = []
            for l, mOUT in zip(
                    self.losses,
                    df_IN_OUT_losses['mean_OUT'].values
                ):
                scores_lossesOUT.append( -l + mOUT)
            return np.array(scores_lossesOUT)
    
    def __get_lira_scores(self, shadows_defence =''):
        """Compute the LiRA scores"""

        lira_dir = os.path.join(self.attacks_dir, f'lira{shadows_defence}')

        if os.path.exists(lira_dir): # can't compute scores if LiRA was not trained in these settings
            nb_challengers = len(self.probits)
            df_IN_OUT_confs = pd.read_pickle(os.path.join(lira_dir, f'IN_OUT_confs__nb_challengers={nb_challengers}.pkl'))

            before_logit = self.probits[np.arange(nb_challengers), self.labels] # take the y^th coordinate corresponding to the ground-truth

            log_epsilon = LiRA.log_epsilon

            sum_without_y_coord = self.probits.copy()
            sum_without_y_coord[np.arange(len(sum_without_y_coord)), self.labels] = 0
            sum_without_y_coord = np.sum(sum_without_y_coord, axis=1)
            confs_target = np.log(before_logit +log_epsilon) - np.log(sum_without_y_coord +log_epsilon)

            # Compute the statistic (for every instance)
            membership_scores = []
            for c, mIN, sIN, mOUT, sOUT in zip(
                    confs_target,
                    df_IN_OUT_confs['mean_IN'].values, df_IN_OUT_confs['std_IN'].values,
                    df_IN_OUT_confs['mean_OUT'].values, df_IN_OUT_confs['std_OUT'].values
                ):
                p_IN  = stats.norm.logpdf(c, mIN, sIN +log_epsilon)
                p_OUT = stats.norm.logpdf(c, mOUT, sOUT +log_epsilon)
                membership_scores.append(p_IN - p_OUT)

            return np.array(membership_scores)


#############################################################################################################
# main function
#############################################################################################################
def main():
    args, save_dir, attacks_dir = check_args(parse_arguments())

    attacks = Attacks(args, save_dir, attacks_dir)
    attacks.plot_ROC_curves(xmin=3e-3)


if __name__ == '__main__':
    main()