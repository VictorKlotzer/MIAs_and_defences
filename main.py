import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics

from utils.helper import mkdir, savefig, load_yaml

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

available_defences = ['vanilla', 'dpsgd', 'label_smoothing', 'advreg', 'relaxloss', 'rtt', 'ttltt']
available_attacks  = ['entropy', 'Mentropy', 'MAST', 'LiRA',
                      'LSattack',
                      'MAST_label_smoothing', 'LiRA_label_smoothing']
available_models   = ['mlp1', 'resnet18', 'vgg11', 'vgg11_bn']
available_datasets = ['CIFAR2', 'CIFAR10', 'CIFAR100']

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model architecture', default='resnet18',
                        choices=available_models)
    parser.add_argument('--dataset', type=str, help='dataset name', default='CIFAR10',
                        choices=available_datasets)
    parser.add_argument('--random_seed', '-s', type=int, default=123, help='random seed')

    parser.add_argument('--defences', nargs='+', type=str, help='defences to use for the target model (if list, seperate defences with a blank space)',
                        default='all',
                        choices=['all'] + available_defences)
    parser.add_argument('--attacks', nargs='+', type=str, help='attacks to run on the target models (if list, seperate attacks with a blank space)',
                        default='all',
                        choices=['all'] + available_attacks)
    
    parser.add_argument('--mode', type=str, help='mode of the process to be run', default='full_pipeline',
                        choices=['full_pipeline', 'defences', 'attacks', 'comparisons'])
    return parser


def check_args(parser):
    '''check and store the arguments as well as retrieve the results_dir'''
    args = parser.parse_args()
    if None in [z[1] for z in args._get_kwargs()]:
        raise Exception(f'No argument can have the value None: {args}')

    results_dir = os.path.join(FILE_DIR, 'results', args.model, args.dataset, f'seed{args.random_seed}')
    mkdir(results_dir)

    ## load configs for specific model and dataset
    default_file = os.path.join(FILE_DIR, 'defences', 'configs', args.dataset, args.model, 'default.yml')
    if not os.path.exists(default_file): default_file = os.path.join(FILE_DIR, 'defences', 'configs', 'default.yml')
    default_configs = load_yaml(default_file)
    parser.set_defaults(**default_configs)
    args = parser.parse_args()

    print(
    f'''
    #  model:       {args.model}
    #  dataset:     {args.dataset}
    #  random seed: {args.random_seed}
    #  mode:        {args.mode}
    #  defences:    {args.defences if not args.defences == 'all' else available_defences}
    #  attacks:     {args.attacks if not args.attacks == 'all' else available_attacks}
    '''
    )

    return args, results_dir


def run_defences(args, results_dir):
    defences = args.defences if not args.defences == 'all' else available_defences
    for defence in defences:
        print(f'\n##  Defence {defence}')
        torch.cuda.empty_cache()

        defence_done_file = os.path.join(results_dir, defence, 'done.txt')
        if os.path.exists(defence_done_file):
            print(f'Already done (remove {defence_done_file} file if you want to retrain)')
        else:
            command = f'python defences/{defence}.py'
            command += f' --model {args.model} --dataset {args.dataset} -s {args.random_seed}'
            if defence == 'rtt':
                print(f'> Temperatures unif_{args.temperatures_min}-{args.temperatures_max}')
                status = os.system(command + f' -Tdist unif -Tmin {args.temperatures_min} -Tmax {args.temperatures_max}')
                print(f'\n> Temperatures beta_{args.temperatures_min}-{args.temperatures_max}')
                status2 = os.system(command + f' -Tdist beta -Tmin {args.temperatures_min} -Tmax {args.temperatures_max}')
                status += status2
            else:
                status = os.system(command)
            if status == 0:
                open(defence_done_file, 'w').close()


def run_attacks(args, results_dir):
    ## Train LiRA
    print(f'\n##  Train LiRA')
    torch.cuda.empty_cache()
    done_file = os.path.join(results_dir, 'attacks', 'lira', 'done.txt')
    if os.path.exists(done_file):
        print(f'Already done (remove {done_file} file if you want to retrain)')
    else:
        command = f'python attacks/lira.py'
        command += f' --model {args.model} --dataset {args.dataset} -s {args.random_seed}'
        status = os.system(command)
        if status == 0:
            open(done_file, 'w').close()

    ## Prepare MAST
    print(f'\n##  Prepare MAST')
    torch.cuda.empty_cache()
    done_file = os.path.join(results_dir, 'attacks', 'mast', 'done.txt')
    if os.path.exists(done_file):
        print(f'Already done (remove {done_file} file if you want to retrain)')
    else:
        command = f'python attacks/mast.py'
        command += f' --model {args.model} --dataset {args.dataset} -s {args.random_seed}'
        status = os.system(command)
        if status == 0:
            open(done_file, 'w').close()

    ## Run attacks
    defences = args.defences if not args.defences == 'all' else available_defences
    for defence in defences:
        print(f'\n##  Attack {defence}')
        torch.cuda.empty_cache()
        
        done_file = os.path.join(results_dir, f'{defence}', 'attacks', 'done.txt')
        if defence == 'rtt': # to run attacks for all trained rtt versions
            rtt_versions = []
            rtt_dir = os.path.join(results_dir, 'rtt')
            for rtt_ver in [d for d in os.listdir(rtt_dir) if os.path.isdir(os.path.join(rtt_dir, d))]:
                rtt_versions.append(rtt_ver)
            
            done_file = os.path.join(results_dir, f'rtt/{rtt_versions[-1]}', 'attacks', 'done.txt')
        
        if os.path.exists(done_file):
            print(f'Already done (remove {done_file} file if you want to re-attack)')
        else:
            command = f'python attacks/run_attacks.py'
            command += f' --model {args.model} --dataset {args.dataset} -s {args.random_seed}'
            command += f' --attacks {"all" if args.attacks == "all" else " ".join(args.attacks)}'
            command += f' --defence {defence}'
            if defence == 'rtt':
                status = 0
                for rtt_ver in rtt_versions:
                    print(f'> Temperatures {rtt_ver}')
                    status_rtt = os.system(command + f'/{rtt_ver}')
                    status += status_rtt
            else:
                status = os.system(command)
            if status == 0:
                open(done_file, 'w').close()


def __ROC_curves_per_attack(attack, defences, xmin =None):
    defences_copy = defences.copy()
    list_fprs = []
    list_tprs = []

    defences_to_remove = []
    for defence in defences_copy:
        try:
            fprs_tprs = np.load(os.path.join(results_dir, defence, 'attacks', f'fpr_tpr_{attack}.npy'))
            list_fprs.append(fprs_tprs[:, 0])
            list_tprs.append(fprs_tprs[:, 1])
        except:
            defences_to_remove.append(defence)
    for defence in defences_to_remove:
        defences_copy.remove(defence)

    if len(defences_copy) == 0: 
        return attack # this attack wasn't performed on any model so return it

    plt.clf()
    fig, axs = plt.subplots(1, 2, dpi=150, figsize=(10, 4))
    for i, log_scale in enumerate([False, True]):
        ax = axs[i]
        for defence, fprs, tprs in zip(defences_copy, list_fprs, list_tprs):
            ax.plot(
                fprs, tprs,
                label=f'{defence} (auc={metrics.auc(fprs, tprs):.3f})',
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
    plt.suptitle(f'{attack}')
    plt.legend(title='Defences:', loc='lower right', fontsize=8)
    savefig( os.path.join(results_dir, f'ROC_{attack}.png') )


def __plot_challengers_curves(defences, results_dir):
    ###########################
    ## Losses
    plt.clf()
    n_col = 4
    n_row = (len(defences)-1)//n_col + 1
    fig, axs = plt.subplots(n_row, n_col, dpi=150, figsize=(n_col*3, n_row*3))
    plt.subplots_adjust(hspace=.5)

    for i, defence in enumerate(defences):
        losses      = np.load( os.path.join(results_dir, defence, 'attacks', 'challengers', 'losses.npy') )
        mem_or_nmem = np.load( os.path.join(results_dir, defence, 'attacks', 'challengers', 'mem_or_nmem.npy') )
        loss_mem  = losses[mem_or_nmem]
        loss_nmem = losses[~mem_or_nmem]

        ax = axs[i//n_col, i%n_col]
        ax.hist(loss_mem, bins=30, histtype='step', label='members')
        ax.hist(loss_nmem, bins=30, histtype='step', label='non-members')
        ax.set_yscale('log')
        ax.set_title(defence, fontsize=10)
        ax.legend(fontsize=9)
    
    i += 1
    while i < n_col*n_row:
        axs[i//n_col, i%n_col].axis('off')
        i += 1

    plt.suptitle(f'Challenger losses for model {args.model} on dataset {args.dataset}')
    savefig( os.path.join(results_dir, 'challengers_losses.png') )

    ###########################
    ## Logits
    plt.clf()
    n_col = 4
    n_row = (len(defences)-1)//n_col + 1
    fig, axs = plt.subplots(n_row, n_col, dpi=150, figsize=(n_col*3, n_row*3))
    plt.subplots_adjust(hspace=.5)

    for i, defence in enumerate(defences):
        logits      = np.load( os.path.join(results_dir, defence, 'attacks', 'challengers', 'logits.npy') )
        mem_or_nmem = np.load( os.path.join(results_dir, defence, 'attacks', 'challengers', 'mem_or_nmem.npy') )
        logits_mem  = logits[mem_or_nmem].max(axis=-1)
        logits_nmem = logits[~mem_or_nmem].max(axis=-1)

        ax = axs[i//n_col, i%n_col]
        ax.hist(logits_mem, bins=30, histtype='step', label='members')
        ax.hist(logits_nmem, bins=30, histtype='step', label='non-members')
        ax.set_yscale('log')
        ax.set_title(defence, fontsize=10)
        ax.legend(fontsize=9)
    
    i += 1
    while i < n_col*n_row:
        axs[i//n_col, i%n_col].axis('off')
        i += 1

    plt.suptitle(f'Challenger logits for model {args.model} on dataset {args.dataset}')
    savefig( os.path.join(results_dir, 'challengers_logits.png') )

    ###########################
    ## Probits
    plt.clf()
    n_col = 4
    n_row = (len(defences)-1)//n_col + 1
    fig, axs = plt.subplots(n_row, n_col, dpi=150, figsize=(n_col*3, n_row*3))
    plt.subplots_adjust(hspace=.5)

    for i, defence in enumerate(defences):
        probits      = np.load( os.path.join(results_dir, defence, 'attacks', 'challengers', 'probits.npy') )
        labels      = np.load( os.path.join(results_dir, defence, 'attacks', 'challengers', 'labels.npy') )
        mem_or_nmem = np.load( os.path.join(results_dir, defence, 'attacks', 'challengers', 'mem_or_nmem.npy') )
        
        gt_probits = probits[np.arange(len(labels)), labels]
        probits_mem  = gt_probits[mem_or_nmem]
        probits_nmem = gt_probits[~mem_or_nmem]

        ax = axs[i//n_col, i%n_col]
        bins = np.linspace(0, 1, 50)
        ax.hist(probits_mem, bins=bins, histtype='step', label='members')
        ax.hist(probits_nmem, bins=bins, histtype='step', label='non-members')
        ax.set_yscale('log')
        ax.set_title(defence, fontsize=10)
        ax.legend(fontsize=9)
    
    i += 1
    while i < n_col*n_row:
        axs[i//n_col, i%n_col].axis('off')
        i += 1

    plt.suptitle(f'Challenger probits for model {args.model} on dataset {args.dataset}')
    savefig( os.path.join(results_dir, 'challengers_probits.png') )

    ###########################
    ## 

    # for i, ver in enumerate(df.index):
    #     # Confidence values
    #     x = np.concatenate([df.loc[ver, 'mem'], df.loc[ver, 'nmem']])

    #     # Compute the differences |x_i - x_j|
    #     x_rep = x[None, :].repeat(len(x), axis=0)
    #     x_diff = x_rep - x_rep.transpose(1, 0, 2)
    #     x_dist = np.tril((x_diff**2).sum(axis=-1), -1)

    #     # Compute intra- and inter-class distances
    #     intra_class_dists = x_dist[mask_labels]
    #     inter_class_dists = x_dist[~mask_labels & np.tril(np.ones((len(labels), len(labels)), dtype=bool), -1)]

    #     # Plot them
    #     ax = plt.subplot(3, 4, i+1)
    #     bins = np.linspace(min(intra_class_dists), max(inter_class_dists), 100)
    #     ax.hist(intra_class_dists, bins=bins, histtype='step', density=True, label='intra-class distance')
    #     ax.hist(inter_class_dists, bins=bins, histtype='step', density=True, label='inter-class distance')

    #     # Add gamma distribution approximations
    #     theta = lambda z: (z * np.log(z)).mean() - z.mean() * (np.log(z)).mean()
    #     k = lambda z: z.mean() / theta(z)

    #     x = np.linspace(0, max(inter_class_dists), 1000)
    #     ka, ta = k(intra_class_dists), theta(intra_class_dists)
    #     ke, te = k(inter_class_dists), theta(inter_class_dists)
    #     ax.plot(x, stats.gamma.pdf(x, a=ka, scale=ta), color='#444', linestyle='--')
    #     ax.plot(x, stats.gamma.pdf(x, a=ke, scale=te), color='#444', linestyle='--')

    #     # Compute interesting area
    #     vv = np.concatenate([intra_class_dists, inter_class_dists])
    #     m, s = np.mean(vv), np.std(vv)
    #     bins = np.linspace(min(intra_class_dists), m+3*s, 1000)
    #     a1, b1 = np.histogram(intra_class_dists, bins=bins, density=True)
    #     a2, b2 = np.histogram(inter_class_dists, bins=bins, density=True)

    #     # kl_divergence = lambda p, q: np.sum(np.where(p != 0, p * np.log(p / q), 0))
    #     kl_gamma = lambda k1, t1, k2, t2: (k1 - k2)*psi(k1) - np.log(gamma(k1)) + np.log(gamma(k2)) + k2 * (np.log(t2) - np.log(t1)) + k1 * (t1 - t2) / t2
    #     ax.set_title(
    #         r"$\bf{" + ver.replace('_', '~') + "}$"
    #         f'\n  intersecting area: {sum(np.minimum(a1, a2) * (b1[-1] - b1[0])/len(a1)):.3f}'
    #         # f'\n  KL: {kl_divergence(np.random.choice(inter_class_dists, size=len(intra_class_dists), replace=False), intra_class_dists):_.0f}'
    #         f'\n  KL gamma: {kl_gamma(ka, ta, ke, te):.3f}',
    #         loc='left', fontsize=11
    #     )
    #     ax.legend()


def compare_attacks_vs_defences(args, results_dir):
    """Attacks vs Defences comparisons"""
    print(f'\n##  Attacks vs Defences comparisons')
    defences = args.defences if not args.defences == 'all' else available_defences
    attacks  = args.attacks if not args.attacks == 'all' else available_attacks

    if 'rtt' in defences:
        defences.remove('rtt')
        rtt_dir = os.path.join(results_dir, 'rtt')
        for rtt in [d for d in os.listdir(rtt_dir) if os.path.isdir(os.path.join(rtt_dir, d))]:
            defences.append('rtt/' + rtt)

    ## Challengers plots
    # __plot_challengers_curves(defences, results_dir)
    
    ## Create ROC curves per attack
    attacks_to_remove = []
    for attack in attacks:
        to_remove = __ROC_curves_per_attack(attack, defences, xmin=3e-3)
        if to_remove: attacks_to_remove.append(to_remove)
    
    for attack in attacks_to_remove:
        attacks.remove(attack)

    ## Create general comparison plot
    auc           = {}
    tpr_at_01fpr  = {}
    tpr_at_001fpr = {}
    train_acc     = {}
    test_acc      = {}
    epoch_time    = {}
    nb_epochs     = {}

    for d in defences:
        df_utility = pd.read_table(os.path.join(results_dir, d, 'log.txt'), sep='\t')
        df_utility.columns = [str(z).strip() for z in df_utility.columns]
        train_acc[d]  = np.mean(df_utility['Train Acc'].values[-10:])
        test_acc[d]   = np.mean(df_utility['Val Acc'].values[-10:])
        epoch_time[d] = np.mean(df_utility['Time'].values)
        nb_epochs[d]  = len(df_utility['Time'])

        auc[d]           = {}
        tpr_at_01fpr[d]  = {}
        tpr_at_001fpr[d] = {}
        for a in attacks:
            try:
                metrics_attacks = pd.read_table(os.path.join(results_dir, d, 'attacks', f'metrics_{a}.txt'), sep='\t', index_col=0)
                auc[d][a]           = metrics_attacks.loc['AUC', a]
                tpr_at_01fpr[d][a]  = metrics_attacks.loc['TPR@0.1FPR', a]
                tpr_at_001fpr[d][a] = metrics_attacks.loc['TPR@0.01FPR', a]
            except:
                pass # if an attack/defence combinaison doesn't exist, no need to compute any attack performance for it
        
    # markers = {
    #     'entropy' : 'v',
    #     'Mentropy' : '^',
    #     'MAST' : 'o',
    #     'LiRA' : 'X', # '*', 'P'
    # }
    markers = ('v', '^', 'o', 'X', 's', '*', 'h', 'H', 'D', 'd', 'P')
    colors  = list(matplotlib.colors.TABLEAU_COLORS.values())
    
    plt.clf()
    fig, axs = plt.subplots(2, 4, figsize=(16, 9))
    plt.subplots_adjust(left=0, right=.95, wspace=.5, hspace=.5)
    
    # Legend
    ax = axs[0, 0]
    ax.axis('off')
    legend_attacks = [
        matplotlib.lines.Line2D([0], [0], marker=markers[j], color='w', label=attack, markerfacecolor='k', markersize=11)
        for j, attack in enumerate(attacks)
    ]
    ax.legend(title='Attacks:', handles=legend_attacks, fontsize=10)
    ax = axs[1, 0]
    ax.axis('off')
    legend_defences = [
        matplotlib.lines.Line2D([0], [0], color=colors[i % len(colors)], label=defence, linewidth=5)
        for i, defence in enumerate(defences)
    ]
    ax.legend(title='Defences:', handles=legend_defences, fontsize=10)

    # Attack metrics
    for k, (metric_name, values) in enumerate({'AUC':auc, 'TPR@0.1FPR':tpr_at_01fpr, 'TPR@0.01FPR':tpr_at_001fpr}.items()):
        ax = axs[0, k+1]
        ax.grid()
        ax.set_axisbelow(True)
        for i, d in enumerate(defences):
            for j, a in enumerate(attacks):
                try:
                    ax.scatter(test_acc[d], values[d][a], marker=markers[j % len(markers)], color=colors[i % len(colors)], edgecolors='k', linewidth=.5)
                except:
                    pass
                
        ax.set_xlabel('model utility (test acc.)')
        ax.set_ylabel(f'attack performance ({metric_name})')
        ax.set_title(metric_name, fontsize=12)

    # Training values
    # > average time per epoch
    ax = axs[1, 1]
    ax.grid(axis='x')
    ax.set_axisbelow(True)
    ax.barh(list(epoch_time.keys()), epoch_time.values(), color=colors[:len(epoch_time)])
    for j, x in enumerate(epoch_time.values()):
        ax.text(x*.95, j, f'{x:.2f}', va='center', ha='right', color='white', fontsize=9, weight='bold')
    ax.invert_yaxis()
    ax.set_xlabel('Time (secondes)')
    ax.set_title('Average epoch time', fontsize=12)

    # > number of training epochs
    ax = axs[1, 2]
    ax.grid(axis='x')
    ax.set_axisbelow(True)
    ax.barh(list(nb_epochs.keys()), nb_epochs.values(), color=colors[:len(nb_epochs)])
    for j, x in enumerate(nb_epochs.values()):
        ax.text(x*.95, j, f'{x:.0f}', va='center', ha='right', color='white', fontsize=9, weight='bold')
    ax.invert_yaxis()
    ax.set_xlabel('Number epochs')
    ax.set_title('Number of training epochs', fontsize=12)

    # > train and test accuracy
    ax = axs[1, 3]
    ax.grid(axis='x')
    ax.set_axisbelow(True)
    y_pos = np.arange(len(test_acc))
    ax.barh(y_pos, test_acc.values(), height=.4, color=colors[:len(test_acc)])
    for j, x in enumerate(test_acc.values()):
        ax.text(x*.95, j, f'{x:.2f}', va='center', ha='right', color='white', fontsize=9, weight='bold')
    ax.barh(y_pos +.4, train_acc.values(), height=.4, color='#333')
    for j, x in enumerate(train_acc.values()):
        ax.text(x*.95, j+.4, f'{x:.2f}', va='center', ha='right', color='white', fontsize=9, weight='bold')
    ax.set_yticks(y_pos +.2, labels=test_acc.keys())
    ax.invert_yaxis()
    ax.set_xlabel('Accuracy (%)')
    ax.set_title('Train and test accuracies', fontsize=12)
    ax.legend(['test', 'train'], loc='upper left')

    plt.suptitle(f'Attacks/defences summary for model {args.model} on dataset {args.dataset}', fontsize=13, weight='bold')
    savefig( os.path.join(results_dir, f'recap.png') )


if __name__ == '__main__':
    args, results_dir = check_args(parse_arguments())

    if args.mode == 'full_pipeline':
        run_defences(args, results_dir)
        run_attacks(args, results_dir)
        compare_attacks_vs_defences(args, results_dir)

    elif args.mode == 'defences':
        run_defences(args, results_dir)

    elif args.mode == 'attacks':
        run_attacks(args, results_dir)

    elif args.mode == 'comparisons':
        compare_attacks_vs_defences(args, results_dir)

  
    #TODO: check the number of wanted challegers in LiRA