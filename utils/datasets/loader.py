import os
import sys
import numpy as np
import torch
import torchvision.transforms as transforms

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(FILE_DIR, 'data')
sys.path.append(os.path.join(FILE_DIR, '../'))

from utils.helper import mkdir
from .cifar import CIFAR2, CIFAR10, CIFAR100


class DatasetLoader():
    def __init__(self, dataset_name, random_seed, batchsize):
        self.set_cuda_device()
        self.random_seed = random_seed
        self.batchsize = batchsize
        self.nb_classes = None

        self.data_dir = os.path.join(DATA_ROOT, dataset_name)
        if os.path.exists(self.data_dir):
            self.data_exists = True
        else:
            self.data_exists = False
        
        self.idx_dir = os.path.join(self.data_dir, f'seed{random_seed}')
        mkdir(self.idx_dir)
        
        print(f'Loading dataset {dataset_name}...')
        if dataset_name == 'CIFAR2':
            self.__cifar2()
        elif dataset_name == 'CIFAR10':
            self.__cifar10()
        elif dataset_name == 'CIFAR100':
            self.__cifar100()
        else:
            raise NotImplementedError
    
    def set_cuda_device(self):
        """The function to set CUDA device."""
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

    def get_train_loader(self):
        return self.dataloader(self.tr_set, suffle=True)

    def get_test_loader(self):
        return self.dataloader(self.te_set, suffle=False)

    def get_shadow_train_loader(self):
        return self.dataloader(self.s_tr_set, suffle=True)

    def get_shadow_test_loader(self):
        return self.dataloader(self.s_te_set, suffle=False)
    
    def get_challengers_loader(self):
        return self.dataloader(self.chal_set, suffle=False)
    
    def get_dataloader(self):
        return self.dataloader
    

    def __set_partition_idx(self, dataset_size, nb_train, nb_test, nb_shadow_train, nb_shadow_test, nb_challengers):
        if dataset_size < nb_train + nb_test + nb_shadow_train + nb_shadow_test:
            raise Exception('Data repartition impossible')
        
        idx_file_path = os.path.join(self.idx_dir, 'idx.npy')
        if os.path.exists(idx_file_path):
            indices = np.load(idx_file_path)
        else:
            np.random.seed(self.random_seed)
            indices = np.arange(dataset_size).astype(int)
            np.random.shuffle(indices)
            np.save(idx_file_path, indices)
        
        self.idx_tr   = indices[:nb_train]
        self.idx_te   = indices[nb_train:(nb_train+nb_test)]
        self.idx_s_tr = indices[(nb_train+nb_test):(nb_train+nb_test+nb_shadow_train)]
        self.idx_s_te = indices[(nb_train+nb_test+nb_shadow_train):(nb_train+nb_test+nb_shadow_train+nb_shadow_test)]
        self.idx_chal = np.concatenate([self.idx_tr[:nb_challengers//2], self.idx_te[:nb_challengers//2]])


    def __cifar2(self, data_augmentation =False):
        self.nb_classes = 2

        # Data partition
        self.__set_partition_idx(
            dataset_size    = 12_000,
            nb_train        =  4_000,
            nb_test         =  2_000,
            nb_shadow_train =  4_000,
            nb_shadow_test  =  2_000,
            nb_challengers  =  1_000
        )      

        # Data transformation
        if data_augmentation:
            print('with data augmentation')
            transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                  transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                       (0.2023, 0.1994, 0.2010))])
        else:
            transform_train = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                       (0.2023, 0.1994, 0.2010))])

        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


        # Set datasets and dataloader
        self.tr_set = CIFAR2(
            root = self.data_dir, indices = self.idx_tr,
            download = not self.data_exists, transform = transform_train
        )
        self.te_set = CIFAR2(
            root = self.data_dir, indices = self.idx_te,
            download = not self.data_exists, transform = transform_test
        )
        self.s_tr_set = CIFAR2(
            root = self.data_dir, indices = self.idx_s_tr,
            download = not self.data_exists, transform = transform_train
        )
        self.s_te_set = CIFAR2(
            root = self.data_dir, indices = self.idx_s_te,
            download = not self.data_exists, transform = transform_test
        )
        self.chal_set = CIFAR2(
            root = self.data_dir, indices = self.idx_chal,
            download = not self.data_exists, transform = transform_test
        )

        self.dataloader = lambda set, suffle: torch.utils.data.DataLoader(set,
                                               batch_size=self.batchsize, shuffle=suffle,
                                               generator=torch.Generator(device=self.device))
        
    
    def __cifar10(self, data_augmentation =False):
        self.nb_classes = 10

        # Data partition
        self.__set_partition_idx(
            dataset_size    = 60_000,
            nb_train        = 20_000,
            nb_test         = 10_000,
            nb_shadow_train = 20_000,
            nb_shadow_test  = 10_000,
            nb_challengers  =  5_000
        )      

        # Data transformation
        if data_augmentation:
            print('with data augmentation')
            transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                  transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                       (0.2023, 0.1994, 0.2010))])
        else:
            transform_train = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                       (0.2023, 0.1994, 0.2010))])

        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


        # Set datasets and dataloader
        self.tr_set = CIFAR10(
            root = self.data_dir, indices = self.idx_tr,
            download = not self.data_exists, transform = transform_train
        )
        self.te_set = CIFAR10(
            root = self.data_dir, indices = self.idx_te,
            download = not self.data_exists, transform = transform_test
        )
        self.s_tr_set = CIFAR10(
            root = self.data_dir, indices = self.idx_s_tr,
            download = not self.data_exists, transform = transform_train
        )
        self.s_te_set = CIFAR10(
            root = self.data_dir, indices = self.idx_s_te,
            download = not self.data_exists, transform = transform_test
        )
        self.chal_set = CIFAR10(
            root = self.data_dir, indices = self.idx_chal,
            download = not self.data_exists, transform = transform_test
        )

        self.dataloader = lambda set, suffle: torch.utils.data.DataLoader(set,
                                               batch_size=self.batchsize, shuffle=suffle,
                                               generator=torch.Generator(device=self.device))
        
    
    def __cifar100(self, data_augmentation =False):
        self.nb_classes = 100

        # Data partition
        self.__set_partition_idx(
            dataset_size    = 60_000,
            nb_train        = 20_000,
            nb_test         = 10_000,
            nb_shadow_train = 20_000,
            nb_shadow_test  = 10_000,
            nb_challengers  =  5_000
        )      

        # Data transformation
        if data_augmentation:
            print('With data augmentation')
            transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                  transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                       (0.2023, 0.1994, 0.2010))])
        else:
            transform_train = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                       (0.2023, 0.1994, 0.2010))])

        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


        # Set datasets and dataloader
        self.tr_set = CIFAR100(
            root = self.data_dir, indices = self.idx_tr,
            download = not self.data_exists, transform = transform_train
        )
        self.te_set = CIFAR100(
            root = self.data_dir, indices = self.idx_te,
            download = not self.data_exists, transform = transform_test
        )
        self.s_tr_set = CIFAR100(
            root = self.data_dir, indices = self.idx_s_tr,
            download = not self.data_exists, transform = transform_train
        )
        self.s_te_set = CIFAR100(
            root = self.data_dir, indices = self.idx_s_te,
            download = not self.data_exists, transform = transform_test
        )
        self.chal_set = CIFAR100(
            root = self.data_dir, indices = self.idx_chal,
            download = not self.data_exists, transform = transform_test
        )

        self.dataloader = lambda set, suffle: torch.utils.data.DataLoader(set,
                                               batch_size=self.batchsize, shuffle=suffle,
                                               generator=torch.Generator(device=self.device))



