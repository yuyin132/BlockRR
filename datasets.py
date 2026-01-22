import torch
import logging

from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import CIFAR10
from typing import Sequence
from torchvision import transforms


def make_dataset(data_name, task_name, data_dir, batch_size, seed, ratio):
    if data_name == 'cifar10_1':
        trainset = CIFAR10(root=data_dir, train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
        ]))

        testset = CIFAR10(root=data_dir, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ]))
        train_num_data = (5000, 4900, 4700, 4600, 4500, 4800, 1000, 1500, 1000, 1500)
        test_num_data = (500, 490, 470, 460, 450, 480, 100, 150, 100, 150)
        num_classes = 10

    if data_name == 'cifar10_2':
        trainset = CIFAR10(root=data_dir, train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
        ]))

        testset = CIFAR10(root=data_dir, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ]))
        train_num_data = (5000, 4900, 4700, 4600, 4500, 4800, 600, 500, 700, 400)
        test_num_data = (500, 490, 470, 460, 450, 480, 60, 50, 70, 40)
        num_classes = 10

    if task_name == 'classification':
        # make datasets imbalanced

        train_total_sample_size = sum(train_num_data)
        train_cnt_dict = dict()
        train_total_cnt = 0
        train_indices = []
        for i in range(len(trainset)):
            if train_total_cnt == train_total_sample_size:
                break
            label = trainset[i][1]
            if label not in train_cnt_dict:
                train_cnt_dict[label] = 1
                train_total_cnt += 1
                train_indices.append(i)
            else:
                if train_cnt_dict[label] == train_num_data[label]:
                    continue
                else:
                    train_cnt_dict[label] += 1
                    train_total_cnt += 1
                    train_indices.append(i)
        train_indices = torch.tensor(train_indices) # , dtype=torch.int32
        indices = torch.randperm(train_indices.size(0))
        shuffled_train_indices = torch.index_select(train_indices, 0, indices)
        
        train_rr_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, sampler=CustomSubsetRandomSampler(shuffled_train_indices), num_workers=8,
            shuffle=False)

        train_length = int(len(train_indices) * ratio)
        est_length = len(train_indices) - train_length
        train_indices_kfold, est_indices = torch.utils.data.random_split(
                    train_indices, (train_length, est_length), generator=torch.Generator().manual_seed(seed))
        logging.info(f"number of estimateset: {len(est_indices)}")
        logging.info(f"number of trainset: {len(train_indices_kfold)}")
        estimate_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size,sampler=CustomSubsetRandomSampler(est_indices),num_workers=8,
            shuffle=False)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, sampler=CustomSubsetRandomSampler(train_indices_kfold), num_workers=8,
            shuffle=False)
        
        test_total_sample_size = sum(test_num_data)
        test_cnt_dict = dict()
        test_total_cnt = 0
        test_indices = []
        for i in range(len(testset)):

            if test_total_cnt == test_total_sample_size:
                break

            label = testset[i][1]
            if label not in test_cnt_dict:
                test_cnt_dict[label] = 1
                test_total_cnt += 1
                test_indices.append(i)
            else:
                if test_cnt_dict[label] == test_num_data[label]:
                    continue
                else:
                    test_cnt_dict[label] += 1
                    test_total_cnt += 1
                    test_indices.append(i)
        test_indices = torch.tensor(test_indices)

        
        logging.info(f"number of testset: {len(test_indices)}, sample distribution: {test_num_data}")
        
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, sampler=CustomSubsetRandomSampler(test_indices), num_workers=8)

        return estimate_loader, trainloader, train_rr_loader, testloader, num_classes



class CustomSubsetRandomSampler(SubsetRandomSampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    def __init__(self, indices: Sequence[int], generator=None) -> None:
        super().__init__(indices, generator)
    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))



class MyTensorDataset(torch.utils.data.Dataset):
    """Custom Dataset for Tensor images with transforms."""
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        
        if self.transform:
            img = self.transform(img)  # ToPIL -> augment -> ToTensor

        return img, label



