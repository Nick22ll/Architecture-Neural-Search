import copy
import os
import sys

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import transforms


def data_transform_cifar10():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return transform

def data_reverse_transform_cifar10():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1/el for el in CIFAR_STD]),
                                   transforms.Normalize(mean=[el * (-1) for el in CIFAR_MEAN],
                                                        std=[1., 1., 1.]),
                                   ])
    return invTrans

class CIFAR10Dataset(Dataset):
    def __init__(self, path, train, device, quantity=100):
        if train:
            dataset = torchvision.datasets.CIFAR10(root=path, train=True, download=True, transform=data_transform_cifar10())
        else:
            dataset = torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform=data_transform_cifar10())

        self.classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

        if quantity == 100:
            self.data = []
            self.labels = torch.empty(0, dtype=torch.int64, device=device)

            for data, label in dataset:
                self.data.append(data.to(device))
                self.labels = torch.hstack((self.labels, torch.tensor(label, dtype=torch.int64, device=device)))
        else:
            self.data = np.empty(len(dataset), dtype=object)
            self.labels = torch.empty(len(dataset), dtype=torch.int64)
            for idx, (data, label) in enumerate(dataset):
                self.data[idx] = data
                self.labels[idx] = torch.tensor(label, dtype=torch.int64)
            quantity = quantity / 100
            classes = np.unique(self.labels)
            to_keep = []
            for label in classes:
                indices = np.where(label == self.labels)[0]
                indices = np.random.choice(indices, int(len(indices) * quantity))
                to_keep.extend(indices)

            self.data = self.data[to_keep]
            self.labels = self.labels[to_keep]

            self.labels = self.labels.to(device)
            for i in range(len(self.data)):
                self.data[i] = self.data[i].to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], self.labels[i]

    def to(self, device):
        self.data = [elem.to(device) for elem in self.data]
        self.labels = self.labels.to(device)

    def half(self):
        self.data = [elem.half() for elem in self.data]

    def split(self, percentage):
        rng = np.random.default_rng(22)
        train_indices = np.empty(0, dtype=int)
        val_indices = np.empty(0, dtype=int)
        for label in np.unique(self.labels):
            label_indices = np.where(self.labels == label)[0]
            to_train = rng.choice(label_indices, int((percentage / 100) * len(label_indices)), replace=False)
            to_val = list(np.delete(label_indices, np.argwhere(np.isin(label_indices, to_train))))
            train_indices = np.hstack((train_indices, to_train))
            val_indices = np.hstack((val_indices, to_val))
        train_copy = copy.deepcopy(self)
        val_copy = copy.deepcopy(self)
        train_copy.data = [train_copy.data[i] for i in train_indices]
        train_copy.labels = train_copy.labels[train_indices]
        val_copy.data = [val_copy.data[i] for i in val_indices]
        val_copy.labels = val_copy.labels[val_indices]
        return train_copy, val_copy



