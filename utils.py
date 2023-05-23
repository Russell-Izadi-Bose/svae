#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Russell Izadi 2023
"""

from munch import Munch
import yaml
import os
import logging
from typing import Tuple
import math

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
import torchaudio

from vocalset import VOCALSET
import random
from itertools import accumulate


import torch
from torch.utils.data import Dataset


class Transform(Dataset):
    def __init__(self, dataset, transform):
        # Store the original dataset, transform function
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        # Return the length of the original dataset
        return len(self.dataset)

    def __getitem__(self, idx):
        # Return the transformed sample
        return self.transform(self.dataset[idx])


class Pair(Dataset):
    """Pairs a sample with a random sample from the same class
    """

    def __init__(self, dataset, attr):
        self.dataset = dataset
        self.attr = attr

    def __getitem__(self, idx):
        path_wav, label = self.dataset[idx]
        indices = self.dataset.indices[self.attr][label[self.attr]]
        idx_ = random.choice(indices)
        path_wav_, label_ = self.dataset[idx_]
        assert label[self.attr] == label_[self.attr]
        return path_wav, path_wav_, self.attr

    def __len__(self):
        return len(self.dataset)


def read_yml(path):
    """Converts a YAML file into an object with hierarchical attributes

    Args:
        path (string): The path to the YAML file (.yml)

    Returns:
        args (Munch): A Munch instance
    """

    assert path.endswith(".yml")
    assert os.path.exists(path), path

    with open(path, 'r', encoding='ASCII') as stream:
        try:
            args = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    args = Munch().fromDict(args)
    return args
