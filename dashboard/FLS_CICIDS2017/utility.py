import numpy as np
import pandas as pd
import os

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional
import torch.optim as optim
from torch.autograd import Variable

import glob
from tqdm import tqdm

def load_datasets(data_dir, hdf_key="cic_ids_2017"):
    # Lists of train, val, test files (X and y)
    x_train_files = glob.glob(data_dir + "/" + "x_train*")
    y_train_files = glob.glob(data_dir + "/" + "y_train*")
    x_val_files = glob.glob(data_dir + "/" + "x_val*")
    y_val_files = glob.glob(data_dir + "/" + "y_val*")
    x_test_files = glob.glob(data_dir + "/" + "x_test*")
    y_test_files = glob.glob(data_dir + "/" + "y_test*")

    x_train_files.sort()
    y_train_files.sort()
    x_val_files.sort()
    y_val_files.sort()
    x_test_files.sort()
    y_test_files.sort()

    assert len(x_train_files) > 0
    assert len(y_train_files) > 0

    x_train_dfs = [pd.read_hdf(file, hdf_key) for file in x_train_files]
    x_val_dfs = [pd.read_hdf(file, hdf_key) for file in x_val_files]
    x_test_dfs = [pd.read_hdf(file, hdf_key) for file in x_test_files]
    y_train_dfs = [pd.read_hdf(file, hdf_key) for file in y_train_files]
    y_val_dfs = [pd.read_hdf(file, hdf_key) for file in y_val_files]
    y_test_dfs = [pd.read_hdf(file, hdf_key) for file in y_test_files]

    x_train = pd.concat(x_train_dfs)
    x_val = pd.concat(x_val_dfs)
    x_test = pd.concat(x_test_dfs)
    y_train = pd.concat(y_train_dfs)
    y_val = pd.concat(y_val_dfs)
    y_test = pd.concat(y_test_dfs)

    return (x_train, y_train), (x_test, y_test), (x_val, y_val)


def euclidean_dist(x, y):
    """
    Computes euclidean distance btw x and y
    Args:
        x (torch.Tensor): shape (n, d). n usually n_way*n_query
        y (torch.Tensor): shape (m, d). m usually n_way
    Returns:
        torch.Tensor: shape(n, m). For each query, the distances to each centroid
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)