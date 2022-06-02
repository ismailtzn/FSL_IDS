import numpy as np
import pandas as pd
import glob
import torch
import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim as optim
from torch.autograd import Variable
import ProtoNet


def load_train_datasets(data_dir, hdf_key="cic_ids_2017"):
    x_train_files = glob.glob(data_dir + "/" + "x_train*")
    y_train_files = glob.glob(data_dir + "/" + "y_train*")

    x_train_files.sort()
    y_train_files.sort()

    assert len(x_train_files) > 0
    assert len(y_train_files) > 0

    x_train_dfs = [pd.read_hdf(file, hdf_key) for file in x_train_files]
    y_train_dfs = [pd.read_hdf(file, hdf_key) for file in y_train_files]

    x_train = pd.concat(x_train_dfs)
    y_train = pd.concat(y_train_dfs)

    return (x_train, y_train)


def load_test_datasets(data_dir, hdf_key="cic_ids_2017"):
    x_test_files = glob.glob(data_dir + "/" + "x_test*")
    y_test_files = glob.glob(data_dir + "/" + "y_test*")

    x_test_files.sort()
    y_test_files.sort()

    x_test_dfs = [pd.read_hdf(file, hdf_key) for file in x_test_files]
    y_test_dfs = [pd.read_hdf(file, hdf_key) for file in y_test_files]

    x_test = pd.concat(x_test_dfs)
    y_test = pd.concat(y_test_dfs)

    return (x_test, y_test)


def load_datasets(data_dir, hdf_key="cic_ids_2017"):
    return load_train_datasets(data_dir, hdf_key), load_test_datasets(data_dir, hdf_key)


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


def get_available_classes(data_y, sample_count):
    available_classes = []
    for idx, name in enumerate(data_y.value_counts().index.tolist()):
        if data_y.value_counts()[idx] >= sample_count:
            available_classes.append(name)
    return available_classes


# Create samples
def extract_sample(n_way, n_support, n_query, data_x, data_y):
    """
    Picks random sample of size n_support+n_query, for n_way classes
    Args:
        n_way (int): number of classes in a classification task
        n_support (int): number of labeled examples per class in the support set
        n_query (int): number of labeled examples per class in the query set
        data_x (np.array): dataset of samples
        data_y (np.array): dataset of labels
    Returns:
        (dict) of:
          (torch.Tensor): sample. Size (n_way, n_support+n_query, (dim))
          (int): n_way
          (int): n_support
          (int): n_query
    """
    sample = []
    available_classes = get_available_classes(data_y, (n_support + n_query))

    random_classes = np.random.choice(available_classes, n_way, replace=False)
    for cls in random_classes:
        datax_cls = data_x[data_y == cls]
        perm = np.random.permutation(datax_cls)
        sample_cls = perm[:(n_support + n_query)]
        sample.append(sample_cls)
    sample = np.array(sample)
    sample = torch.from_numpy(sample).float()
    sample = sample.view(n_way, (n_support + n_query), 1, 78)
    return ({
        "sample_data": sample,
        "n_way": n_way,
        "n_support": n_support,
        "n_query": n_query
    })


def load_protonet_conv(**kwargs):
    """
    Loads the prototypical network model
    Arg:
        x_dim (tuple): dimension of input instance
        hid_dim (int): dimension of hidden layers in conv blocks
        z_dim (int): dimension of embedded instance
    Returns:
        Model (Class ProtoNet)
    """
    x_dim = kwargs["x_dim"]
    hid_dim = kwargs["hid_dim"]
    z_dim = kwargs["z_dim"]

    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

    encoder = nn.Sequential(
        conv_block(x_dim[0], hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, z_dim),
        ProtoNet.Flatten()
    )

    return ProtoNet.ProtoNet(encoder)
