import copy
import numbers

import numpy as np
import pandas as pd
import glob
import torch
import torch.nn as nn
import torch.nn.functional
import ProtoNet
from torch.utils.tensorboard.summary import hparams
from torch.utils.tensorboard import SummaryWriter
from tabulate import tabulate


def load_train_datasets(data_dir, hdf_key="cic_ids_2017"):
    x_train_files = glob.glob(data_dir + "/" + "x_meta_train*")
    y_train_files = glob.glob(data_dir + "/" + "y_meta_train*")

    x_train_files.sort()
    y_train_files.sort()

    assert len(x_train_files) > 0
    assert len(y_train_files) > 0

    x_train_dfs = [pd.read_hdf(file, hdf_key) for file in x_train_files]
    y_train_dfs = [pd.read_hdf(file, hdf_key) for file in y_train_files]

    x_train = pd.concat(x_train_dfs)
    y_train = pd.concat(y_train_dfs)

    return x_train, y_train


def load_test_datasets(data_dir, hdf_key="cic_ids_2017"):
    x_test_files = glob.glob(data_dir + "/" + "x_meta_test*")
    y_test_files = glob.glob(data_dir + "/" + "y_meta_test*")

    x_test_files.sort()
    y_test_files.sort()

    x_test_dfs = [pd.read_hdf(file, hdf_key) for file in x_test_files]
    y_test_dfs = [pd.read_hdf(file, hdf_key) for file in y_test_files]

    x_test = pd.concat(x_test_dfs)
    y_test = pd.concat(y_test_dfs)

    return x_test, y_test


def load_datasets(data_dir, hdf_key="cic_ids_2017"):
    return load_train_datasets(data_dir, hdf_key), load_test_datasets(data_dir, hdf_key)


def load_val_datasets(data_dir, hdf_key="cic_ids_2017"):
    x_val_files = glob.glob(data_dir + "/" + "x_meta_val*")
    y_val_files = glob.glob(data_dir + "/" + "y_meta_val*")

    x_val_files.sort()
    y_val_files.sort()

    x_val_dfs = [pd.read_hdf(file, hdf_key) for file in x_val_files]
    y_val_dfs = [pd.read_hdf(file, hdf_key) for file in y_val_files]

    x_val = pd.concat(x_val_dfs)
    y_val = pd.concat(y_val_dfs)

    return x_val, y_val


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
          (torch.Tensor): sample. Size (n_way, n_support+n_query, (dim))
          (list): True label indexes
          (list): Used class names.
    """
    support_set = []
    s_true_labels = []
    query_set = []
    q_true_labels = []
    feature_count = data_x.shape[1]
    available_classes = get_available_classes(data_y, (n_support + n_query))

    class_labels = np.random.choice(available_classes, n_way, replace=False)
    class_labels.sort()

    for i, cls in enumerate(class_labels):
        datax_cls = data_x[data_y == cls]
        perm = np.random.permutation(datax_cls)

        support_set.append(perm[:n_support])
        s_true_labels.extend([i] * n_support)
        query_set.append(perm[n_support:n_support + n_query])
        q_true_labels.extend([i] * n_query)

    support_set = np.array(support_set)
    support_set = torch.from_numpy(support_set).float()
    support_set = support_set.view(n_way, n_support, 1, feature_count)
    support_set = support_set.cuda()
    s_true_labels = torch.Tensor(s_true_labels).cuda()
    s_true_labels = s_true_labels.view(n_way, n_support, 1).long()
    s_true_labels.requires_grad_(False)

    query_set = np.array(query_set)
    query_set = torch.from_numpy(query_set).float()
    query_set = query_set.view(n_way, n_query, 1, feature_count)
    query_set = query_set.cuda()

    q_true_labels = torch.Tensor(q_true_labels).cuda()
    q_true_labels = q_true_labels.view(n_way, n_query, 1).long()
    q_true_labels.requires_grad_(False)

    return (support_set, s_true_labels), (query_set, q_true_labels), class_labels


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
    n_way = kwargs["n_way"]
    n_support = kwargs["n_support"]
    n_query = kwargs["n_query"]

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

    return ProtoNet.ProtoNet(encoder, n_way, n_support, n_query)


def load_protonet_ann(**kwargs):
    """
    Loads the prototypical network model
    Arg:
        x_dim (tuple): dimension of input instance
        hid_dim (int): dimension of hidden layers
        z_dim (int): dimension of embedded instance
    Returns:
        Model (Class ProtoNet)
    """
    x_dim = kwargs["x_dim"]
    hid_dim = kwargs["hid_dim"]
    z_dim = kwargs["z_dim"]
    n_way = kwargs["n_way"]
    n_support = kwargs["n_support"]
    n_query = kwargs["n_query"]

    layer1 = nn.Sequential(
        nn.Linear(in_features=x_dim[1], out_features=hid_dim),
        nn.BatchNorm1d(1),
        nn.ReLU(),
        nn.Dropout(0.2)
    )
    layer2 = nn.Sequential(
        nn.Linear(in_features=hid_dim, out_features=hid_dim),
        nn.BatchNorm1d(1),
        nn.ReLU(),
        nn.Dropout(0.2)
    )
    layer3 = nn.Sequential(
        nn.Linear(in_features=hid_dim, out_features=z_dim),
        nn.BatchNorm1d(1),
        nn.ReLU(),
        nn.Dropout(0.2)
    )
    encoder = nn.Sequential(
        layer1,
        layer2,
        layer3,
        ProtoNet.Flatten()
    )

    return ProtoNet.ProtoNet(encoder, n_way, n_support, n_query)

def sum_dicts(x, y):
    result = {}
    if x.keys() != y.keys():
        raise ValueError("x and y should be in same dict format")
    for key in x.keys():
        if isinstance(x[key], numbers.Number) and isinstance(y[key], numbers.Number):
            result[key] = x[key] + y[key]
        elif isinstance(x[key], dict) and isinstance(y[key], dict):
            result[key] = sum_dicts(x[key], y[key])
        else:
            raise ValueError("x[key] and y[key] should be in same dict format")
    return result


def multiply_dict_with_number(d, n):
    result = {}

    for key in d.keys():
        if isinstance(d[key], numbers.Number):
            result[key] = d[key] * n
        elif isinstance(d[key], dict):
            result[key] = multiply_dict_with_number(d[key], n)
        else:
            raise ValueError("d[key] should be number")
    return result


def average_history(history):
    avg_metrics = None

    for item in history["metrics"]:
        if avg_metrics is None:
            avg_metrics = copy.deepcopy(item)
            continue

        avg_metrics = sum_dicts(avg_metrics, item)
    avg_metrics = multiply_dict_with_number(avg_metrics, (1 / len(history["metrics"])))

    avg_hist = {
        "metrics": avg_metrics,
        "avg_cf_matrix": np.around(np.average([x for x in history["cf_matrix"]], axis=0), decimals=3),
        "sum_cf_matrix": sum(history["cf_matrix"])
    }

    return avg_hist


def tabulate_metrics(metrics, tablefmt="html"):
    table = []
    header = ["F1-Score", "Precision", "Recall"]

    non_class_keys = ["macro avg", "weighted avg", "accuracy", "loss"]
    for key, value in metrics.items():
        if key not in non_class_keys:
            table.append([key, value["f1-score"], value["precision"], value["recall"]])

    table.append(["", "", "", ""])
    table.append(["*MACRO AVG*", metrics["macro avg"]["f1-score"], metrics["macro avg"]["precision"], metrics["macro avg"]["recall"]])
    table.append(["*WEIGHTED AVG*", metrics["weighted avg"]["f1-score"], metrics["weighted avg"]["precision"], metrics["weighted avg"]["recall"]])

    return tabulate(table, headers=header, tablefmt=tablefmt)


def tabulate_cf_matrix(cf_table, tablefmt="html", showindex="always"):
    if isinstance(showindex, str):
        header = range(cf_table.shape[1])
    else:
        header = showindex
    return tabulate(cf_table, tablefmt=tablefmt, showindex=showindex, headers=header)


def add_hparams(writer, param_dict, metrics_dict):
    exp, ssi, sei = hparams(param_dict, metrics_dict)
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)
    for k, v in metrics_dict.items():
        writer.add_scalar(k, v)


setattr(SummaryWriter, "add_hparams", add_hparams)
