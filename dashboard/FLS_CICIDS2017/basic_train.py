import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim as optim
from torch.autograd import Variable

import utility
from tqdm import tqdm


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
    random_classes = np.random.choice(np.unique(data_y), n_way, replace=False)
    for cls in random_classes:
        datax_cls = data_x[data_y == cls]
        perm = np.random.permutation(datax_cls)
        sample_cls = perm[:(n_support + n_query)]
        sample.append(sample_cls)
    sample = np.array(sample)
    sample = torch.from_numpy(sample).float()
    sample = sample.view(n_way, (n_support + n_query), 78, 1)
    # sample = sample.permute(0, 1, 4, 2, 3)
    return ({
        "sample_data": sample,
        "n_way": n_way,
        "n_support": n_support,
        "n_query": n_query
    })


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def load_protonet_conv(**kwargs):
    """
    Loads the prototypical network model
    Arg:
        x_dim (tuple): dimension of input image
        hid_dim (int): dimension of hidden layers in conv blocks
        z_dim (int): dimension of embedded image
    Returns:
        Model (Class ProtoNet)
    """
    x_dim = kwargs["x_dim"]
    hid_dim = kwargs["hid_dim"]
    z_dim = kwargs["z_dim"]

    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

    encoder = nn.Sequential(
        conv_block(x_dim[0], hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, z_dim),
        Flatten()
    )

    return ProtoNet(encoder)


class ProtoNet(nn.Module):
    def __init__(self, encoder):
        """
        Args:
            encoder : CNN encoding the images in sample
            n_way (int): number of classes in a classification task
            n_support (int): number of labeled examples per class in the support set
            n_query (int): number of labeled examples per class in the query set
        """
        super(ProtoNet, self).__init__()
        self.encoder = encoder.cuda()

    def set_forward_loss(self, sample):
        """
        Computes loss, accuracy and output for classification task
        Args:
            sample (torch.Tensor): shape (n_way, n_support+n_query, (dim))
        Returns:
            torch.Tensor: shape(2), loss, accuracy and y_hat
        """
        sample_data = sample["sample_data"].cuda()
        n_way = sample["n_way"]
        n_support = sample["n_support"]
        n_query = sample["n_query"]

        x_support = sample_data[:, :n_support]
        x_query = sample_data[:, n_support:]

        # target indices are 0 ... n_way-1
        target_inds = torch.arange(0, n_way).view(n_way, 1, 1).expand(n_way, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)
        target_inds = target_inds.cuda()

        # encode images of the support and the query set
        x = torch.cat([x_support.contiguous().view(n_way * n_support, *x_support.size()[2:]),
                       x_query.contiguous().view(n_way * n_query, *x_query.size()[2:])], 0)

        z = self.encoder.forward(x)
        z_dim = z.size(-1)  # usually 64
        z_proto = z[:n_way * n_support].view(n_way, n_support, z_dim).mean(1)
        z_query = z[n_way * n_support:]

        # compute distances
        dists = utility.euclidean_dist(z_query, z_proto)

        # compute probabilities
        log_p_y = torch.nn.functional.log_softmax(-dists, dim=1).view(n_way, n_query, -1)

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            "loss": loss_val.item(),
            "acc": acc_val.item(),
            "y_hat": y_hat
        }


def train(model, optimizer, train_x, train_y, n_way, n_support, n_query, max_epoch, epoch_size):
    """
    Trains the protonet
    Args:
        model
        optimizer
        train_x (np.array): training set
        train_y(np.array): labels of training set
        n_way (int): number of classes in a classification task
        n_support (int): number of labeled examples per class in the support set
        n_query (int): number of labeled examples per class in the query set
        max_epoch (int): max epochs to train on
        epoch_size (int): episodes per epoch
    """
    # divide the learning rate by 2 at each epoch
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=-1)
    epoch = 0  # epochs done so far
    stop = False  # status to know when to stop

    while epoch < max_epoch and not stop:
        running_loss = 0.0
        running_acc = 0.0

        for episode in tqdm(range(epoch_size), desc="Epoch {:d} train".format(epoch + 1)):
            sample = extract_sample(n_way, n_support, n_query, train_x, train_y)
            optimizer.zero_grad()
            loss, output = model.set_forward_loss(sample)
            running_loss += output["loss"]
            running_acc += output["acc"]
            loss.backward()
            optimizer.step()
        epoch_loss = running_loss / epoch_size
        epoch_acc = running_acc / epoch_size
        print("Epoch {:d} -- Loss: {:.4f} Acc: {:.4f}".format(epoch + 1, epoch_loss, epoch_acc))
        epoch += 1
        scheduler.step()


def test(model, test_x, test_y, n_way, n_support, n_query, test_episode):
    """
    Tests the protonet
    Args:
        model: trained model
        test_x (np.array): testing set
        test_y (np.array): labels of testing set
        n_way (int): number of classes in a classification task
        n_support (int): number of labeled examples per class in the support set
        n_query (int): number of labeled examples per class in the query set
        test_episode (int): number of episodes to test on
    """
    running_loss = 0.0
    running_acc = 0.0
    for episode in tqdm(range(test_episode)):
        sample = extract_sample(n_way, n_support, n_query, test_x, test_y)
        loss, output = model.set_forward_loss(sample)
        running_loss += output["loss"]
        running_acc += output["acc"]
    avg_loss = running_loss / test_episode
    avg_acc = running_acc / test_episode
    print("Test results -- Loss: {:.4f} Acc: {:.4f}".format(avg_loss, avg_acc))

def basic_train_test():
    # Check GPU support, please do activate GPU
    print(torch.cuda.is_available())

    (train_x, train_y), (test_x, test_y), (_, _) = utility.load_datasets("../../datasets/CIC_IDS_2017/cic_ids_2017_small")
    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    model = load_protonet_conv(
        x_dim=(78, 1),
        hid_dim=64,
        z_dim=64,
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    n_way = 4
    n_support = 5
    n_query = 5

    max_epoch = 5
    epoch_size = 100

    train(model, optimizer, train_x, train_y, n_way, n_support, n_query, max_epoch, epoch_size)

    test_episode = 100
    test(model, test_x, test_y, n_way, n_support, n_query, test_episode)

    one_sample = extract_sample(n_way, n_support, n_query, test_x, test_y)
    one_sample_loss, one_sample_output = model.set_forward_loss(one_sample)

if __name__ == "__main__":
    basic_train_test()
