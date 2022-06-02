import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim as optim
from torch.autograd import Variable

import utility
from tqdm import tqdm





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
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.75, last_epoch=-1)
    epoch = 0  # epochs done so far
    stop = False  # status to know when to stop

    while epoch < max_epoch and not stop:
        running_loss = 0.0
        running_acc = 0.0

        for episode in tqdm(range(epoch_size), desc="Epoch {:d} train".format(epoch + 1)):
            sample = utility.extract_sample(n_way, n_support, n_query, train_x, train_y)
            optimizer.zero_grad()
            loss, output = model.set_forward_loss(sample)
            running_loss += output["loss"]
            running_acc += output["acc"]
            loss.backward()
            optimizer.step()
        epoch_loss = running_loss / epoch_size
        epoch_acc = running_acc / epoch_size
        print("\rEpoch {:d} -- Loss: {:.4f} Acc: {:.4f}".format(epoch + 1, epoch_loss, epoch_acc))
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
        sample = utility.extract_sample(n_way, n_support, n_query, test_x, test_y)
        loss, output = model.set_forward_loss(sample)
        running_loss += output["loss"]
        running_acc += output["acc"]
    avg_loss = running_loss / test_episode
    avg_acc = running_acc / test_episode
    print("\rTest results -- Loss: {:.4f} Acc: {:.4f}".format(avg_loss, avg_acc))


def basic_train_test():
    # Check GPU support, please do activate GPU
    print(torch.cuda.is_available())

    n_way = 11
    n_support = 5
    n_query = 25
    sample_count = n_support + n_query
    (train_x, train_y), (test_x, test_y) = utility.load_datasets("../../datasets/CIC_IDS_2017/cic_ids_2017_prepared_150")

    print("n_way: {}, n_support: {}, n_query: {}".format(n_way, n_support, n_query))
    print("Shapes => train_x:{}, train_y:{}, test_x:{}, test_y:{}"
          .format(train_x.shape, train_y.shape, test_x.shape, test_y.shape))
    print("Train Data classes: {}".format(utility.get_available_classes(train_y, sample_count)))
    print("Test Data classes: {}".format(utility.get_available_classes(test_y, sample_count)))

    model = utility.load_protonet_conv(
        x_dim=(1, 78),
        hid_dim=64,
        z_dim=64,
    )
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    max_epoch = 10
    epoch_size = 1000

    train(model, optimizer, train_x, train_y, n_way, n_support, n_query, max_epoch, epoch_size)
    torch.save(model, "latest_model")
    test_episode = 1000
    test(model, test_x, test_y, n_way, n_support, n_query, test_episode)

    one_sample = utility.extract_sample(n_way, n_support, n_query, test_x, test_y)
    one_sample_loss, one_sample_output = model.set_forward_loss(one_sample)


if __name__ == "__main__":
    basic_train_test()
