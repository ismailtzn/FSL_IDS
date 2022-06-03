import torch
import torch.nn.functional
import torch.optim as optim

import utility
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter


def train(model, train_x, train_y, max_epoch, epoch_size, writer):
    """
    Trains the protonet
    Args:
        model
        train_x (np.array): training set
        train_y(np.array): labels of training set
        max_epoch (int): max epochs to train on
        epoch_size (int): episodes per epoch
        writer (SummaryWriter): writer instance for tensorflow
    """
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.75, last_epoch=-1)

    epoch = 0  # epochs done so far
    stop = False  # status to know when to stop



    while epoch < max_epoch and not stop:
        running_loss = 0.0
        running_acc = 0.0

        for episode in tqdm(range(epoch_size), desc="Epoch {:d} train".format(epoch + 1)):
            meta_train_sample = utility.extract_sample(model.n_way, model.n_support, model.n_query, train_x, train_y)
            optimizer.zero_grad()
            x = model.pre_process_meta_sample(meta_train_sample)
            outputs = model.forward(x)
            loss, acc_val = model.get_protonet_loss_accuracy(outputs)
            writer.add_scalar("Meta Train Accuracy", acc_val.item(), epoch_size * epoch + episode)
            writer.add_scalar("Meta Train Loss", loss.item(), epoch_size * epoch + episode)
            running_loss += loss.item()
            running_acc += acc_val
            loss.backward()
            optimizer.step()

        epoch_loss = running_loss / epoch_size
        epoch_acc = running_acc / epoch_size
        print("\rEpoch {:d} -- Loss: {:.4f} Acc: {:.4f}".format(epoch + 1, epoch_loss, epoch_acc))
        writer.add_scalar("Meta Train Accuracy/Epochs", epoch_acc, epoch)
        writer.add_scalar("Meta Train Loss/Epochs", epoch_loss, epoch)
        writer.flush()

        epoch += 1
        scheduler.step()

    writer.add_hparams(
        {"lr": lr, "max_epoch": max_epoch},
        {
            "accuracy": epoch_acc,
            "loss": epoch_loss,
        },
    )
    writer.flush()


def test(model, test_x, test_y, test_episode, writer):
    """
    Tests the protonet
    Args:
        model: trained model
        test_x (np.array): testing set
        test_y (np.array): labels of testing set
        test_episode (int): number of episodes to test on
        writer (SummaryWriter): writer instance for tensorflow
    """
    running_loss = 0.0
    running_acc = 0.0
    for episode in tqdm(range(test_episode)):
        meta_test_sample = utility.extract_sample(model.n_way, model.n_support, model.n_query, test_x, test_y)
        x = model.pre_process_meta_sample(meta_test_sample)
        outputs = model.forward(x)
        loss, acc_val = model.get_protonet_loss_accuracy(outputs)
        writer.add_scalar("Meta Test Accuracy", acc_val.item(), episode)
        writer.add_scalar("Meta Test Loss", loss.item(), episode)
        running_loss += loss.item()
        running_acc += acc_val
    avg_loss = running_loss / test_episode
    avg_acc = running_acc / test_episode
    print("\rTest results -- Loss: {:.4f} Acc: {:.4f}".format(avg_loss, avg_acc))


def basic_train_test():
    # Check GPU support, please do activate GPU
    print("GPU is ready: {}".format(torch.cuda.is_available()))

    writer = SummaryWriter()

    n_way = 5
    n_support = 5
    n_query = 5
    sample_count = n_support + n_query

    (train_x, train_y), (test_x, test_y) = utility.load_datasets("../../datasets/CIC_IDS_2017/cic_ids_2017_prepared_21")

    print("n_way: {}, n_support: {}, n_query: {}".format(n_way, n_support, n_query))
    print("Shapes => train_x:{}, train_y:{}, test_x:{}, test_y:{}".format(train_x.shape, train_y.shape, test_x.shape, test_y.shape))
    print("Train Data classes: {}".format(utility.get_available_classes(train_y, sample_count)))
    print("Test Data classes: {}".format(utility.get_available_classes(test_y, sample_count)))

    model = utility.load_protonet_conv(
        x_dim=(1, 78),
        hid_dim=64,
        z_dim=64,
        n_way=n_way,
        n_support=n_support,
        n_query=n_query
    )
    # writer.add_graph(model, torch.rand(1, 1, 78).cuda())
    writer.add_graph(model.encoder, torch.rand(1, 1, 78).cuda())
    writer.flush()

    max_epoch = 50
    epoch_size = 1000

    train(model, train_x, train_y, max_epoch, epoch_size, writer)
    torch.save(model, "latest_model")

    model.n_way = n_way
    model.n_support = n_support
    model.n_query = n_query
    test_episode = 1000
    test(model, test_x, test_y, test_episode, writer)


if __name__ == "__main__":
    basic_train_test()
