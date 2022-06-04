import logging
import os
from pprint import pformat

import torch
import torch.nn.functional
import torch.optim as optim

import utility
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


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
    epoch_acc = 0
    epoch_loss = 0

    while epoch < max_epoch:
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
        logging.info("Epoch {:d} -- Loss: {:.4f} Acc: {:.4f}".format(epoch + 1, epoch_loss, epoch_acc))
        writer.add_scalar("Meta Train Accuracy/Epochs", epoch_acc, epoch)
        writer.add_scalar("Meta Train Loss/Epochs", epoch_loss, epoch)
        writer.flush()

        epoch += 1
        scheduler.step()

    param_dict = {
        "lr": lr,
        "max_epoch": max_epoch
    }
    metric_dict = {
        "meta_train/accuracy": epoch_acc,
        "meta_train/loss": epoch_loss,
    }

    return param_dict, metric_dict


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
    logging.info("\rTest results -- Loss: {:.4f} Acc: {:.4f}".format(avg_loss, avg_acc))

    param_dict = {
        "meta_episode": test_episode,
        "meta_test_n": model.n_way,
        "meta_test_k": model.n_support,
        "meta_test_q": model.n_query
    }
    metric_dict = {
        "meta_test/accuracy": avg_acc,
        "meta_test/loss": avg_loss,
    }
    return param_dict, metric_dict


def parse_configuration():
    # TODO::: implement config parser here!
    config = {
        "log_dir": "logs",
        "experiment_dir_prefix": "prototypical",
        "experiment_time": datetime.now().strftime("%Y_%m_%d:%H_%M_%S"),
        "meta_train_n_way": 5,
        "meta_train_k_shot": 5,
        "meta_train_query_count": 5,
        "meta_train_max_epoch": 10,
        "meta_train_epoch_size": 1000,
        "dataset_dir": "../../datasets/CIC_IDS_2017/cic_ids_2017_prepared_21",
        "model_x_dim": (1, 78),
        "model_hid_dim": 64,
        "model_z_dim": 64,
        "save_model_path": "latest_model",
        "meta_test_n_way": 5,
        "meta_test_k_shot": 5,
        "meta_test_query_count": 5,
        "meta_test_episode_count": 5
    }
    return config


def basic_train_test(config):
    # Check GPU support, please do activate GPU
    logging.info("GPU is ready: {}".format(torch.cuda.is_available()))

    log_dir = "runs/{}_{}".format(config["experiment_dir_prefix"], config["experiment_time"])

    writer = SummaryWriter(log_dir)

    n_way = config["meta_train_n_way"]
    n_support = config["meta_train_k_shot"]
    n_query = config["meta_train_query_count"]
    sample_count = n_support + n_query

    (train_x, train_y), (test_x, test_y) = utility.load_datasets("../../datasets/CIC_IDS_2017/cic_ids_2017_prepared_21")

    logging.info("n_way: {}, n_support: {}, n_query: {}".format(n_way, n_support, n_query))
    logging.info("Shapes => train_x:{}, train_y:{}, test_x:{}, test_y:{}".format(train_x.shape, train_y.shape, test_x.shape, test_y.shape))
    logging.info("Train Data classes: {}".format(utility.get_available_classes(train_y, sample_count)))
    logging.info("Test Data classes: {}".format(utility.get_available_classes(test_y, sample_count)))

    model = utility.load_protonet_conv(
        x_dim=config["model_x_dim"],
        hid_dim=config["model_hid_dim"],
        z_dim=config["model_z_dim"],
        n_way=n_way,
        n_support=n_support,
        n_query=n_query
    )

    logging.info(model.encoder)
    writer.add_graph(model.encoder, torch.rand(1, 1, 78).cuda())
    writer.flush()

    param_dict, metric_dict = train(model, train_x, train_y, config["meta_train_max_epoch"], config["meta_train_epoch_size"], writer)
    torch.save(model, "latest_model")

    model.n_way = config["meta_test_n_way"]
    model.n_support = config["meta_test_k_shot"]
    model.n_query = config["meta_test_query_count"]

    test_param_dict, test_metric_dict = test(model, test_x, test_y, config["meta_test_episode_count"], writer)
    param_dict.update(test_param_dict)
    metric_dict.update(test_metric_dict)
    writer.add_hparams(param_dict, metric_dict)
    writer.flush()

    writer.close()


def initialize_logger(config):
    if not os.path.exists(config["log_dir"]):
        os.makedirs(config["log_dir"])

    # Setup logging
    log_filename = "{}/run_{}_{}.log".format(config["log_dir"], config["experiment_dir_prefix"], config["experiment_time"])

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_filename, "w+"), logging.StreamHandler()],
        level=logging.INFO
    )
    logging.info("Initialized logging. log_filename = {}".format(log_filename))

    logging.info("Running script with following parameters\n{}".format(pformat(config)))


if __name__ == "__main__":
    config = parse_configuration()
    initialize_logger(config)
    basic_train_test(config)
