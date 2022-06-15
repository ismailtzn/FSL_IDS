#!/usr/bin/python3
import argparse
import logging
import os
import pickle
import torch
import torch.nn.functional
import torch.optim as optim

import utility
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


# noinspection DuplicatedCode
def parse_configuration():
    now = datetime.now().strftime("%Y_%m_%d:%H_%M_%S")

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="individual_logs")
    parser.add_argument("--experiment_dir_prefix", type=str, default="prototypical")
    parser.add_argument("--experiment_time", type=str, default=now)
    parser.add_argument("--meta_train_n_way", type=int, default=5)
    parser.add_argument("--meta_train_k_shot", type=int, default=5)
    parser.add_argument("--meta_train_query_count", type=int, default=5)
    parser.add_argument("--meta_train_max_epoch", type=int, default=10)
    parser.add_argument("--meta_train_epoch_size", type=int, default=1000)
    parser.add_argument("--dataset_dir", type=str, default="../../datasets/CIC_IDS_2017/cic_ids_2017_prepared_21")
    parser.add_argument("--model_x_dim0", type=int, default=1)
    parser.add_argument("--model_x_dim1", type=int, default=78)
    parser.add_argument("--model_hid_dim", type=int, default=64)
    parser.add_argument("--model_z_dim", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--learning_rate_decay", type=float, default=0.75)
    parser.add_argument("--save_model_path", default="models/model_{}".format(now))
    parser.add_argument("--save_history_path", default="history/history_{}.pkl".format(now))
    parser.add_argument("--meta_test_n_way", type=int, default=5)
    parser.add_argument("--meta_test_k_shot", type=int, default=5)
    parser.add_argument("--meta_test_query_count", type=int, default=5)
    parser.add_argument("--meta_test_episode_count", type=int, default=5)
    parser.add_argument("--experiment_id", type=str, default=now)

    config = parser.parse_args()

    return config


# noinspection DuplicatedCode
def initialize_logger(config):
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    # Setup logging
    log_filename = "{}/run_{}_{}.log".format(config.log_dir, config.experiment_dir_prefix, config.experiment_time)

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_filename, "w+"), logging.StreamHandler()],
        level=logging.INFO
    )
    logging.info("Initialized logging. log_filename = {}".format(log_filename))

    logging.info("Running script with following parameters")
    for arg in vars(config):
        logging.info("Parameter Name: {}    Value: {}".format(arg, getattr(config, arg)))


# noinspection DuplicatedCode
def train(model, train_x, train_y, config, writer):
    """
    Trains the protonet
    Args:
        model: protonet model
        train_x (np.array): training set
        train_y(np.array): labels of training set
        config:
        writer (SummaryWriter): writer instance for tensorflow
    """
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=config.learning_rate_decay, last_epoch=-1)

    epoch = 0  # epochs done so far
    epoch_acc = 0
    epoch_loss = 0

    early_stop_acc_threshold = 0.99
    early_stop_counter = 3

    while epoch < config.meta_train_max_epoch:
        running_loss = 0.0
        running_acc = 0.0

        for episode in tqdm(range(config.meta_train_epoch_size), desc="Epoch {:d} train".format(epoch + 1)):
            (support_set, s_true_labels), (query_set, q_true_labels), class_labels = utility.extract_sample(model.n_way, model.n_support, model.n_query, train_x, train_y)
            optimizer.zero_grad()
            outputs = model.forward(query_set, support_set=support_set)
            loss, metrics, cf_matrix = model.get_protonet_loss_metrics(outputs, q_true_labels, class_labels=class_labels)
            running_loss += metrics["loss"]
            running_acc += metrics["accuracy"]
            loss.backward()
            optimizer.step()

        epoch_loss = running_loss / config.meta_train_epoch_size
        epoch_acc = running_acc / config.meta_train_epoch_size
        logging.info("Epoch {:d} -- Loss: {:.4f} Acc: {:.4f}".format(epoch + 1, epoch_loss, epoch_acc))
        writer.add_scalar("MetaTrain/Accuracy", epoch_acc, epoch)
        writer.add_scalar("MetaTrain/Loss", epoch_loss, epoch)
        writer.flush()

        epoch += 1
        scheduler.step()

        if epoch_acc > early_stop_acc_threshold:
            early_stop_counter -= 1
            if early_stop_counter == 0:
                logging.info("Early stopping since training accuracy is over threshold({}) at least {} consecutive times.".format(early_stop_acc_threshold, early_stop_counter))
                break
        else:
            early_stop_counter = 3

    param_dict = {
        "MetaTrain/learning_rate": config.learning_rate,
        "MetaTrain/learning_rate_decay": config.learning_rate_decay,
        "MetaTrain/max_epoch": config.meta_train_max_epoch,
        "MetaTrain/epoch_size": config.meta_train_epoch_size,
        "MetaTrain/early_stop_counter": early_stop_counter,
        "MetaTrain/n": model.n_way,
        "MetaTrain/k": model.n_support,
        "MetaTrain/q": model.n_query
    }
    metric_dict = {
        "MetaTrain/total_accuracy": epoch_acc,
        "MetaTrain/total_loss": epoch_loss,
    }

    return param_dict, metric_dict


# noinspection DuplicatedCode
def test(model, test_x, test_y, config, writer):
    """
    Tests the protonet
    Args:
        model: trained model
        test_x (np.array): testing set
        test_y (np.array): labels of testing set
        config:
        writer (SummaryWriter): writer instance for tensorflow
    """

    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    history = {"metrics": [], "cf_matrix": [], "class_labels": []}
    for episode in tqdm(range(config.meta_test_episode_count)):
        (support_set, s_true_labels), (query_set, q_true_labels), class_labels = utility.extract_sample(model.n_way, model.n_support, model.n_query, test_x, test_y)
        outputs = model.forward(query_set, support_set=support_set)
        loss, metrics, cf_matrix = model.get_protonet_loss_metrics(outputs, q_true_labels, class_labels=class_labels)
        writer.add_scalar("MetaTest/Accuracy", metrics["accuracy"], episode)
        writer.add_scalar("MetaTest/Loss", metrics["loss"], episode)
        running_loss += metrics["loss"]
        running_acc += metrics["accuracy"]
        history["metrics"].append(metrics)
        history["cf_matrix"].append(cf_matrix)
        history["class_labels"].append(class_labels)
    avg_loss = running_loss / config.meta_test_episode_count
    avg_acc = running_acc / config.meta_test_episode_count
    logging.info("\rTest results -- Loss: {:.4f} Acc: {:.4f}".format(avg_loss, avg_acc))

    avg_history = utility.average_history(history)

    param_dict = {
        "MetaTest/episode": config.meta_test_episode_count,
        "MetaTest/n": model.n_way,
        "MetaTest/k": model.n_support,
        "MetaTest/q": model.n_query
    }
    metric_dict = {
        "MetaTest/AvgAccuracy": avg_acc,
        "MetaTest/AvgLoss": avg_loss,
        "MetaTest/f1-score_macro_avg": avg_history["metrics"]["macro avg"]["f1-score"],
        "MetaTest/f1-score_weighted_avg": avg_history["metrics"]["weighted avg"]["f1-score"],
        "MetaTest/precision_macro_avg": avg_history["metrics"]["macro avg"]["precision"],
        "MetaTest/precision_weighted_avg": avg_history["metrics"]["weighted avg"]["precision"],
        "MetaTest/recall_macro_avg": avg_history["metrics"]["macro avg"]["recall"],
        "MetaTest/recall_weighted_avg": avg_history["metrics"]["weighted avg"]["recall"]
    }
    return param_dict, metric_dict, history


def validate(model, x_val, y_val, config, writer):
    model.eval()
    sample_size = x_val.shape[0]
    model.n_way = len(y_val.unique())
    model.n_support = config.meta_test_k_shot
    model.n_query = int((sample_size / model.n_way) - model.n_support)

    (support_set, s_true_labels), (query_set, q_true_labels), class_labels = utility.extract_sample(model.n_way, model.n_support, model.n_query, x_val, y_val)
    outputs = model.forward(query_set, support_set=support_set)
    loss, metrics, cf_matrix = model.get_protonet_loss_metrics(outputs, q_true_labels, class_labels=class_labels)
    writer.add_scalar("MetaValidation/Accuracy", metrics["accuracy"])
    writer.add_scalar("MetaValidation/Loss", metrics["loss"])
    logging.info("\rValidation results -- Loss: {:.4f} Acc: {:.4f}".format(metrics["loss"], metrics["accuracy"]))

    param_dict = {
        "MetaValidation/n": model.n_way,
        "MetaValidation/k": model.n_support,
        "MetaValidation/q": model.n_query
    }
    metric_dict = {
        "MetaValidation/accuracy": metrics["accuracy"],
        "MetaValidation/loss": metrics["loss"],
        "MetaValidation/f1-score_macro_avg": metrics["macro avg"]["f1-score"],
        "MetaValidation/f1-score_weighted_avg": metrics["weighted avg"]["f1-score"],
        "MetaValidation/precision_macro_avg": metrics["macro avg"]["precision"],
        "MetaValidation/precision_weighted_avg": metrics["weighted avg"]["precision"],
        "MetaValidation/recall_macro_avg": metrics["macro avg"]["recall"],
        "MetaValidation/recall_weighted_avg": metrics["weighted avg"]["recall"]
    }
    history = {"metrics": [metrics], "cf_matrix": [cf_matrix], "class_labels": [class_labels]}

    return param_dict, metric_dict, history


def log_history(config, history, writer, history_type="Test"):
    # Save history
    save_history_path = "{}_{}".format(config.save_history_path, history_type)
    if not os.path.exists(os.path.dirname(save_history_path)):
        os.makedirs(os.path.dirname(save_history_path))

    with open(save_history_path, "wb") as f:
        pickle.dump(history, f)
        logging.info("Writing {} history object to file {}, pickle version {}".format(
            history_type,
            save_history_path,
            pickle.format_version
        ))

        # To load use following
        # with open(save_history_path, "rb") as f:
        #     loaded_dict = pickle.load(f)

    avg_history = utility.average_history(history)

    logging.info("{} Metrics \n{}".format(history_type, utility.tabulate_metrics(avg_history["metrics"], "github")))
    logging.info("{} Average Confusion Matrix \n{}".format(
        history_type,
        utility.tabulate_cf_matrix(avg_history["avg_cf_matrix"], "github", history["class_labels"][-1].tolist()))
    )
    logging.info("{} Sum Confusion Matrix \n{}".format(
        history_type,
        utility.tabulate_cf_matrix(avg_history["avg_cf_matrix"], "github", history["class_labels"][-1].tolist()))
    )

    writer.add_text("{}/Average History".format(history_type), utility.tabulate_metrics(avg_history["metrics"]))
    writer.flush()

    writer.add_text(
        "{}/Average Confusion Matrix".format(history_type),
        utility.tabulate_cf_matrix(avg_history["avg_cf_matrix"], showindex=history["class_labels"][-1].tolist())
    )
    writer.flush()

    writer.add_text(
        "{}/Sum Confusion Matrix".format(history_type),
        utility.tabulate_cf_matrix(avg_history["sum_cf_matrix"], showindex=history["class_labels"][-1].tolist())
    )
    writer.flush()


# noinspection DuplicatedCode
def basic_train_test(config):
    # Check GPU support, please do activate GPU
    logging.info("GPU is ready: {}".format(torch.cuda.is_available()))

    log_dir = "runs/{}_{}".format(config.experiment_dir_prefix, config.experiment_time)

    writer = SummaryWriter(log_dir)

    n_way = config.meta_train_n_way
    n_support = config.meta_train_k_shot
    n_query = config.meta_train_query_count
    sample_count = n_support + n_query

    (train_x, train_y), (test_x, test_y) = utility.load_datasets(config.dataset_dir)

    logging.info("n_way: {}, n_support: {}, n_query: {}".format(n_way, n_support, n_query))
    logging.info("Shapes => train_x:{}, train_y:{}, test_x:{}, test_y:{}".format(train_x.shape, train_y.shape, test_x.shape, test_y.shape))
    logging.info("Train Data classes: {}".format(utility.get_available_classes(train_y, sample_count)))
    logging.info("Test Data classes: {}".format(utility.get_available_classes(test_y, sample_count)))

    model = utility.load_protonet_conv(
        x_dim=(config.model_x_dim0, config.model_x_dim1),
        hid_dim=config.model_hid_dim,
        z_dim=config.model_z_dim,
        n_way=n_way,
        n_support=n_support,
        n_query=n_query
    )
    logging.info(model.encoder)
    writer.add_graph(model.encoder, torch.rand(1, 1, 78).cuda())
    writer.flush()

    param_dict, metric_dict = train(
        model,
        train_x,
        train_y,
        config,
        writer
    )

    # Save model
    if not os.path.exists(os.path.dirname(config.save_model_path)):
        os.makedirs(os.path.dirname(config.save_model_path))
    torch.save(model, config.save_model_path)

    model.n_way = config.meta_test_n_way
    model.n_support = config.meta_test_k_shot
    model.n_query = config.meta_test_query_count
    test_param_dict, test_metric_dict, test_history = test(
        model,
        test_x,
        test_y,
        config,
        writer
    )

    param_dict.update(test_param_dict)
    metric_dict.update(test_metric_dict)

    log_history(config, test_history, writer, "Test")

    x_val, y_val = utility.load_val_datasets(config.dataset_dir)
    val_param_dict, val_metric_dict, val_history = validate(
        model,
        x_val,
        y_val,
        config,
        writer
    )
    param_dict.update(val_param_dict)
    metric_dict.update(val_metric_dict)
    log_history(config, val_history, writer, "Validation")

    param_dict["model_x_dim_0"] = config.model_x_dim0
    param_dict["model_x_dim_1"] = config.model_x_dim1
    param_dict["model_hid_dim"] = config.model_hid_dim
    param_dict["model_z_dim"] = config.model_z_dim
    param_dict["experiment_id"] = config.experiment_id

    writer.add_hparams(param_dict, metric_dict)
    writer.flush()
    writer.close()


if __name__ == "__main__":
    configuration = parse_configuration()
    initialize_logger(configuration)
    basic_train_test(configuration)
