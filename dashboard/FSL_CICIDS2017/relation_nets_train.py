#!/usr/bin/python3
import argparse
import logging
import os
import pickle
import torch
import torch.nn.functional
import torch.optim as optim
import csv

import RelationNet
import utility
import numpy as np
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


# noinspection DuplicatedCode
def parse_configuration():
    now = datetime.now().strftime("%Y_%m_%d:%H_%M_%S")

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="individual_logs")
    parser.add_argument("--experiment_dir_prefix", type=str, default="relation_nets")
    parser.add_argument("--experiment_time", type=str, default=now)
    parser.add_argument("--meta_train_n_way", type=int, default=9)
    parser.add_argument("--meta_train_k_shot", type=int, default=5)
    parser.add_argument("--meta_train_query_count", type=int, default=5)
    parser.add_argument("--meta_train_max_epoch", type=int, default=10)
    parser.add_argument("--meta_train_epoch_size", type=int, default=1000)
    parser.add_argument("--dataset_dir", type=str, default="../../datasets/CIC_IDS_2017/cic_ids_2017_prepared_4-way_test_36-per_label_10-per_val")
    parser.add_argument("--model_x_dim0", type=int, default=1)
    parser.add_argument("--model_x_dim1", type=int, default=78)
    parser.add_argument("--model_encoder_hid_dim", type=int, default=128)
    parser.add_argument("--model_encoder_out_dim", type=int, default=64)
    parser.add_argument("--model_rel_net_hid_dim", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--learning_rate_decay", type=float, default=0.75)
    parser.add_argument("--save_model_path", default="models/model_{}".format(now))
    parser.add_argument("--save_history_path", default="history/history_{}".format(now))
    parser.add_argument("--meta_test_n_way", type=int, default=4)
    parser.add_argument("--meta_test_k_shot", type=int, default=5)
    parser.add_argument("--meta_test_query_count", type=int, default=5)
    parser.add_argument("--meta_test_episode_count", type=int, default=5)
    parser.add_argument("--experiment_id", type=str, default=now)
    parser.add_argument("--early_stop_change_acc_threshold", type=float, default=0.0002)
    parser.add_argument("--early_stop_acc_window_length", type=int, default=5)

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
    # TODO::: weights init

    model.n_way = config.meta_train_n_way
    model.n_support = config.meta_train_k_shot
    model.n_query = config.meta_train_query_count

    feature_encoder_optim = optim.Adam(model.feature_encoder.parameters(), lr=config.learning_rate)
    feature_encoder_scheduler = optim.lr_scheduler.StepLR(feature_encoder_optim, step_size=10000, gamma=config.learning_rate_decay)
    relation_network_optim = optim.Adam(model.relation_network.parameters(), lr=config.learning_rate)
    relation_network_scheduler = optim.lr_scheduler.StepLR(relation_network_optim, step_size=10000, gamma=config.learning_rate_decay)

    epoch = 0  # epochs done so far
    epoch_acc = 0
    epoch_loss = 0

    early_stop_acc_window = []
    history = {"metrics": [], "cf_matrix": [], "class_labels": []}
    early_stop = False

    while epoch < config.meta_train_max_epoch and not early_stop:
        running_loss = 0.0
        running_acc = 0.0

        for episode in tqdm(range(config.meta_train_epoch_size), desc="Epoch {:d} train".format(epoch + 1)):
            (support_set, s_true_labels), (query_set, q_true_labels), class_labels = utility.extract_sample(
                model.n_way,
                model.n_support,
                model.n_query,
                train_x,
                train_y
            )
            feature_encoder_optim.zero_grad()
            relation_network_optim.zero_grad()

            outputs = model.forward(query_set, support_set=support_set)
            loss, metrics, cf_matrix = model.get_relation_net_loss_metrics(outputs, q_true_labels, class_labels=class_labels)
            running_loss += metrics["loss"]
            running_acc += metrics["accuracy"]
            history["metrics"].append(metrics)
            history["cf_matrix"].append(cf_matrix)
            history["class_labels"].append(class_labels)
            loss.backward()
            feature_encoder_optim.step()
            relation_network_optim.step()

        epoch_loss = running_loss / config.meta_train_epoch_size
        epoch_acc = running_acc / config.meta_train_epoch_size
        logging.info("Epoch {:d} -- Loss: {:.4f} Acc: {:.4f}".format(epoch + 1, epoch_loss, epoch_acc))
        writer.add_scalar("MetaTrain/Accuracy", epoch_acc, epoch)
        writer.add_scalar("MetaTrain/Loss", epoch_loss, epoch)
        writer.flush()

        epoch += 1
        feature_encoder_scheduler.step()
        relation_network_scheduler.step()

        early_stop_acc_window.append(epoch_acc)
        if len(early_stop_acc_window) > config.early_stop_acc_window_length:
            early_stop_acc_window.pop(0)
            change_mean = np.abs([early_stop_acc_window[i] - early_stop_acc_window[i - 1] for i in range(1, len(early_stop_acc_window))]).mean()
            logging.info("Epoch {:d} -- change_mean {:.10f}".format(epoch, change_mean))
            if epoch_acc > 0.995 or change_mean < config.early_stop_change_acc_threshold:
                logging.info("Early stopping")
                early_stop = True

    param_dict = {
        "MetaTrain/learning_rate": config.learning_rate,
        "MetaTrain/learning_rate_decay": config.learning_rate_decay,
        "MetaTrain/max_epoch": config.meta_train_max_epoch,
        "MetaTrain/epoch_size": config.meta_train_epoch_size,
        "MetaTrain/early_stop": early_stop,
        "MetaTrain/n": model.n_way,
        "MetaTrain/k": model.n_support,
        "MetaTrain/q": model.n_query
    }
    metric_dict = {
        "MetaTrain/total_accuracy": epoch_acc,
        "MetaTrain/total_loss": epoch_loss,
    }

    return param_dict, metric_dict, history


# noinspection DuplicatedCode
def test(model, test_x, test_y, config, writer):
    model.n_way = config.meta_test_n_way
    model.n_support = config.meta_test_k_shot
    model.n_query = config.meta_test_query_count

    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    history = {"metrics": [], "cf_matrix": [], "class_labels": []}
    for episode in tqdm(range(config.meta_test_episode_count)):
        (support_set, s_true_labels), (query_set, q_true_labels), class_labels = utility.extract_sample(
            model.n_way,
            model.n_support,
            model.n_query,
            test_x,
            test_y
        )

        outputs = model.forward(query_set, support_set=support_set)
        loss, metrics, cf_matrix = model.get_relation_net_loss_metrics(outputs, q_true_labels, class_labels=class_labels)
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

    (support_set, s_true_labels), (query_set, q_true_labels), class_labels = utility.extract_sample(
        model.n_way,
        model.n_support,
        model.n_query,
        x_val,
        y_val
    )

    outputs = model.forward(query_set, support_set=support_set)
    loss, metrics, cf_matrix = model.get_relation_net_loss_metrics(outputs, q_true_labels, class_labels=class_labels)
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


def log_experiment_params(config, experiment_params):
    # Save history
    save_experiment_params_path = "{}_experiment_params.pkl".format(config.save_history_path)
    if not os.path.exists(os.path.dirname(save_experiment_params_path)):
        os.makedirs(os.path.dirname(save_experiment_params_path))

    with open(save_experiment_params_path, "wb") as f:
        pickle.dump(save_experiment_params_path, f)
        logging.info("Writing experiment_params object to file {}, pickle version {}".format(
            save_experiment_params_path,
            pickle.format_version
        ))

    fieldnames = [key for key in experiment_params.keys()]
    fieldnames.remove("experiment_id")
    fieldnames.insert(0, "experiment_id")

    save_experiment_params_csv_path = "{}_experiment_params.csv".format(config.save_history_path)
    with open(save_experiment_params_csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerow(experiment_params)


def log_history(config, history, writer, history_type="Test"):
    # Save history
    save_history_path = "{}_{}.pkl".format(config.save_history_path, history_type)
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

    feature_encoder = RelationNet.RelNetEncoder(
        x_dim0=config.model_x_dim0,
        hid_dim=config.model_encoder_hid_dim,
        out_dim=config.model_encoder_out_dim
    )
    relation_network = RelationNet.RelationNetwork(
        input_size=(2 * config.model_encoder_out_dim),
        hidden_size=config.model_rel_net_hid_dim
    )

    model = RelationNet.RelationNet(
        feature_encoder,
        relation_network,
        n_way,
        n_support,
        n_query
    )

    logging.info("Feature Encoder:")
    logging.info(feature_encoder)
    logging.info("Relation Network:")
    logging.info(relation_network)
    writer.add_graph(model, input_to_model=(torch.rand(model.n_way * model.n_query, 1, 78).cuda(), torch.rand(model.n_way, model.n_support, 1, 78).cuda()))
    writer.flush()

    param_dict, metric_dict, train_history = train(
        model,
        train_x,
        train_y,
        config,
        writer
    )

    if not os.path.exists(os.path.dirname(config.save_model_path)):
        os.makedirs(os.path.dirname(config.save_model_path))
    torch.save(model, config.save_model_path)

    log_history(config, train_history, writer, "Train")

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
    param_dict["model_encoder_hid_dim"] = config.model_encoder_hid_dim
    param_dict["model_encoder_out_dim"] = config.model_encoder_out_dim
    param_dict["model_rel_net_hid_dim"] = config.model_rel_net_hid_dim
    param_dict["experiment_id"] = config.experiment_id
    param_dict["experiment_type"] = config.experiment_dir_prefix

    writer.add_hparams(param_dict, metric_dict)
    writer.flush()

    log_experiment_params(config, {**param_dict, **metric_dict})

    writer.close()


if __name__ == "__main__":
    configuration = parse_configuration()
    initialize_logger(configuration)
    basic_train_test(configuration)
