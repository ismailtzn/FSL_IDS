#!/usr/bin/env python3
import argparse
import os
import sys
import logging
import time
from pprint import pformat
import pandas as pd
import numpy as np
import re
import zipfile

# Data Processing Imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer


def reduce_column(s, to_keep):
    '''
    Reduces the string values found in a column
    to the values provided in the list "to_keep".
    ---
    Input:
        s: string
        to_keep: list of strings
    Returns:
        string, s if s should be kept, else "other"
    '''
    s = s.lower().strip()
    if s not in to_keep:
        return "other"
    else:
        return s


def write_to_hdf(df, filename, key, compression_level, mode="a", fmt="fixed"):
    logging.info("Writing dataset to HDF5 format. filename={}".format(filename))
    t0 = time.time()

    df.to_hdf(filename, key=key, mode=mode, complevel=compression_level, complib="zlib", format=fmt)

    logging.info("Writing complete. time_to_write={}".format(time.time() - t0))


def load_datasets(files_list):
    logging.info("Loading datasets in files")

    dfs = []
    for filename in files_list:
        df = pd.read_csv(filename, header=0)
        dfs.append(df)
    all_data = pd.concat(dfs, ignore_index=True)

    logging.info(all_data.info())

    all_data.columns = all_data.columns.str.strip()
    all_data.columns = all_data.columns.str.replace(" ", "_")
    all_data = all_data.rename(columns={"attack_cat": "Label"})
    all_data.drop(columns=["label"], inplace=True)
    logging.info("Loading datasets complete")
    return all_data


def count_labels(label_col):
    label_counts = label_col.value_counts(dropna=False)
    total_count = sum(label_counts)

    logging.info("Total count = {}".format(total_count))

    label_percentages = label_counts / total_count

    return label_counts, label_percentages


def initial_setup(params):
    if not os.path.exists(params["output_dir"]):
        os.makedirs(params["output_dir"])

    # Setup logging
    log_filename = params["output_dir"] + "/" + "run_log.log"

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_filename, "w+"), logging.StreamHandler()],
        level=logging.INFO
    )
    logging.info("Initialized logging. log_filename = {}".format(log_filename))

    logging.info("Running script with following parameters\n{}".format(pformat(params)))


def split_meta_datasets(all_data, params):
    # ignore labels with small number of datapoint
    min_sample_count_per_train_class = params["k_shot"] * 2 + params["q_val"] + params["q_train"]
    min_sample_count_per_test_class = params["k_shot"] * 2 + params["q_val"] + params["q_test"]
    validation_sample_per_class = params["k_shot"] + params["q_val"]

    labels_sorted = all_data["Label"].value_counts().keys().to_list()

    if params["n_train"] + params["n_test"] > len(labels_sorted):
        raise ValueError("n_train + n_test should be less than or equal to length of labels")

    meta_train_labels = labels_sorted[:params["n_train"]]
    meta_test_labels = labels_sorted[params["n_train"]:params["n_train"] + params["n_test"]]

    meta_train_df = all_data[all_data["Label"].isin(meta_train_labels)].copy()
    meta_test_df = all_data[all_data["Label"].isin(meta_test_labels)].copy()

    meta_train_df = meta_train_df.groupby("Label", group_keys=False).apply(lambda x: x.sample(min_sample_count_per_train_class))
    meta_test_df = meta_test_df.groupby("Label", group_keys=False).apply(lambda x: x.sample(min_sample_count_per_test_class))

    meta_val_df_from_train = meta_train_df.groupby("Label", group_keys=False).apply(lambda x: x.sample(validation_sample_per_class)).copy()
    meta_train_df = meta_train_df.drop(meta_val_df_from_train.index)
    meta_val_df_from_test = meta_test_df.groupby("Label", group_keys=False).apply(lambda x: x.sample(validation_sample_per_class)).copy()
    meta_test_df = meta_test_df.drop(meta_val_df_from_test.index)
    meta_val_df = pd.concat([meta_val_df_from_train, meta_val_df_from_test])

    remaining_df = all_data.drop(meta_train_df.index)  # save for later usages
    remaining_df = remaining_df.drop(meta_test_df.index)  # save for later usages
    remaining_df = remaining_df.drop(meta_val_df.index)  # save for later usages

    x_meta_train = meta_train_df.loc[:, meta_train_df.columns != "Label"]
    y_meta_train = meta_train_df["Label"]

    x_meta_test = meta_test_df.loc[:, meta_test_df.columns != "Label"]
    y_meta_test = meta_test_df["Label"]

    x_meta_val = meta_val_df.loc[:, meta_val_df.columns != "Label"]
    y_meta_val = meta_val_df["Label"]

    logging.info("Train set :\n{}".format(meta_train_df["Label"].value_counts()))
    logging.info("Test set :\n{}".format(meta_test_df["Label"].value_counts()))
    logging.info("Validation set :\n{}".format(meta_val_df["Label"].value_counts()))
    logging.info("Remaining df :\n{}".format(remaining_df["Label"].value_counts()))

    return x_meta_train, y_meta_train, x_meta_test, y_meta_test, x_meta_val, y_meta_val, remaining_df


def pre_process_dataset(params):
    # Load data
    logging.info("Loading datasets")

    all_data = load_datasets(params["unsw_nb15_files_list"])

    # Check class labels
    label_counts, label_perc = count_labels(all_data["Label"])
    logging.info("Label counts below")
    logging.info("\n{}".format(label_counts))

    logging.info("Label percentages below")
    pd.set_option("display.float_format", "{:.4f}".format)
    logging.info("\n{}".format(label_perc))
    return all_data


def prepare_unsw_nb15_datasets(params):
    all_data = pre_process_dataset(params)

    x_meta_train, y_meta_train, x_meta_test, y_meta_test, x_meta_val, y_meta_val, remaining_df = split_meta_datasets(all_data, params)
    # Save data files in HDF format
    logging.info("Saving prepared datasets (train, val, test) to: {}".format(params["output_dir"]))

    write_to_hdf(x_meta_train, params["output_dir"] + "/" + "x_meta_train.h5", params["hdf_key"], 5)
    write_to_hdf(y_meta_train, params["output_dir"] + "/" + "y_meta_train.h5", params["hdf_key"], 5)

    write_to_hdf(x_meta_test, params["output_dir"] + "/" + "x_meta_test.h5", params["hdf_key"], 5)
    write_to_hdf(y_meta_test, params["output_dir"] + "/" + "y_meta_test.h5", params["hdf_key"], 5)

    write_to_hdf(x_meta_val, params["output_dir"] + "/" + "x_meta_val.h5", params["hdf_key"], 5)
    write_to_hdf(y_meta_val, params["output_dir"] + "/" + "y_meta_val.h5", params["hdf_key"], 5)

    write_to_hdf(remaining_df, params["output_dir"] + "/" + "remaining_df.h5", params["hdf_key"], 5)

    logging.info("Saving complete")

    logging.info("Meta train dataset shape = {}".format(x_meta_train.shape))
    logging.info("Meta test dataset shape = {}".format(x_meta_test.shape))
    logging.info("Meta val dataset shape = {}".format(x_meta_val.shape))
    logging.info("{} Labels used int meta train = \n{}".format(len(y_meta_train.unique()), ", ".join([str(x) for x in y_meta_train.unique()])))
    logging.info("{} Labels used int meta test = \n{}".format(len(y_meta_test.unique()), ", ".join([str(x) for x in y_meta_test.unique()])))
    logging.info("{} Labels used int meta val = \n{}".format(len(y_meta_val.unique()), ", ".join([str(x) for x in y_meta_val.unique()])))


def parse_configuration():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf_key", type=str, default="unsw_nb15")
    parser.add_argument("--output_dir_prefix", type=str, default="unsw_nb15_prepared")
    parser.add_argument("--unsw_nb15_datasets_dir", type=str, default="UNSW-NB15 - CSV Files")
    parser.add_argument("--n_train", type=int, default=6)
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--q_train", type=int, default=465)
    parser.add_argument("--n_test", type=int, default=3)
    parser.add_argument("--q_test", type=int, default=465)
    parser.add_argument("--q_val", type=int, default=125)

    config = parser.parse_args()

    params = {
        "output_dir": "{}_{}-way{}-shot_T{}n{}q_E{}n{}q_V{}n{}q".format(
            config.output_dir_prefix,
            config.n_test,
            config.k_shot,
            config.n_train,
            config.q_train,
            config.n_test,
            config.q_test,
            (config.n_train + config.n_test),
            config.q_val,
        ),
        "n_val": (config.n_train + config.n_test),
        "unsw_nb15_files_list": [
            config.unsw_nb15_datasets_dir + "/a part of training and testing set/UNSW_NB15_testing-set.csv",
            config.unsw_nb15_datasets_dir + "/a part of training and testing set/UNSW_NB15_training-set.csv"
        ]
    }
    for arg in vars(config):
        params[arg] = getattr(config, arg)

    return params


if __name__ == "__main__":
    parameters = parse_configuration()
    initial_setup(parameters)
    prepare_unsw_nb15_datasets(parameters)
    logging.info("Data preparation complete")
