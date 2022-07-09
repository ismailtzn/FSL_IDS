#!/usr/bin/env python3
import argparse
import os
import sys
import logging
import time
from pprint import pformat
import pandas as pd
import numpy as np
import glob


def write_to_hdf(df, filename, key, compression_level, mode="a", fmt="fixed"):
    logging.info("Writing dataset to HDF5 format. filename={}".format(filename))
    t0 = time.time()

    df.to_hdf(filename, key=key, mode=mode, complevel=compression_level, complib="zlib", format=fmt)

    logging.info("Writing complete. time_to_write={}".format(time.time() - t0))


def count_labels(label_col):
    label_counts = label_col.value_counts(dropna=False)
    total_count = sum(label_counts)

    logging.info("Total count = {}".format(total_count))

    label_percentages = label_counts / total_count

    return label_counts, label_percentages


def initial_setup(config):
    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)

    # Setup logging
    log_filename = config.out_dir + "/" + "run_log.log"

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_filename, "w+"), logging.StreamHandler()],
        level=logging.INFO
    )
    logging.info("Initialized logging. log_filename = {}".format(log_filename))

    logging.info("Running script with following parameters\n{}".format(pformat(config)))


def load_train_datasets(data_dir, hdf_key="cic_ids_2017"):
    x_train_files = glob.glob(data_dir + "/" + "x_meta_train*")
    y_train_files = glob.glob(data_dir + "/" + "y_meta_train*")
    x_test_files = glob.glob(data_dir + "/" + "x_meta_test*")
    y_test_files = glob.glob(data_dir + "/" + "y_meta_test*")

    x_train_files.extend(x_test_files)
    y_train_files.extend(y_test_files)
    x_train_files.sort()
    y_train_files.sort()

    x_train_dfs = [pd.read_hdf(file, hdf_key) for file in x_train_files]
    y_train_dfs = [pd.read_hdf(file, hdf_key) for file in y_train_files]

    x_train = pd.concat(x_train_dfs)
    y_train = pd.concat(y_train_dfs)

    return x_train, y_train


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


def load_datasets(data_dir, hdf_key="cic_ids_2017"):
    return load_train_datasets(data_dir, hdf_key), load_val_datasets(data_dir, hdf_key)


def rearrange_datasets(config):
    (x_all, y_all), (x_meta_val, y_meta_val) = load_datasets(config.base_dir)

    x_test_dfs = []
    y_test_series = []


    for class_name in config.meta_test_class:
        y_test = y_all[y_all == class_name].copy()
        y_test_series.append(y_test)
        y_all = y_all.drop(y_test.index)
        x_test_df = x_all.loc[y_test.index]
        x_test_dfs.append(x_test_df)
        x_all = x_all.drop(x_test_df.index)


    x_meta_train = x_all
    y_meta_train = y_all
    x_meta_test = pd.concat(x_test_dfs)
    y_meta_test = pd.concat(y_test_series)

    # Save data files in HDF format
    logging.info("Saving prepared datasets (train, val, test) to: {}".format(config.out_dir))

    write_to_hdf(x_meta_train, config.out_dir + "/" + "x_meta_train.h5", config.hdf_key, 5)
    write_to_hdf(y_meta_train, config.out_dir + "/" + "y_meta_train.h5", config.hdf_key, 5)

    write_to_hdf(x_meta_test, config.out_dir + "/" + "x_meta_test.h5", config.hdf_key, 5)
    write_to_hdf(y_meta_test, config.out_dir + "/" + "y_meta_test.h5", config.hdf_key, 5)

    write_to_hdf(x_meta_val, config.out_dir + "/" + "x_meta_val.h5", config.hdf_key, 5)
    write_to_hdf(y_meta_val, config.out_dir + "/" + "y_meta_val.h5", config.hdf_key, 5)


    logging.info("Saving complete")

    logging.info("Meta train dataset shape = {}".format(x_meta_train.shape))
    logging.info("Meta test dataset shape = {}".format(x_meta_test.shape))
    logging.info("Meta val dataset shape = {}".format(x_meta_val.shape))
    logging.info("Meta train dataset  \n{}".format(y_meta_train.value_counts()))
    logging.info("Meta test dataset  \n{}".format(y_meta_test.value_counts()))
    logging.info("Meta val dataset  \n{}".format(y_meta_val.value_counts()))
    logging.info("{} Labels used int meta train = \n{}".format(len(y_meta_train.unique()), ", ".join([str(x) for x in y_meta_train.unique()])))
    logging.info("{} Labels used int meta test = \n{}".format(len(y_meta_test.unique()), ", ".join([str(x) for x in y_meta_test.unique()])))
    logging.info("{} Labels used int meta val = \n{}".format(len(y_meta_val.unique()), ", ".join([str(x) for x in y_meta_val.unique()])))


def parse_configuration():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf_key", type=str, default="cic_ids_2017")
    parser.add_argument("--base_dir", type=str, default="cic_ids_2017_prepared_4-way10-shot_T8n460q_E4n460q_V12n120q")
    parser.add_argument("--out_dir", type=str, default="cic_ids_2017_new")
    parser.add_argument("--meta_test_class", type=list, default=["PortScan", "Bot", "FTP-Patator", "SSH-Patator"])
    config = parser.parse_args()

    return config


if __name__ == "__main__":
    configuration = parse_configuration()
    initial_setup(configuration)
    rearrange_datasets(configuration)
    logging.info("Data preparation complete")
