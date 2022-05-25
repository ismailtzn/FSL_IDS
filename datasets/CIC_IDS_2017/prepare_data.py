#!/usr/bin/env python3

import os
import sys
import logging
import time
from pprint import pformat
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
import zipfile


class Params:
    pass


def read_csv(filename, header_row=0, dtypes=None, columns_to_read=None):
    t0 = time.time()
    logging.info('Reading CSV dataset {}'.format(filename))

    if columns_to_read is not None:
        dataset_df = pd.read_csv(filename, header=header_row, dtype=dtypes, usecols=columns_to_read)
    else:
        dataset_df = pd.read_csv(filename, header=header_row, dtype=dtypes)

    logging.info('Reading complete. time_to_read={:.2f} sec'.format(time.time() - t0))

    return dataset_df  # This is a Pandas DataFrame


def write_to_hdf(df, filename, key, compression_level, mode='a', format='fixed'):
    logging.info('Writing dataset to HDF5 format. filename={}'.format(filename))
    t0 = time.time()

    df.to_hdf(filename, key=key, mode=mode, complevel=compression_level, complib='zlib', format=format)

    logging.info('Writing complete. time_to_write={}'.format(time.time() - t0))


def load_datasets(files_list, header_row=0, strip_col_name_spaces=False, dtypes=None, columns_to_read=None):
    def strip_whitespaces(str):
        return str.strip()

    logging.info('Loading datasets in files')

    dfs = []
    for filename in files_list:
        df = read_csv(filename, header_row, dtypes=dtypes, columns_to_read=columns_to_read)
        dfs.append(df)

    all_data = pd.concat(dfs, ignore_index=True)

    if strip_col_name_spaces:
        all_data.rename(columns=strip_whitespaces, inplace=True)

    logging.info('Loading datasets complete')
    return all_data


def count_labels(label_col):
    label_counts = label_col.value_counts(dropna=False)
    total_count = sum(label_counts)

    logging.info("Total count = {}".format(total_count))

    label_percentages = label_counts / total_count

    return label_counts, label_percentages


def split_dataset(X, y, split_rates, random_seed=None):
    assert sum(split_rates) == 1

    X_2 = X
    y_2 = y
    result_splits = []

    for i, split in enumerate(split_rates[:-1]):  # Must not split at the last element
        remain_rate_sum = sum(split_rates[i:])
        remain_rate = 1 - (split / remain_rate_sum)
        # print("i={}, split={}, remain_rate={}, remain_rate_sum={}".format(i, split, remain_rate, remain_rate_sum))

        # Split into 2 parts, X_1 and X_2
        X_1, X_2, y_1, y_2 = train_test_split(X_2, y_2, stratify=y_2, test_size=remain_rate, random_state=random_seed)
        result_splits.append((X_1, y_1))

    result_splits.append((X_2, y_2))  # Final remaining part

    return result_splits


def initial_setup(output_dir, params):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Setup logging
    log_filename = output_dir + "/" + "run_log.log"

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_filename, "w+"), logging.StreamHandler()],
        level=logging.INFO
    )
    logging.info("Initialized logging. log_filename = {}".format(log_filename))

    logging.info("Running script with following parameters\n{}".format(pformat(params.__dict__)))


def print_dataset_sizes(datasets):
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = datasets
    logging.info("No. of features = {}".format(X_train.shape[1]))
    logging.info("Training examples = {}".format(X_train.shape[0]))
    logging.info("Validation examples = {}".format(X_val.shape[0]))
    logging.info("Test examples = {}".format(X_test.shape[0]))


def prepare_ids2017_datasets(params):
    # Load data
    logging.info("Loading datasets")
    data_files_list = [params.ids2017_datasets_dir + "/" + filename for filename in params.ids2017_files_list]
    all_data = load_datasets(data_files_list, header_row=0, strip_col_name_spaces=True)

    # Remove unicode values in class labels
    logging.info("Converting unicode labels to ascii")
    all_data["Label"] = all_data["Label"].apply(lambda x: x.encode("ascii", "ignore").decode("utf-8"))
    all_data["Label"] = all_data["Label"].apply(lambda x: re.sub(" +", " ", x))  # Remove double spaces

    # Following type conversion and casting (both) are necessary to convert the values in cols 14, 15 detected as objects
    # Otherwise, the training algorithm does not work as expected
    logging.info("Converting object type in columns 14, 15 to float64")
    all_data["Flow Bytes/s"] = all_data["Flow Bytes/s"].apply(lambda x: np.float64(x))
    all_data["Flow Packets/s"] = all_data["Flow Packets/s"].apply(lambda x: np.float64(x))
    all_data["Flow Bytes/s"] = all_data["Flow Bytes/s"].astype(np.float64)
    all_data["Flow Packets/s"] = all_data["Flow Packets/s"].astype(np.float64)

    # Remove some invalid values/ rows in the dataset
    # nan_counts = all_data.isna().sum()
    # logging.info(nan_counts)
    logging.info("Removing invalid values (inf, nan)")
    prev_rows = all_data.shape[0]
    all_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    all_data.dropna(inplace=True)  # Some rows (1358) have NaN values in the Flow Bytes/s column. Get rid of them
    logging.info("Removed no. of rows = {}".format(prev_rows - all_data.shape[0]))

    # Remove samples from classes with a very small no. of samples (cannot split with those classes)
    logging.info("Removing instances of rare classes")
    rare_classes = ["Infiltration", "Web Attack Sql Injection", "Heartbleed"]
    all_data.drop(all_data[all_data["Label"].isin(rare_classes)].index, inplace=True)  # Inplace drop

    # Check class labels
    label_counts, label_perc = count_labels(all_data["Label"])
    logging.info("Label counts below")
    logging.info("\n{}".format(label_counts))

    logging.info("Label percentages below")
    pd.set_option("display.float_format", "{:.4f}".format)
    logging.info("\n{}".format(label_perc))

    X = all_data.loc[:, all_data.columns != "Label"]  # All columns except the last
    y = all_data["Label"]

    # Take only 8% as the small subset
    if params.ids2017_small:
        logging.info("Splitting dataset into 2 (small subset, discarded)")
        splits = split_dataset(X, y, [0.08, 0.92])
        (X, y), (discarded, discarded) = splits
        logging.info("Small subset no. of examples = {}".format(X.shape[0]))

    # Split into 3 sets (train, validation, test)
    logging.info("Splitting training set into 3 (train, validation, test)")
    splits = split_dataset(X, y, [0.6, 0.2, 0.2])
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = splits

    # Save data files in HDF format
    logging.info("Saving prepared datasets (train, val, test) to: {}".format(params.output_dir))

    write_to_hdf(X_train, params.output_dir + "/" + "X_train.h5", params.hdf_key, 5)
    write_to_hdf(y_train, params.output_dir + "/" + "y_train.h5", params.hdf_key, 5)

    write_to_hdf(X_val, params.output_dir + "/" + "X_val.h5", params.hdf_key, 5)
    write_to_hdf(y_val, params.output_dir + "/" + "y_val.h5", params.hdf_key, 5)

    write_to_hdf(X_test, params.output_dir + "/" + "X_test.h5", params.hdf_key, 5)
    write_to_hdf(y_test, params.output_dir + "/" + "y_test.h5", params.hdf_key, 5)

    logging.info("Saving complete")

    print_dataset_sizes(splits)


def extract_databases():
    dataset_zips = ["GeneratedLabelledFlows", "MachineLearningCSV"]
    for dataset_zip_name in dataset_zips:
        if not os.path.isdir(dataset_zip_name):
            with zipfile.ZipFile(dataset_zip_name + ".zip", "r") as zip_ref:
                zip_ref.extractall(dataset_zip_name)


def main():
    extract_databases()

    initial_setup(params.output_dir, params)

    prepare_ids2017_datasets(params)  # Small subset vs. full is controlled by config flag

    logging.info("Data preparation complete")


if __name__ == "__main__":
    # Script params
    params = Params()

    # Common params
    params.hdf_key = "cic_ids_2017"

    # IDS 2017 params
    params.ids2017_small = len(sys.argv) < 2 or sys.argv[1] != "--full"
    # params.ids2017_small = False

    if params.ids2017_small:
        params.output_dir = "cic_ids_2017_small"
    else:
        params.output_dir = "cic_ids_2017_full"

    params.ids2017_datasets_dir = "MachineLearningCSV/MachineLearningCVE"
    params.ids2017_files_list = [
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Monday-WorkingHours.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",  # Issue with flows file
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Wednesday-workingHours.pcap_ISCX.csv"
    ]

    params.ids2017_hist_num_bins = 10000

    params.ids2017_flows_dir = "GeneratedLabelledFlows/TrafficLabelling"
    params.ids2017_flow_seqs_max_flow_seq_length = 100
    params.ids2017_flow_seqs_max_flow_duration_secs = 3
    main()
