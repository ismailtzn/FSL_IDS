#!/usr/bin/env python3
import os
import sys
import logging
import time
from pprint import pformat
import pandas as pd
import numpy as np
import re
import zipfile


def read_csv(filename, header_row=0, dtypes=None, columns_to_read=None):
    t0 = time.time()
    logging.info("Reading CSV dataset {}".format(filename))

    if columns_to_read is not None:
        dataset_df = pd.read_csv(filename, header=header_row, dtype=dtypes, usecols=columns_to_read)
    else:
        dataset_df = pd.read_csv(filename, header=header_row, dtype=dtypes)

    logging.info("Reading complete. time_to_read={:.2f} sec".format(time.time() - t0))

    return dataset_df


def write_to_hdf(df, filename, key, compression_level, mode="a", fmt="fixed"):
    logging.info("Writing dataset to HDF5 format. filename={}".format(filename))
    t0 = time.time()

    df.to_hdf(filename, key=key, mode=mode, complevel=compression_level, complib="zlib", format=fmt)

    logging.info("Writing complete. time_to_write={}".format(time.time() - t0))


def load_datasets(files_list, header_row=0, dtypes=None, columns_to_read=None):
    logging.info("Loading datasets in files")

    dfs = []
    for filename in files_list:
        df = read_csv(filename, header_row, dtypes=dtypes, columns_to_read=columns_to_read)
        dfs.append(df)

    all_data = pd.concat(dfs, ignore_index=True)

    all_data.columns = all_data.columns.str.strip()
    all_data.columns = all_data.columns.str.replace(" ", "_")

    logging.info("Loading datasets complete")
    return all_data


def count_labels(label_col):
    label_counts = label_col.value_counts(dropna=False)
    total_count = sum(label_counts)

    logging.info("Total count = {}".format(total_count))

    label_percentages = label_counts / total_count

    return label_counts, label_percentages


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

    logging.info("Running script with following parameters\n{}".format(pformat(params)))


def split_datasets(all_data, params):
    # ignore labels with small number of datapoint
    df = all_data.groupby("Label").filter(lambda x: len(x) > params["data_per_label"]).copy()
    df = df.groupby("Label", group_keys=False).apply(lambda x: x.sample(params["data_per_label"]))

    remaining_df = all_data.drop(df.index)  # save for later usages
    train_sample_count_per_label = int(params["data_per_label"] * 0.6)
    test_sample_count_per_label = int(params["data_per_label"] * 0.2)

    train_df = df.groupby("Label", group_keys=False).apply(lambda x: x.sample(train_sample_count_per_label)).copy()
    df.drop(train_df.index, inplace=True)
    test_df = df.groupby("Label", group_keys=False).apply(lambda x: x.sample(test_sample_count_per_label)).copy()
    df.drop(test_df.index, inplace=True)
    validation_df = df

    x_train = train_df.loc[:, train_df.columns != "Label"]
    y_train = train_df["Label"]

    x_test = test_df.loc[:, test_df.columns != "Label"]
    y_test = test_df["Label"]

    x_validation = validation_df.loc[:, validation_df.columns != "Label"]
    y_validation = validation_df["Label"]

    logging.info("Train set {}".format(train_df["Label"].value_counts()))
    logging.info("Test set :\n{}".format(test_df["Label"].value_counts()))
    logging.info("Validation set :\n{}".format(validation_df["Label"].value_counts()))
    logging.info("Validation set :\n{}".format(validation_df["Label"].value_counts()))

    return (x_train, y_train), (x_test, y_test), (x_validation, y_validation), remaining_df


def prepare_ids2017_datasets(params):
    # Load data
    logging.info("Loading datasets")
    data_files_list = [params["ids2017_datasets_dir"] + "/" + filename for filename in params["ids2017_files_list"]]
    all_data = load_datasets(data_files_list, header_row=0)

    # Remove unicode values in class labels
    logging.info("Converting unicode labels to ascii")
    all_data["Label"] = all_data["Label"].apply(lambda x: x.encode("ascii", "ignore").decode("utf-8"))
    all_data["Label"] = all_data["Label"].apply(lambda x: re.sub(" +", " ", x))  # Remove double spaces

    # Following type conversion and casting (both) are necessary to convert the values in cols 14, 15 detected as objects
    # Otherwise, the training algorithm does not work as expected
    logging.info("Converting object type in columns 14, 15 to float64")
    all_data["Flow_Bytes/s"] = all_data["Flow_Bytes/s"].apply(lambda x: np.float64(x))
    all_data["Flow_Packets/s"] = all_data["Flow_Packets/s"].apply(lambda x: np.float64(x))
    all_data["Flow_Bytes/s"] = all_data["Flow_Bytes/s"].astype(np.float64)
    all_data["Flow_Packets/s"] = all_data["Flow_Packets/s"].astype(np.float64)

    logging.info("Removing invalid values (inf, nan)")
    prev_rows = all_data.shape[0]
    all_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    all_data.dropna(inplace=True)  # Some rows (1358) have NaN values in the Flow Bytes/s column. Get rid of them
    logging.info("Removed no. of rows = {}".format(prev_rows - all_data.shape[0]))

    # Check class labels
    label_counts, label_perc = count_labels(all_data["Label"])
    logging.info("Label counts below")
    logging.info("\n{}".format(label_counts))

    logging.info("Label percentages below")
    pd.set_option("display.float_format", "{:.4f}".format)
    logging.info("\n{}".format(label_perc))

    (x_train, y_train), (x_test, y_test), (x_validation, y_validation), remaining_df = split_datasets(all_data, params)
    # Save data files in HDF format
    logging.info("Saving prepared datasets (train, val, test) to: {}".format(params["output_dir"]))

    write_to_hdf(x_train, params["output_dir"] + "/" + "x_train.h5", params["hdf_key"], 5)
    write_to_hdf(y_train, params["output_dir"] + "/" + "y_train.h5", params["hdf_key"], 5)

    write_to_hdf(x_test, params["output_dir"] + "/" + "x_test.h5", params["hdf_key"], 5)
    write_to_hdf(y_test, params["output_dir"] + "/" + "y_test.h5", params["hdf_key"], 5)

    write_to_hdf(x_validation, params["output_dir"] + "/" + "x_validation.h5", params["hdf_key"], 5)
    write_to_hdf(y_validation, params["output_dir"] + "/" + "y_validation.h5", params["hdf_key"], 5)

    write_to_hdf(remaining_df, params["output_dir"] + "/" + "remaining_df.h5", params["hdf_key"], 5)

    logging.info("Saving complete")

    logging.info("No. of features = {}".format(x_train.shape[1]))
    logging.info("{} Labels used = \n{}".format(len(y_train.unique()), ", ".join([str(x) for x in y_train.unique()])))
    logging.info("Training examples = {}".format(x_train.shape[0]))
    logging.info("Test examples = {}".format(x_test.shape[0]))
    logging.info("Validation examples = {}".format(x_validation.shape[0]))


def extract_databases():
    # dataset_zips = ["GeneratedLabelledFlows", "MachineLearningCSV"]
    dataset_zips = ["MachineLearningCSV"]
    for dataset_zip_name in dataset_zips:
        if not os.path.isdir(dataset_zip_name):
            with zipfile.ZipFile(dataset_zip_name + ".zip", "r") as zip_ref:
                zip_ref.extractall(dataset_zip_name)


def main(params):
    extract_databases()
    initial_setup(params["output_dir"], params)
    prepare_ids2017_datasets(params)
    logging.info("Data preparation complete")


if __name__ == "__main__":
    parameters = {"hdf_key": "cic_ids_2017", "output_dir": "cic_ids_2017_prepared", "data_per_label": 1500}

    if len(sys.argv) > 1 and sys.argv[1].isnumeric():
        parameters["data_per_label"] = int(sys.argv[1])
    parameters["output_dir"] = parameters["output_dir"] + "_" + str(parameters["data_per_label"])

    parameters["ids2017_datasets_dir"] = "MachineLearningCSV/MachineLearningCVE"
    parameters["ids2017_files_list"] = [
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Monday-WorkingHours.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",  # Issue with flows file
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Wednesday-workingHours.pcap_ISCX.csv"
    ]

    main(parameters)
