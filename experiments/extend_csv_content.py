#!/usr/bin/python3
import os
import sys
import pickle
import numpy as np
import copy
import numbers
import pandas as pd
import csv


def sum_dicts(x, y):
    result = {}
    if x.keys() != y.keys():
        raise ValueError("x and y should be in same dict format")
    for key in x.keys():
        if isinstance(x[key], numbers.Number) and isinstance(y[key], numbers.Number):
            result[key] = x[key] + y[key]
        elif isinstance(x[key], dict) and isinstance(y[key], dict):
            result[key] = sum_dicts(x[key], y[key])
        else:
            raise ValueError("x[key] and y[key] should be in same dict format")
    return result


def multiply_dict_with_number(d, n):
    result = {}
    for key in d.keys():
        if isinstance(d[key], numbers.Number):
            result[key] = d[key] * n
        elif isinstance(d[key], dict):
            result[key] = multiply_dict_with_number(d[key], n)
        else:
            raise ValueError("d[key] should be number")
    return result


def average_history(history):
    avg_metrics = None
    for item in history["metrics"]:
        if avg_metrics is None:
            avg_metrics = copy.deepcopy(item)
            continue
        avg_metrics = sum_dicts(avg_metrics, item)
    avg_metrics = multiply_dict_with_number(avg_metrics, (1 / len(history["metrics"])))
    avg_hist = {
        "metrics": avg_metrics,
        "avg_cf_matrix": np.around(np.average([x for x in history["cf_matrix"]], axis=0), decimals=3),
        "sum_cf_matrix": sum(history["cf_matrix"])
    }
    return avg_hist


if __name__ == "__main__":
    base_dir = sys.argv[1].rstrip("/")
    history_dir = os.path.join(base_dir, "history")
    experiment_ids = []
    for file in os.listdir(history_dir):
        if file.endswith("_experiment_params.csv"):
            experiment_ids.append(file.rstrip("_experiment_params.csv"))

    for exp_id in experiment_ids:
        extended_dfs = []
        for step in ["Test", "Validation"]:
            pkl_file_path = os.path.join(history_dir, "{}_{}.pkl".format(exp_id, step))
            with open(pkl_file_path, "rb") as f:
                history = pickle.load(f)
            avg_history = average_history(history)

            metrics = avg_history["metrics"]
            for item in ["accuracy", "loss", "weighted avg", "macro avg"]:
                metrics.pop(item)

            metrics_df = {"experiment_id": exp_id.lstrip("history_")}
            for key, val in metrics.items():
                metrics_df["Meta{}/{}_f1-score".format(step, key)] = [val["f1-score"]]
            metrics_df = pd.DataFrame(metrics_df)
            metrics_df.set_index("experiment_id", inplace=True)
            extended_dfs.append(metrics_df)

        experiment_params_file_path = os.path.join(history_dir, exp_id + "_experiment_params.csv")
        df = pd.read_csv(experiment_params_file_path, sep="\t")
        df.set_index("experiment_id", inplace=True)
        extended_dfs.insert(0, df)
        df = pd.concat(extended_dfs, axis=1)

        result_file_path = os.path.join(history_dir, exp_id + "_experiment_result.csv")
        df.to_csv(result_file_path, escapechar="", sep="\t", quoting=csv.QUOTE_NONE)
