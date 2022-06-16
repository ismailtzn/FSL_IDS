#!/usr/bin/python3
import logging
import pandas as pd
import glob
import os
import csv

if __name__ == "__main__":
    files = os.path.join("merged/csvs", "*.csv")
    files = glob.glob(files)

    df = pd.concat(map(lambda f: pd.read_csv(f, sep="\t"), files), ignore_index=True)
    df = df.set_index("experiment_id")
    df = df.sort_values(by="MetaTest/AvgAccuracy", ascending=False)
    df.to_csv("combined_results.csv", escapechar="", sep="\t", quoting=csv.QUOTE_NONE)
