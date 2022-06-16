#!/usr/bin/python3
import logging
import pandas as pd
import glob
import os



if __name__ == "__main__":
    files = os.path.join("merged/csvs", "sales*.csv")
    files = glob.glob(files)

    df = pd.concat(map(pd.read_csv, files), ignore_index=True)
    df = df.set_index("experiment_id")
    print(df)
    df.to_csv(combined_results.csv, sep="\t")
