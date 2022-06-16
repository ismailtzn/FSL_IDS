#!/usr/bin/bash

rm -rf merged
mkdir merged
for i in *.tar.gz
do
  tar -xjf $i --one-top-level -C merged
done

mkdir merged/runs
for i in merged/experiment_*
do
  mv $i/runs/* merged/runs
done

mkdir merged/csvs
for i in merged/experiment_*
do
  mv $i/history/*.csv merged/csvs
done
python3 export_to_csv.py

tensorboard --logdir=merged/runs
