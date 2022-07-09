#!/bin/bash
dt=$(date '+%Y_%m_%d--%H_%M_%S');
hostInfo=$(hostname)
current_dir=$(pwd)
tar -cjvf experiment_$dt.tar.gz run_experiments.log runs history models individual_logs run_experiments_logs $1
mkdir -p ../../experiments/$2
mv experiment_$dt.tar.gz ../../experiments/$2
git pull
git add ../../experiments/$2/experiment_$dt.tar.gz
git commit -m "Added experiment $dt from $hostInfo -- current dir: $current_dir"
git push
