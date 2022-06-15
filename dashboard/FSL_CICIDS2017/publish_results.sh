#!/bin/bash
dt=$(date '+%Y_%m_%d--%H_%M_%S');
hostInfo=$(hostname)
tar -cvf experiment_$dt.tar.gz run_experiments.log runs history models individual_logs run_experiments_logs $1
mv experiment_$dt.tar.gz ../../experiments
git pull
git add ../../experiments/experiment_$dt.tar.gz
git commit -m "Added experiment $dt from $hostInfo"
git push
