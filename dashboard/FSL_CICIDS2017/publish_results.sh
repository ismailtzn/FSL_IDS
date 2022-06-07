#!/bin/bash
dt=$(date '+%Y_%m_%d--%H_%M_%S');
tar -I 'gzip -9' -cf experiment_$dt.tar.gz run_experiments.log runs models individual_logs run_experiments_logs $1
# tar -c --tape-length=50M --file=experiment_$dt.{00..50}.tar.gz run_experiments.log runs models individual_logs run_experiments_logs $1
mv experiment_$dt.tar.gz ../../experiments
git add ../../experiments/experiment_$dt.tar.gz
git commit -m "Added experiment $dt"
git push
