#!/usr/bin/bash

rm -rf run_experiments_logs
mkdir run_experiments_logs
i=0
for conf in $2/*
do
	i=$((i+1))
	echo "running $i ..."
	python3 $1 $conf > "run_experiments_logs/experiment_$i.log" 2>&1
done
