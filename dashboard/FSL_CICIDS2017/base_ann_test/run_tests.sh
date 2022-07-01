#!/usr/bin/bash

i=0
for conf in configs/*
do
	i=$((i+1))
	python3 run_base_test.py $conf > "run_experiments_logs/experiment_$i.log" 2>&1
done

