#!/bin/bash

i=0
while [ "$i" -lt "$1" ]
do
	i=$(($i+1))
	./run_tests.sh run_base_test_n-way_k-shot.py configs_k-shot
done
