#!/bin/bash
if [[ $# -ge 1 ]]; then
  input=$1
  lineCount=$(wc -l < $input)
  echo Running test using $input, test count: $lineCount
else
  echo Please provide text file that contains config parameters
  echo Run this script using following command
  echo "nohup ./run_experiments.sh test_parameters.txt > run_experiments.log 2>&1 &"
  exit
fi

rm -rf run_experiments_logs
mkdir run_experiments_logs
i=0
while IFS= read -r line
do
  i=$((i+1))
  echo "[$i/$lineCount] -> ./basic_train.py $line"
  ./basic_train.py $line > "run_experiments_logs/experiment_$i.log" 2>&1 &
done < "$input"
