#!/bin/bash
if [[ $# -ge 2 ]]; then
  scriptFile=$1
  input=$2
  lineCount=$(wc -l < $input)
  echo Running test using $input, test count: $lineCount
else
  echo Please provide text file that contains config parameters
  echo Run this script using following command
  echo "nohup ./run_experiments.sh {test_script} {PARAMETERS_FILE} > run_experiments.log 2>&1 &"
  exit
fi

./clear_logs.sh
mkdir run_experiments_logs

i=0
while IFS= read -r line
do
  i=$((i+1))
  echo "[$i/$lineCount] -> python3 $scriptFile $line"
  python3 $scriptFile $line > "run_experiments_logs/experiment_$i.log" 2>&1
done < "$input"

./publish_results.sh $input $3
