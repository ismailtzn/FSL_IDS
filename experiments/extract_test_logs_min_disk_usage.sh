#!/usr/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

rm -rf merged
mkdir merged
mkdir merged/runs
mkdir merged/csvs

mkdir tmp
for i in *.tar.gz
do
  tar -xjf $i -C tmp
  cp -rf tmp/runs/* merged/runs
  cp tmp/history/*.csv merged/csvs
  rm -rf tmp/*
done
rm -rf tmp

python3 "$SCRIPT_DIR/export_to_csv.py"

# clear
rm -rf csvs experiment_*/
