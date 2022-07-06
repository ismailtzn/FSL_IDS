#!/usr/bin/bash
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

rm -rf merged
mkdir merged
for i in *.tar.gz; do
    tar -xjf "$i" --one-top-level -C merged
done

mkdir merged/runs
for i in merged/experiment_*; do
    cp -rf "$i"/runs/* merged/runs
done

mkdir merged/csvs
for i in merged/experiment_*; do
    python3 "$SCRIPT_DIR"/extend_csv_content.py "$i"
    cp "$i"/history/*_experiment_result.csv merged/csvs
done
python3 "$SCRIPT_DIR"/export_to_csv.py

# clear
rm -rf csvs experiment_*/
