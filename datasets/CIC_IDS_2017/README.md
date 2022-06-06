Before you run "prepare_data.py", do not forget to download following zip files
from http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/

* MachineLearningCSV.zip
* GeneratedLabelledFlows.zip -> No need now

Command line arguments:

* --hdf_key default => "cic_ids_2017"
* --output_dir_prefix default => "cic_ids_2017_prepared"
* --sample_per_class default => 21
* --meta_test_class_count default => 5
* --ids2017_datasets_dir default => "MachineLearningCSV/MachineLearningCVE"