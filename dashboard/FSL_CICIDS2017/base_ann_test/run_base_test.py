import glob
import logging
import os
import shutil
import sys

import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score

import utility
from attack_classifier import AttackClassifier


def load_configurations(config_file_path):
    txt_content = ""
    with open(config_file_path, 'r') as f:
        for line in f:
            txt_content += line

    config = eval(txt_content)
    return config['exp_config'], config['classifier_config']


def create_result_dir(results_dir):
    os.makedirs(results_dir, exist_ok=True)
    logging.info('Created result directory: {}'.format(results_dir))


def load_datasets(data_dir, hdf_key="cic_ids_2017"):
    x_train_files = glob.glob(data_dir + "/" + "x_meta_train*")
    y_train_files = glob.glob(data_dir + "/" + "y_meta_train*")
    x_test_files = glob.glob(data_dir + "/" + "x_meta_test*")
    y_test_files = glob.glob(data_dir + "/" + "y_meta_test*")
    x_val_files = glob.glob(data_dir + "/" + "x_meta_val*")
    y_val_files = glob.glob(data_dir + "/" + "y_meta_val*")

    x_train_files.sort()
    y_train_files.sort()
    x_test_files.sort()
    y_test_files.sort()
    x_val_files.sort()
    y_val_files.sort()

    assert len(x_train_files) > 0
    assert len(y_train_files) > 0

    x_train_files.extend(x_test_files)
    y_train_files.extend(y_test_files)
    x_train_dfs = [pd.read_hdf(file, hdf_key) for file in x_train_files]
    y_train_dfs = [pd.read_hdf(file, hdf_key) for file in y_train_files]
    x_val_dfs = [pd.read_hdf(file, hdf_key) for file in x_val_files]
    y_val_dfs = [pd.read_hdf(file, hdf_key) for file in y_val_files]

    x_train = pd.concat(x_train_dfs)
    x_val = pd.concat(x_val_dfs)
    y_train = pd.concat(y_train_dfs)
    y_val = pd.concat(y_val_dfs)

    return (x_train, y_train), (x_val, y_val)


def evaluate_intrusion_detector(classifier, X_test, y_test, label_encoder):
    y_test_pred_enc = classifier.predict_classes(X_test)
    y_test_pred = label_encoder.inverse_transform(y_test_pred_enc.flatten())

    accuracy = accuracy_score(y_test, y_test_pred)
    f1_weighted = f1_score(y_test, y_test_pred, average='weighted')
    report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
    report.pop('accuracy')
    report.pop('macro avg')
    report.pop('weighted avg')

    general_info = {'accuracy': accuracy, 'f1_score': f1_weighted}
    detailed_info = {}

    for class_name in report:
        class_report = report[class_name]
        detailed_info[class_name] = class_report['f1-score']

    evaluation_info = (general_info, detailed_info)
    return evaluation_info


def write_report(result_dir, evaluation_info):
    general_info, detailed_info = evaluation_info

    general_df = pd.DataFrame(general_info, index=[0])
    detailed_df = pd.DataFrame(detailed_info, index=[0])

    general_df.to_csv(result_dir + '/base_general_eval_info_table.csv', sep='\t')
    detailed_df.to_csv(result_dir + '/base_detailed_eval_info_table.csv', sep='\t')


def run_experiment(exp_config, classifier_config):
    # load dataset
    datasets_orig = load_datasets(exp_config['dataset_dir'])
    (X_train, y_train), (X_test, y_test) = datasets_orig
    y_train_enc, label_encoder = utility.encode_data(y_train)

    classifier = AttackClassifier(classifier_config)
    history = classifier.fit(X_train, y_train_enc)
    evaluation_info = evaluate_intrusion_detector(classifier, X_test, y_test, label_encoder)
    write_report(exp_config['results_dir'], evaluation_info)
    utility.plot_training_history(history, exp_config['results_dir'])
    utility.save_training_history(history, exp_config['results_dir'])


def main():
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)

    config_file_path = sys.argv[1]
    # config_file_path = "config.txt"
    exp_config, classifier_config = load_configurations(config_file_path)

    create_result_dir(exp_config['results_dir'])
    shutil.copy(config_file_path, exp_config['results_dir'])

    run_experiment(exp_config, classifier_config)


if __name__ == "__main__":
    main()
