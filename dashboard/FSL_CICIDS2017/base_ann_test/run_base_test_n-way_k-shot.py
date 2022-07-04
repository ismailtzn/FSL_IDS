import glob
import logging
import os
import shutil
import sys

import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from tabulate import tabulate

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


def load_datasets(exp_config, hdf_key="cic_ids_2017"):
    data_dir = exp_config['dataset_dir']
    x_test_files = glob.glob(data_dir + "/" + "x_meta_test*")
    y_test_files = glob.glob(data_dir + "/" + "y_meta_test*")

    x_test_files.sort()
    y_test_files.sort()

    x_test_dfs = [pd.read_hdf(file, hdf_key) for file in x_test_files]
    y_test_dfs = [pd.read_hdf(file, hdf_key) for file in y_test_files]

    x_test_df = pd.concat(x_test_dfs)
    y_test_df = pd.concat(y_test_dfs)

    y_train = y_test_df.groupby(y_test_df).sample(n=exp_config["k"]).copy()
    x_train = x_test_df.loc[y_train.index].copy()

    x_test = x_test_df.drop(x_train.index)
    y_test = y_test_df.drop(y_train.index)


    return (x_train, y_train), (x_test, y_test)


def evaluate_intrusion_detector(classifier, X_test, y_test, label_encoder):
    y_test_pred_enc = classifier.predict_classes(X_test)
    y_test_pred = label_encoder.inverse_transform(y_test_pred_enc.flatten())

    accuracy = accuracy_score(y_test, y_test_pred)
    f1_weighted = f1_score(y_test, y_test_pred, average='weighted')
    report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
    cf_matrix = confusion_matrix(y_test, y_test_pred)

    general_info = {'accuracy': accuracy, 'f1_score': f1_weighted}
    detailed_info = {}

    for class_name in report:
        if class_name in ["accuracy", "macro avg", "weighted avg"]:
            continue
        class_report = report[class_name]
        detailed_info[class_name] = class_report['f1-score']

    evaluation_info = (general_info, detailed_info, cf_matrix, report)
    return evaluation_info


def write_report(result_dir, evaluation_info, classes):
    general_info, detailed_info, cf_matrix, report = evaluation_info
    classes = list(classes)
    with open(result_dir + "/cf_matrix.csv", "a") as f:
        f.write("\n")
        f.write(tabulate(cf_matrix, tablefmt="plain", showindex=classes, headers=classes))

    general_df = pd.DataFrame(general_info, index=[0])
    detailed_df = pd.DataFrame(detailed_info, index=[0])

    general_out_path = result_dir + '/base_general_eval_info_table.csv'
    general_df.to_csv(general_out_path, sep='\t', mode='a', header=not os.path.exists(general_out_path))
    detailed_out_path = result_dir + '/base_detailed_eval_info_table.csv'
    detailed_df.to_csv(detailed_out_path, sep='\t', mode='a', header=not os.path.exists(detailed_out_path))


def run_experiment(exp_config, classifier_config):
    # load dataset
    datasets_orig = load_datasets(exp_config)
    (X_train, y_train), (X_test, y_test) = datasets_orig
    # y_train_enc, label_encoder = utility.encode_data(y_train)
    label_encoder, unused = utility.encode_labels(y_test, encoder=None)
    unused, y_train_enc = utility.encode_labels(y_train, encoder=label_encoder)

    classifier = AttackClassifier(classifier_config)
    history = classifier.fit(X_train, y_train_enc)
    evaluation_info = evaluate_intrusion_detector(classifier, X_test, y_test, label_encoder)
    write_report(exp_config['results_dir'], evaluation_info, label_encoder.classes_)
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
