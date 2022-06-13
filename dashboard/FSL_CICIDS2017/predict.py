import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional
import utility


def predict():
    model = torch.load("models/model_2022_06_13:00_30_30")
    model.eval()

    test_x, test_y = utility.load_train_datasets("../../datasets/CIC_IDS_2017/cic_ids_2017_prepared_21")

    n_way = 5
    n_support = 5
    n_query = 1

    (support_set, s_true_labels), (query_set, q_true_labels), class_labels = utility.extract_sample(n_way, n_support, 0, test_x, test_y)
    model.set_encoded_prototypes(support_set, s_true_labels)

    model.n_way = n_way
    model.n_support = n_support
    model.n_query = n_query

    # TODO:: Get validation dataset here then use it! For now i will use extract_sample function
    (test_sample, s_true_labels), (query_set, q_true_labels), class_labels = utility.extract_sample(n_way, n_support, n_query, test_x, test_y)
    print(model.predict(query_set.view(-1, 1, 78)))


if __name__ == '__main__':
    predict()
