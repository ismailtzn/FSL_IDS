import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional
import utility


def predict():
    model = torch.load("latest_model")
    model.eval()

    test_x, test_y = utility.load_train_datasets("../../datasets/CIC_IDS_2017/cic_ids_2017_prepared_150")

    n_way = 4
    n_support = 5
    n_query = 5
    one_sample = utility.extract_sample(n_way, n_support, n_query, test_x, test_y)
    one_sample_loss, one_sample_output = model.set_forward_loss(one_sample)


if __name__ == '__main__':
    predict()
