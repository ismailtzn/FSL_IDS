#!/usr/bin/python3
import numpy as np
import itertools

# ./test_parameter_generator.py > new_tests/test_parameters.txt
# split -l 32 new_tests/test_parameters.txt --numeric-suffixes --additional-suffix=.txt new_tests/test_parameters_
if __name__ == '__main__':
    template = "--experiment_dir_prefix {experiment_dir_prefix} --meta_train_n_way {meta_train_n_way} --meta_train_k_shot {meta_train_k_shot} --meta_train_query_count {meta_train_query_count} --meta_train_max_epoch {meta_train_max_epoch} --meta_train_epoch_size {meta_train_epoch_size} --dataset_dir {dataset_dir} --model_x_dim0 {model_x_dim0} --model_x_dim1 {model_x_dim1} --model_hid_dim {model_hid_dim} --model_z_dim {model_z_dim} --learning_rate {learning_rate} --learning_rate_decay {learning_rate_decay} --meta_test_n_way {meta_test_n_way} --meta_test_k_shot {meta_test_k_shot} --meta_test_query_count {meta_test_query_count} --meta_test_episode_count {meta_test_episode_count} "

    experiment_dir_prefix = ["prototypical"]
    meta_train_n_way_range = [9]
    meta_train_k_shot_range = [5]
    meta_train_query_count_range = range(5, 16, 5)
    meta_train_max_epoch_range = range(20, 41, 20)
    meta_train_epoch_size_range = range(1000, 5001, 4000)
    dataset_dir = ["../../datasets/CIC_IDS_2017/cic_ids_2017_prepared_21"],
    model_x_dim0_range = [1]
    model_x_dim1_range = [78]
    model_hid_dim_range = range(64, 129, 64)
    model_z_dim_range = range(64, 129, 64)
    learning_rate_range = np.arange(0.001, 0.01, 0.005)
    learning_rate_decay_range = np.arange(0.50, 0.751, 0.25)
    meta_test_n_way_range = [5]
    meta_test_k_shot_range = [5]
    meta_test_query_count_range = [5]
    meta_test_episode_count_range = [1000]

    for experiment_dir_prefix, meta_train_n_way, meta_train_k_shot, meta_train_query_count, meta_train_max_epoch, meta_train_epoch_size, dataset_dir, model_x_dim0, model_x_dim1, model_hid_dim, \
        model_z_dim, learning_rate, learning_rate_decay, meta_test_n_way, meta_test_k_shot, meta_test_query_count, meta_test_episode_count in itertools.product(experiment_dir_prefix,
                                                                                                                                                                meta_train_n_way_range,
                                                                                                                                                                meta_train_k_shot_range,
                                                                                                                                                                meta_train_query_count_range,
                                                                                                                                                                meta_train_max_epoch_range,
                                                                                                                                                                meta_train_epoch_size_range,
                                                                                                                                                                dataset_dir,
                                                                                                                                                                model_x_dim0_range,
                                                                                                                                                                model_x_dim1_range,
                                                                                                                                                                model_hid_dim_range,
                                                                                                                                                                model_z_dim_range,
                                                                                                                                                                learning_rate_range,
                                                                                                                                                                learning_rate_decay_range,
                                                                                                                                                                meta_test_n_way_range,
                                                                                                                                                                meta_test_k_shot_range,
                                                                                                                                                                meta_test_query_count_range,
                                                                                                                                                                meta_test_episode_count_range
                                                                                                                                                                ):
        generated = template.format(
            experiment_dir_prefix=experiment_dir_prefix,
            meta_train_n_way=meta_train_n_way,
            meta_train_k_shot=meta_train_k_shot,
            meta_train_query_count=meta_train_query_count,
            meta_train_max_epoch=meta_train_max_epoch,
            meta_train_epoch_size=meta_train_epoch_size,
            dataset_dir=dataset_dir,
            model_x_dim0=model_x_dim0,
            model_x_dim1=model_x_dim1,
            model_hid_dim=model_hid_dim,
            model_z_dim=model_z_dim,
            learning_rate=learning_rate,
            learning_rate_decay=learning_rate_decay,
            meta_test_n_way=meta_test_n_way,
            meta_test_k_shot=meta_test_k_shot,
            meta_test_query_count=meta_test_query_count,
            meta_test_episode_count=meta_test_episode_count
        )
        print(generated)
