# -*- coding: utf-8 -*-

##############################################################
# test.py
# Copyright (C) 2018 Tsubasa Hirakawa. All rights reserved.
##############################################################


import os
import multiprocessing as mp

from MaxEntIRL import MaxEntIRL
from util import chunk_list, read_text



BASE_FILE = "./data/basename.txt"
TRAJECTORY_PATH = "./data/tracking"
FEATURE_MAP_FILE = "./data/feature_map/feature_map_3d.npy"
IMAGE_FILE = "./data/image/image2.png"
RESULT_DIR = "./RESULT"



def optimal_control_single_thread(input_base_list, input_weight_path):
    for base in input_base_list:
        optimal_control_one_sample(base, input_weight_path)



def optimal_control_one_sample(input_base, input_weight_path):
    model = MaxEntIRL()
    model.load_trajectory(TRAJECTORY_PATH + input_base + ".npy")
    model.load_reward_weights(input_weight_path)
    model.load_features(FEATURE_MAP_FILE)
    model.load_image(IMAGE_FILE)

    model.compute_reward()

    # soft value function
    model.compute_soft_value_function(os.path.join(RESULT_DIR, "value_%s.npy" % input_base))

    # policy
    model.compute_policy(os.path.join(RESULT_DIR, "policy_%s.npy" % input_base))

    # forecast distribution
    model.compute_forecast_distribution(os.path.join(RESULT_DIR, "prob_%s.npy" % input_base))

    # probability map
    model.map_probability(os.path.join(RESULT_DIR, "result_%s.png" % input_base))



if __name__ == '__main__':

    trained_weight_file = "./RESULT/weight.txt"
    n_cpu = mp.cpu_count()

    if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)

    base_list = read_text(BASE_FILE)
    split_base_list = chunk_list(base_list, (len(base_list) / n_cpu) + 1)

    threads = []
    for b_list in split_base_list:
        threads.append(mp.Process(target=optimal_control_single_thread,
                                  args=(b_list, trained_weight_file)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
