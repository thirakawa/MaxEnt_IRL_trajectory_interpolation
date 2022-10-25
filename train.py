# -*- coding: utf-8 -*-

##############################################################
# train.py
# Copyright (C) 2018 Tsubasa Hirakawa. All rights reserved.
##############################################################


import os
import time
import math
import numpy as np
import multiprocessing as mp

from MaxEntIRL import MaxEntIRL
from util import chunk_list, read_text


BASE_FILE = "./data/basename.txt"
BASE_FILE_2 = "./data/basename2.txt"
TRAJECTORY_PATH = "./data/tracking"
FEATURE_MAP_FILE = "./data/feature_map/feature_map_3d.npy"
IMAGE_FILE = "./data/image/image2.png"
RESULT_DIR = "./RESULT"
CACHE_DIR = "./CACHE"


class Trainer:

    def __init__(self, input_basename_list):

        self.FLOAT_MAX = 1e30
        self.FLOAT_MIN = 1e-30
        self.n_cpu = mp.cpu_count()

        self.basename_list = input_basename_list
        self.split_base_list = chunk_list(
            self.basename_list,
            int(len(self.basename_list) / self.n_cpu) + 1
        )
        self.n_feature = np.load(FEATURE_MAP_FILE).shape[0]
        self.n_data = len(input_basename_list)

        self.w = np.ones(self.n_feature, dtype=np.float32) * 0.5
        self.w_best = []

        # empirical feature count
        self.f_empirical = np.zeros(self.n_feature, dtype=np.float32)
        self.f_expected = np.zeros(self.n_feature, dtype=np.float32)
        self.f_gradient = np.zeros(self.n_feature, dtype=np.float32)
        self.f_gradient_best = []

        self.loglikelihood = 0.0
        self.min_loglikelihood = -self.FLOAT_MAX
        self.lam = 0.01
        self.DELTA = 0.01
        self.converged = False
        self.pid = os.getpid()

        # compute empirical feature count
        for bname in self.basename_list:
            tmp_model = MaxEntIRL()
            tmp_model.load_trajectory(os.path.join(TRAJECTORY_PATH, f"{bname}.npy"))
            tmp_model.update_weight(self.w)
            tmp_model.load_features(FEATURE_MAP_FILE)
            tmp_model.load_image(IMAGE_FILE)
            self.f_empirical += tmp_model.compute_empirical_feature_count()
        self.f_empirical /= self.n_feature
        print("empirical feature count:", self.f_empirical)

        # make cache directory
        if not os.path.exists(CACHE_DIR):
            os.mkdir(CACHE_DIR)

    def backward_forward_pass(self):

        # thread = []
        # mp.set_start_method('fork')
        # for th_i, b_list in enumerate(self.split_base_list):
            # thread.append(mp.Process(target=self.back_forward_single_thread, args=(b_list, self.w, th_i)))

        # for t in thread:
            # t.start()
        # for t in thread:
            # t.join()

        self.loglikelihood = 0.0
        self.f_expected *= 0.0
        # for th_i, t in enumerate(thread):
        for th_i in range(0, 7):
            ll_tmp = np.load(os.path.join(CACHE_DIR, f"{62137}-{th_i}-ll.npy"))
            f_exp_tmp = np.load(os.path.join(CACHE_DIR, f"{62137}-{th_i}-fexp.npy"))

            self.loglikelihood += np.sum(ll_tmp)
            self.f_expected += np.sum(f_exp_tmp, axis=0)

        self.loglikelihood /= float(self.n_data)
        self.f_expected /= float(self.n_data)

    def back_forward_single_thread(self, basename, weight, thread_index):

        loglikelihood_tmp = []
        f_expected_list = []

        for bn in basename:
            print(bn)
            _start = time.time()

            model = MaxEntIRL()
            model.load_trajectory(os.path.join(TRAJECTORY_PATH, bn + ".npy"))
            model.update_weight(weight)
            model.load_features(FEATURE_MAP_FILE)
            model.load_image(IMAGE_FILE)

            model.compute_reward()

            model.compute_soft_value_function()
            model.compute_policy()
            model.compute_forecast_distribution()

            loglikelihood_tmp.append(model.compute_trajectory_likelihood())
            f_expected_list.append(model.accumulate_expected_feature_count())

            _end = time.time()

            print("done. time", _end - _start)

        # save
        np.save(os.path.join(CACHE_DIR, "%d-%d-ll.npy" % (self.pid, thread_index)), np.array(loglikelihood_tmp))
        np.save(os.path.join(CACHE_DIR, "%d-%d-fexp.npy" % (self.pid, thread_index)), np.array(f_expected_list))

    def gradient_update(self):

        improvement = self.loglikelihood - self.min_loglikelihood

        if improvement > self.DELTA:
            self.min_loglikelihood = self.loglikelihood
        elif -self.DELTA < improvement < self.DELTA:
            improvement = 0

        print("improved by", improvement)

        # update parameters
        if improvement < 0:
            print("NO IMPROVEMENT: decrease step size and redo")
            self.lam = self.lam * 0.5

            for f in range(self.n_feature):
                self.w[f] = self.w_best[f] * math.exp(self.lam * self.f_gradient[f])

        elif improvement > 0:
            print("IMPROVEMENT: increase step size")
            self.w_best = self.w.copy()
            self.lam = self.lam * 2.0

            for f in range(self.n_feature):
                self.f_gradient[f] = self.f_empirical[f] - self.f_expected[f]
            for f in range(self.n_feature):
                self.w[f] = self.w_best[f] * math.exp(self.lam * self.f_gradient[f])

        elif improvement == 0:
            print("CONVERGED")
            self.converged = True

        print("lambda:", self.lam)
        print("f_empirical:", self.f_empirical)
        print("f_expected:", self.f_expected)

    def save_parameter(self, output_filename):
        np.savetxt(output_filename, self.w)


if __name__ == '__main__':

    if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)

    with open(BASE_FILE) as base_file:
        basename_list = [line.strip() for line in base_file]

    trainer = Trainer(basename_list)

    iteration = 0
    while not trainer.converged:
        start = time.time()

        trainer.backward_forward_pass()
        trainer.gradient_update()
        trainer.save_parameter(os.path.join(RESULT_DIR, "weight-%03d.txt" % iteration))
        iteration += 1

        end = time.time()
        print("time of this iteration:", end - start, "s")

    trainer.save_parameter(os.path.join(RESULT_DIR, "weight.txt"))
    print("train: done.")
