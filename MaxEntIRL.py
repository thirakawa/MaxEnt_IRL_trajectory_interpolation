# -*- coding: utf-8 -*-

##############################################################
# MaxEntIRL.py
# Copyright (C) 2018 Tsubasa Hirakawa. All rights reserved.
##############################################################

import sys
import numpy as np
import cv2



class MaxEntIRL:

    def __init__(self):
        self.w = None
        self.feature_map = []
        self.reward = []
        self.V = []
        self.pax = []
        self.D = []
        self.img = []
        self.start = []
        self.end = []
        self.trajectory = []
        self.FLOAT_MAX = 1e30
        self.FLOAT_MIN = 1e-30
        self.state_size = []
        self.n_action = 9
        self.max_length= 610


    def load_trajectory(self, input_filename, verbose=False):
        if verbose:
            print "load terminal points..."
        self.trajectory = np.load(input_filename)
        self.start = self.trajectory[0, :]
        self.end = self.trajectory[-1, :]
        if verbose:
            print "    length of the trajectory:", self.trajectory.shape[0]
            print "    start:", self.start
            print "    end:", self.end
            print ""


    def load_reward_weights(self, input_filename, verbose=False):
        if verbose:
            print "load reward weights..."
        self.w = np.loadtxt(input_filename)
        if verbose:
            print " number of weights loaded:", self.w.shape[0]
            print ""


    def load_features(self, input_filename, verbose=False):
        if verbose:
            print "load features..."

        self.feature_map = np.load(input_filename)

        if self.feature_map.shape[0] != self.w.shape[0]:
            print "ERROR: weight and feature have different length."
            sys.exit(-1)
        self.n_feature = self.feature_map.shape[0]
        self.state_size = self.feature_map[0].shape
        if verbose:
            print "    number of feature maps loaded:", self.feature_map.shape[0]
            print "    state space size:", self.state_size
            print ""


    def load_image(self, input_filename, verbose=False):
        if verbose:
            print "load image..."
        self.img = cv2.imread(input_filename, 1)
        if verbose:
            print "    done"
            print ""


    def compute_reward(self, bias=0.0, verbose=False):
        if verbose:
            print "compute reward value..."
        self.reward = np.zeros(self.state_size, dtype=np.float32)
        for i in range(self.n_feature):
            self.reward += self.w[i] * self.feature_map[i]
        self.reward += float(bias)
        if verbose:
            print "    done"
            print ""


    def compute_soft_value_function(self, output_filename=None, verbose=False):
        if verbose:
            print "compute soft value function..."
        v = [np.ones(self.state_size, dtype=np.float32) * -self.FLOAT_MAX,
             np.ones(self.state_size, dtype=np.float32) * -self.FLOAT_MAX]

        # init goal
        v[0][self.end[0], self.end[1], self.end[2]] = 0.0

        is_searched = np.zeros(self.state_size, dtype=np.bool)
        is_searched[self.end[0], self.end[1], self.end[2]] = True

        backward_mask = self.make_backward_mask()

        v_padded = np.ones([self.state_size[0] + 2, self.state_size[1] + 2, self.state_size[2] + 2],
                           dtype=np.float32) * -self.FLOAT_MAX

        n = 0
        b_time = self.end[2]
        while True:
            b_time -= 1

            v_padded[1:1 + self.state_size[0], 1:1 + self.state_size[1], 1:1 + self.state_size[2]] = v[0].copy()

            if b_time >= 0:
                is_searched[:, :, b_time] = backward_mask[:, :, b_time].copy()

            for y in range(3):
                for x in range(3):
                    if y == 1 and x == 1:
                        continue
                    minv = np.minimum(v[0], v_padded[y:y + self.state_size[0],
                                            x:x + self.state_size[1],
                                            2:2 + self.state_size[2]])
                    maxv = np.maximum(v[0], v_padded[y:y + self.state_size[0],
                                            x:x + self.state_size[1],
                                            2:2 + self.state_size[2]])

                    softmax = maxv + np.log(1.0 + np.exp(minv - maxv))

                    v[0][is_searched] = softmax[is_searched]

            v[0][is_searched] += self.reward[is_searched]

            if np.sum(v[0][is_searched] > 0) > 0:
                print "    ERROR: elements of V[0] should be lower that 0."
                sys.exit(-1)

            # init goal
            v[0][self.end[0], self.end[1], self.end[2]] = 0.0

            # convergence criteria
            residual = np.abs(v[0] - v[1])
            maxval = np.max(residual[backward_mask])
            v[1] = v[0].copy()

            if maxval < 0.9:
                break

            n += 1
            # max iteration
            if n > 1000:
                print "    WARNING: max number of iterations", n
                break

        self.V = v[0].copy()

        # line break and save value map
        print ""
        if output_filename is not None:
            np.save(output_filename, self.V)


    def compute_policy(self, output_filename=None, verbose=False):
        if verbose:
            print "compute policy..."

        policy = []
        for i in range(self.n_action):
            policy.append(np.zeros(self.state_size, dtype=np.float32))

        self.V[self.V <= -self.FLOAT_MAX] = -np.inf
        v_padded = np.ones([self.state_size[0] + 2, self.state_size[1] + 2, self.state_size[2] + 2],
                           dtype=np.float32) * -np.inf
        v_padded[1:1 + self.state_size[0], 1:1 + self.state_size[1], 1:1 + self.state_size[2]] = self.V.copy()

        for t in range(self.state_size[2]):
            if t == self.end[2]:
                break

            sub = np.zeros([self.n_action, self.state_size[0], self.state_size[1]], dtype=np.float32)

            sub[0, :, :] = v_padded[0:0 + self.state_size[0], 0:0 + self.state_size[1], t + 2].copy()
            sub[1, :, :] = v_padded[0:0 + self.state_size[0], 1:1 + self.state_size[1], t + 2].copy()
            sub[2, :, :] = v_padded[0:0 + self.state_size[0], 2:2 + self.state_size[1], t + 2].copy()
            sub[3, :, :] = v_padded[1:1 + self.state_size[0], 0:0 + self.state_size[1], t + 2].copy()
            sub[4, :, :] = v_padded[1:1 + self.state_size[0], 1:1 + self.state_size[1], t + 2].copy()
            sub[5, :, :] = v_padded[1:1 + self.state_size[0], 2:2 + self.state_size[1], t + 2].copy()
            sub[6, :, :] = v_padded[2:2 + self.state_size[0], 0:0 + self.state_size[1], t + 2].copy()
            sub[7, :, :] = v_padded[2:2 + self.state_size[0], 1:1 + self.state_size[1], t + 2].copy()
            sub[8, :, :] = v_padded[2:2 + self.state_size[0], 2:2 + self.state_size[1], t + 2].copy()

            max_values = np.max(sub, axis=0)
            is_not_inf_mask = np.logical_not(np.isinf(max_values))

            p = sub.copy()
            p[:, is_not_inf_mask] = sub[:, is_not_inf_mask] - max_values[is_not_inf_mask]
            p = np.exp(p)
            p[4, :, :] = 0.0

            summation = np.sum(p, axis=0)
            is_pos = summation > 0

            for i in range(self.n_action):
                policy[i][is_pos, t] = p[i, is_pos] / summation[is_pos]
                if i == 4:
                    policy[i][np.logical_not(is_pos), t] = 0.0
                else:
                    policy[i][np.logical_not(is_pos), t] = 1.0 / (self.n_action - 1.0)

        # save policy map
        if verbose:
            print "    done"
            print ""

        self.pax = np.array(policy)

        if output_filename is not None:
            np.save(output_filename, self.pax)


    def compute_forecast_distribution(self, output_filename=None, verbose=False):
        if verbose:
            print "compute forecast distribution..."

        self.D = np.zeros(self.state_size, dtype=np.float32)
        N = [self.D.copy(), self.D.copy()]

        N[0][self.start[0], self.start[1], self.start[2]] = 1.0

        col = self.state_size[0]
        row = self.state_size[1]
        length = self.state_size[2]

        border = self.make_border_mask()

        forward_mask = self.make_forward_mask()
        backward_mask = self.make_backward_mask()
        back_forward_mask = np.logical_and(forward_mask, backward_mask)

        mask = np.zeros(self.state_size, dtype=np.bool)

        n = 0
        while True:
            N[1] *= 0.0

            mask[:, :, n] = back_forward_mask[:, :, n].copy()
            # mask = np.logical_and(mask, back_forward_mask)
            padded_mask = np.lib.pad(mask, 1, 'constant', constant_values=False)

            n_pax_tmp = []
            for i in range(9):
                n_pax_tmp.append(N[0] * self.pax[i, :, :, :])

            # north-west (top-left)
            N[1][padded_mask[2:2 + col, 2:2 + row, 0:length]] += n_pax_tmp[0][np.logical_and(mask, border[0])]
            # north (top)
            N[1][padded_mask[2:2 + col, 1:1 + row, 0:length]] += n_pax_tmp[1][np.logical_and(mask, border[1])]
            # north-east (top-right)
            N[1][padded_mask[2:2 + col, 0:0 + row, 0:length]] += n_pax_tmp[2][np.logical_and(mask, border[2])]
            # west (left)
            N[1][padded_mask[1:1 + col, 2:2 + row, 0:length]] += n_pax_tmp[3][np.logical_and(mask, border[3])]
            # east (right)
            N[1][padded_mask[1:1 + col, 0:0 + row, 0:length]] += n_pax_tmp[5][np.logical_and(mask, border[5])]
            # south-west (bottom-left)
            N[1][padded_mask[0:0 + col, 2:2 + row, 0:length]] += n_pax_tmp[6][np.logical_and(mask, border[6])]
            # south (bottom)
            N[1][padded_mask[0:0 + col, 1:1 + row, 0:length]] += n_pax_tmp[7][np.logical_and(mask, border[7])]
            # south-east (bottom-right)
            N[1][padded_mask[0:0 + col, 0:0 + row, 0:length]] += n_pax_tmp[8][np.logical_and(mask, border[8])]

            # init goal
            N[1][self.end[0], self.end[1], self.end[2]] = 0.0

            self.D += N[1]

            n0_tmp, n1_tmp = N

            N = [n1_tmp, n0_tmp]

            n += 1
            if n > self.end[2]:
                break

        # output prob
        if verbose:
            print "    done"
            print ""

        if output_filename is not None:
            np.save(output_filename, self.D)


    def map_probability(self, output_filename=None, verbose=False):
        print "mapping probability..."
        probability = np.sum(self.D, axis=2)
        dst = self.color_map_cumulative_prob(probability)
        dst[dst < 1] = self.img[dst < 1]
        dst = cv2.addWeighted(dst, 0.5, self.img, 0.5, 0)

        for point in self.trajectory:
            dst[point[0], point[1], 0] = 0
            dst[point[0], point[1], 1] = 0
            dst[point[0], point[1], 2] = 0

        if output_filename is not None:
            cv2.imwrite(output_filename, dst)
        print "    done"
        print ""

    def make_backward_mask(self):
        print "    make backward mask..."
        mask = np.zeros(self.state_size, dtype=np.bool)
        mask[self.end[0], self.end[1], self.end[2]] = True

        neighbour8 = np.ones(9, dtype=np.uint8).reshape((3, 3))
        mask_slice = mask[:, :, self.end[2]].astype(np.uint8).copy()

        for i in reversed(range(self.end[2])):
            # all of elements of slice is 1 (True)
            if np.unique(mask_slice)[0] == 1:
                mask[:, :, 0:i + 1] = 1
                break
            # otherwise
            mask_slice = cv2.dilate(mask_slice, neighbour8, 1)
            mask[:, :, i] = mask_slice.copy()

        return mask

    def make_forward_mask(self):
        print "    make forward mask..."
        mask = np.zeros(self.state_size, dtype=np.bool)
        mask[self.start[0], self.start[1], self.start[2]] = True

        neighbour8 = np.ones(9, dtype=np.uint8).reshape((3, 3))
        mask_slice = mask[:, :, self.start[2]].astype(np.uint8).copy()

        for i in range(self.state_size[2]):
            if np.unique(mask_slice)[0] == 1:
                mask[:, :, i:self.state_size[2]] = 1
                break
            mask[:, :, i] = mask_slice.copy()
            mask_slice = cv2.dilate(mask_slice, neighbour8, 1)

        return mask

    def make_border_mask(self):
        top = np.ones(self.state_size, dtype=np.bool)
        top[0, :, :] = False
        bottom = np.ones(self.state_size, dtype=np.bool)
        bottom[-1, :, :] = False
        left = np.ones(self.state_size, dtype=np.bool)
        left[:, 0] = False
        right = np.ones(self.state_size, dtype=np.bool)
        right[:, -1] = False

        border = []
        border.append(np.logical_and(top, left))  # top-left
        border.append(top.copy())  # top
        border.append(np.logical_and(top, right))  # top-right
        border.append(left.copy())  # left
        border.append(np.ones(self.state_size, dtype=np.bool))  # center (not used)
        border.append(right.copy())  # right
        border.append(np.logical_and(bottom, left))  # bottom-left
        border.append(bottom.copy())  # bottom
        border.append(np.logical_and(bottom, right))  # bottom-right
        return border

    def color_map_cumulative_prob(self, input_src):
        im = input_src.copy()
        hsv = np.zeros([im.shape[0], im.shape[1], 3], dtype=np.uint8)

        minVal = 1e-4
        maxVal = 0.2

        im[im <= minVal] = 0.
        im = (im - minVal) / (maxVal - minVal) * 255.

        # hue
        hsv[:, :, 0] = im.astype(np.uint8).copy()
        # saturation
        im_sat = ((-im.astype(np.float64) / 255.) + 1.0) * 255.0
        hsv[:, :, 1] = im_sat.astype(np.uint8).copy()
        # value
        is_nonzero = cv2.compare(im, 0, cv2.CMP_GT)
        hsv[:, :, 2] = is_nonzero.copy()

        dst = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB_FULL)
        return dst


    # for training ==========================================================
    def compute_empirical_feature_count(self):
        f_empirical = np.zeros(self.n_feature, dtype=np.float32)
        for coord in self.trajectory:
            f_empirical += self.feature_map[:, coord[0], coord[1], coord[2]]
        return f_empirical


    def update_weight(self, input_weight):
        self.w = input_weight


    def compute_trajectory_likelihood(self):
        trans_index = np.array([[0,1,2],
                                [3,-1,5],
                                [6,7,8]], dtype=np.int32)
        ll = 0.0

        for t in range(self.trajectory.shape[0] - 1):
            dx = self.trajectory[t + 1, 1] - self.trajectory[t, 1]
            dy = self.trajectory[t + 1, 0] - self.trajectory[t, 0]

            a = trans_index[dy + 1, dx + 1]

            if a < 0:
                print "ERROR: invalid action %d(%d, %d)" % (t, dx, dy)
                print "preprocess trajectory data property"
                sys.exit(-1)

            val = np.log(self.pax[a][self.trajectory[t, 0], self.trajectory[t, 1], self.trajectory[t, 2]])

            if val < -self.FLOAT_MAX:
                ll = -self.FLOAT_MAX
                break

            ll += val
        return ll


    def accumulate_expected_feature_count(self):
        f_expected_tmp = []
        for f in self.n_feature:
            F = self.D * self.feature_map[f, :, :, :]
            f_expected_tmp.append(np.sum(F))
        return np.array(f_expected_tmp, dtype=np.float32)