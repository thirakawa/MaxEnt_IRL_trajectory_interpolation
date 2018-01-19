# -*- coding: utf-8 -*-

##############################################################
# MaxEntIRL.py
# Copyright (C) 2018 Tsubasa Hirakawa. All rights reserved.
##############################################################


if __name__ == '__main__':

    import numpy as np

    map_2d = np.load("feature_map_2d.npy")
    n_feature, h, w = map_2d.shape

    map_3d = np.zeros([n_feature, h, w, 610], dtype=np.float16)

    for i in range(map_3d.shape[3]):
        map_3d[:, :, :, i] = map_2d.copy()

    np.save("feature_map_3d.npy", map_3d)
