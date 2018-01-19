# Maximum Entropy Inverse Reinforcement Learning for Trajectory Interpolation


This repository contains example codes of trajectory interpolation method on the basis of maximum entropy inverse reinforcement learning.



## Requirements
- Language: Python (2.7.\*)
- Modules: Numpy, OpenCV, multiprocessing


## Note
- The program requires a relatively higher performance computer to process in realistic computational time. The training takes approximately 3~4 days with 32 multi-thread processing and 256 GB memory.


## Usage
### Making 3D feature map
> cd data/feature_map \
> python make_3d_feature_map.py 


### Training
> python train.py


### Inference (interpolation)
> python test.py



## Reference
1. T. Hirakawa, T. Yamashita, K. Yoda, T. Tamaki, H. Fujiyoshi, "Travel Time-dependent Maximum Entropy Inverse Reinforcement Learning for Seabird Trajectory Prediction," In Proc. of Asian Conference on Pattern Recognition (ACPR2017), pp. 430-435, Nov. 2017.
