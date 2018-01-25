# Maximum Entropy Inverse Reinforcement Learning for Trajectory Interpolation

This repository contains example codes of trajectory interpolation method on the basis of maximum entropy inverse reinforcement learning [1].



## Citation
If you find this codes useful, please cite this paper:
* T. Hirakawa, T. Yamashita, T. Tamaki, H. Fujiyoshi, Y. Umezu, I. Takeuchi, S. Matsumoto, K. Yoda, "Can AI predict animal movements? Filling gaps in animal trajectories using Inverse Reinforcement Learning," Ecology (submitted).



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
1. T. Hirakawa, T. Yamashita, T. Tamaki, H. Fujiyoshi, Y. Umezu, I. Takeuchi, S. Matsumoto, K. Yoda, "Can AI predict animal movements? Filling gaps in animal trajectories using Inverse Reinforcement Learning," (in preparation).
