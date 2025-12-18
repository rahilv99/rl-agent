# PPO-PyTorch

### UPDATE [April 2021] : 

- merged discrete and continuous algorithms
- added linear decaying for the continuous action space `action_std`; to make training more stable for complex environments
- added different learning rates for actor and critic
- episodes, timesteps and rewards are now logged in `.csv` files
- utils to plot graphs from log files
- utils to test and make gifs from preTrained networks
- `PPO_colab.ipynb` combining all the files to train / test / plot graphs / make gifs on google colab in a convenient jupyter-notebook

#### [Open `PPO_colab.ipynb` in Google Colab](https://colab.research.google.com/github/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_colab.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_colab.ipynb)


## Introduction

This repository provides a Minimal PyTorch implementation of Proximal Policy Optimization (PPO) with clipped objective for OpenAI gymnasium environments. It is primarily intended for beginners in [Reinforcement Learning](https://en.wikipedia.org/wiki/Reinforcement_learning) for understanding the PPO algorithm. It can still be used for complex environments but may require some hyperparameter-tuning or changes in the code. A concise explaination of PPO algorithm can be found [here](https://stackoverflow.com/questions/46422845/what-is-the-way-to-understand-proximal-policy-optimization-algorithm-in-rl) and a thorough explaination of all the details for implementing best performing PPO can be found [here](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) (All are not implemented in this repo yet). 


To keep the training procedure simple : 
  - It has a **constant standard deviation** for the output action distribution (**multivariate normal with diagonal covariance matrix**) for the continuous environments, i.e. it is a hyperparameter and NOT a trainable parameter. However, it is **linearly decayed**. (action_std significantly affects performance)
  - It uses simple **monte-carlo estimate** for calculating advantages and NOT Generalized Advantage Estimate (check out the OpenAI spinning up implementation for that).
  - It is a **single threaded implementation**, i.e. only one worker collects experience. [One of the older forks](https://github.com/rhklite/Parallel-PPO-PyTorch) of this repository has been modified to have Parallel workers

## Usage

- To train a new network : run `train.py`
- To test a preTrained network : run `test.py`
- To plot graphs using log files : run `plot_graph.py`
- To save images for gif and make gif using a preTrained network : run `make_gif.py`
- All parameters and hyperparamters to control training / testing / graphs / gifs are in their respective `.py` file
- `PPO_colab.ipynb` combines all the files in a jupyter-notebook
- All the **hyperparameters used for training (preTrained) policies are listed** in the [`README.md` in PPO_preTrained directory](https://github.com/nikhilbarhate99/PPO-PyTorch/tree/master/PPO_preTrained)

#### Note :
  - if the environment runs on CPU, use CPU as device for faster training. Box-2d and Roboschool run on CPU and training them on GPU device will be significantly slower because the data will be moved between CPU and GPU often

## Dependencies
Trained and Tested on:
```
Python 3
PyTorch
NumPy
gymnasium
```
Training Environments 
```
Box-2d
Roboschool
pybullet
```
