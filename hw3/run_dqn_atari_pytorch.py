import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import numpy as np

import dqn_pytorch
from dqn_utils import *
from atari_wrappers import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


def atari_learn(opt, env, num_timesteps):
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-4 * lr_multiplier),
                                         (num_iterations / 10, 1e-4 * lr_multiplier),
                                         (num_iterations / 2,  5e-5 * lr_multiplier),
                                    ],
                                    outside_value=5e-5 * lr_multiplier)

    optimizer = dqn_pytorch.OptimizerSpec(
        constructor=optim.Adam,
        kwargs=dict(eps=1e-4),
        lr_schedule=lr_schedule
    )

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (num_iterations / 2, 0.01),
        ], outside_value=0.01
    )
    
    dqn_pytorch.train_qnet(opt,
        env,
        optimizer,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.99,
        learning_starts=50000,
        learning_freq=4,
        frame_history_len=4,
        target_update_freq=10000,
        grad_norm_clipping=10
    )
    env.close()

def get_env(task, seed):
    env_id = task.env_id
    env = gym.make(env_id)
    env.seed(seed)

    expt_dir = '/tmp/hw3_vid_dir2/'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
    env = wrap_deepmind(env)

    return env

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', '-gpu', action='store_true')
    args = parser.parse_args()

    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari40M')

    # Change the index to select a different game.
    task = benchmark.tasks[3]

    # Run training
    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(task, seed)
    atari_learn(args, env, num_timesteps=task.max_timesteps)

if __name__ == "__main__":
    main()
