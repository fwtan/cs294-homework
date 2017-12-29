#!/usr/bin/env python

#-*- coding: utf-8 -*-
import argparse
import os.path as osp

def str2bool(v):
    return v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--no', type=int, default=376)
net_arg.add_argument('--na', type=int, default=17)

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--num_workers', type=int, default=4)

# Training parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--batch_size', type=int, default=128)
train_arg.add_argument('--cuda', type=str2bool, default=False)
train_arg.add_argument('--lr', type=float, default=0.001)
train_arg.add_argument('--n_epochs', type=int, default=20)
train_arg.add_argument('--beta1', type=float, default=0.5)

# Inference parameters
exp_arg = add_argument_group('Expert')
exp_arg.add_argument('--pretrained_policy', type=str)
exp_arg.add_argument('--envname', type=str)
exp_arg.add_argument("--max_timesteps", type=int, default=1000)
exp_arg.add_argument("--max_trainsamples", type=int, default=100000)
exp_arg.add_argument('--num_rollouts', type=int, default=100)

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--rng_seed', type=int, default=0)
misc_arg.add_argument('--epsilon',  type=float, default=1e-5)
misc_arg.add_argument('--dagger',   type=str2bool, default=False)

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed