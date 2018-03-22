#!/usr/bin/env python

import os, sys
import cv2, json
import time, math
import copy, pickle
import random, pickle
import numpy as np
import os.path as osp
from tqdm import *
import matplotlib.pyplot as plt
from config import get_config

import tensorflow as tf
import tf_util, gym
import load_policy

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

class dataset(Dataset):
    def __init__(self, opt, expert_data):
        self.opt = opt 
        self.observations = expert_data['observations']
        self.actions = expert_data['actions']
    
    def __len__(self):
        return self.actions.shape[0]

    def __getitem__(self, idx):
        return {'obs': self.observations[idx].squeeze().astype(np.float32),\
                'act': self.actions[idx].squeeze().astype(np.float32)}

def get_loader(db, config, shuffle=False):
    return DataLoader(db, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_workers)

class mlp_policy(nn.Module):
    def __init__(self, no, na):
        super(mlp_policy, self).__init__()

        self.no = no
        self.na = na

        self.main = nn.Sequential(
            nn.Linear(no, 128), 
            nn.Tanh(),
            # nn.LeakyReLU(0.2, True),
            # nn.Linear(128, 128), 
            # nn.LeakyReLU(0.2, True),
            # nn.Tanh(),
            nn.Linear(128, na), 
            # nn.Tanh(),
        )
        
    def forward(self, O):
        A = self.main(O)
        return A
        
class model(object):
    def __init__(self, opt):
        self.opt = opt
        self.net = mlp_policy(opt.no, opt.na)
        if opt.cuda:
            self.net = self.net.cuda()

    def train(self):
        ###########   OPTIMIZER   ##########
        criterion = nn.MSELoss()
        # optimizer = optim.Adam(self.net.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        optimizer = optim.SGD(self.net.parameters(), lr=self.opt.lr, momentum=0.9)

        obs = torch.FloatTensor(self.opt.batch_size, self.opt.no)
        act = torch.FloatTensor(self.opt.batch_size, self.opt.na)

        if self.opt.cuda:
            criterion.cuda()
            obs, act = obs.cuda(), act.cuda()

        ##########   MAIN LOOP   ###########
        losses = []
        reward_mean = [] 
        reward_std = []

        prev_obs = None
        new_obs  = None
        expert_data = None
        plt.switch_backend('agg')

        self.net.train()

        for epoch in range(1, self.opt.n_epochs+1):
            ###########   RUN GT POLICY   ##########
            if epoch == 1:
                expert_data = run_expert(self.opt)
                prev_obs = expert_data['observations']
            elif self.opt.dagger:
                prev_obs = np.vstack((prev_obs, new_obs))
                prev_obs = np.unique(prev_obs, axis=0)
                if len(prev_obs) > self.opt.max_trainsamples:
                    indices = np.random.permutation(range(len(prev_obs)))
                    prev_obs = prev_obs[indices,:]
                    prev_obs = prev_obs[:self.opt.max_trainsamples]
                expert_data  = run_expert(self.opt, prev_obs)
            db = dataset(self.opt, expert_data)
            loader = get_loader(db, self.opt, True)

            terr = 0.0
            self.net.train()
            ###########   TRAINING   ##########
            for i, entry in tqdm(enumerate(loader, 1)):
                ############################
                # (0) data loading
                ############################
                obs_cpu = entry['obs']
                act_cpu = entry['act'] 
                batch_size = obs_cpu.size(0)
                if self.opt.cuda:
                    obs_cpu, act_cpu = obs_cpu.cuda(), act_cpu.cuda()
                obs.resize_as_(obs_cpu).copy_(obs_cpu)
                act.resize_as_(act_cpu).copy_(act_cpu)
                obsv, actv = Variable(obs), Variable(act)

                ############################
                # (1) Update network 
                ###########################
                self.net.zero_grad()
                pred = self.net(obsv)
                loss = criterion(pred, actv)
                loss.backward()
                optimizer.step()
                terr += loss.data[0]
                
                if i % 10 == 0:
                    print('[%d/%d][%d/%d] Loss: %.4f.'
                    % (epoch, self.opt.n_epochs, i, len(loader), loss.data[0]))
            terr /= i            
            losses.append(terr)
        
            ###########   SAMPLING   ##########
            # testing mode: for bn layers if there is any.
            self.net.eval()
            env = gym.make(self.opt.envname)
            max_steps = self.opt.max_timesteps or env.spec.timestep_limit
            returns = []
            new_obs = []
            for i in tqdm(range(self.opt.num_rollouts)):
                obs_np = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    obs_np = obs_np[None,:].astype(np.float32)
                    obs_th = torch.from_numpy(obs_np)
                    if self.opt.cuda:
                        obs_th = obs_th.cuda()
                    action = self.net(Variable(obs_th)).data.numpy()
                    obs_np, r, done, _ = env.step(action)
                    new_obs.append(obs_np)
                    totalr += r
                    steps += 1
                    if steps % 100 == 0: 
                        print("Roll-out %d: [%d/%d]"%(i, steps, max_steps))
                    if steps >= max_steps:
                        break
                returns.append(totalr)
            reward_mean.append(np.mean(returns))
            reward_std.append(np.std(returns))
            new_obs = np.array(new_obs)

        fig = plt.figure()
        plt.plot(losses, 'b', label='loss')
        plt.grid(True)
        fig.savefig('train_losses.jpg', bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure()
        plt.errorbar(x=range(1,len(reward_mean)+1), y=reward_mean, yerr = reward_std)
        plt.grid(True)
        fig.savefig('val_rewards.jpg', bbox_inches='tight')
        plt.close(fig)

def run_expert(opt, obs=None):
    policy_fn = load_policy.load_policy(opt.pretrained_policy)
    tf_config = tf.ConfigProto(allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(config=tf_config):
        tf_util.initialize()
        env = gym.make(opt.envname)
        observations = []
        actions = []
        if obs is not None:
            for i in range(len(obs)):
                x = obs[i]
                y = policy_fn(x[None,:])
                observations.append(x)
                actions.append(y)
        else:
            max_steps = opt.max_timesteps or env.spec.timestep_limit
            for i in range(opt.num_rollouts):
                obs = env.reset()
                done = False
                steps = 0
                while not done:
                    action = policy_fn(obs[None,:])
                    observations.append(obs)
                    actions.append(action)
                    obs, r, done, _ = env.step(action)
                    steps += 1
                    if steps % 100 == 0: 
                        print("Roll-out %d: [%d/%d]"%(i, steps, max_steps))
                    if steps >= max_steps:
                        break
        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}
        return expert_data


if __name__ == '__main__':
    ###############   SEEDS   ##################
    config, unparsed = get_config()
    np.random.seed(config.rng_seed)
    random.seed(config.rng_seed)
    torch.manual_seed(config.rng_seed)
    if(config.cuda):
        torch.cuda.manual_seed_all(config.rng_seed)
        cudnn.benchmark = True

    my_policy = model(config)
    my_policy.train()
                