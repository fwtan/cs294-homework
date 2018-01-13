import sys
import gym.spaces
import itertools
import numpy as np
import random
from collections import namedtuple
from dqn_utils import *
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class QNet(nn.Module):
    def __init__(self, nc, na):
        super(QNet, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(nc, 32, kernel_size=8, stride=4, padding=4), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            Flatten(),
            nn.Linear(7744, 512),
            nn.ReLU(),
            nn.Linear(512, na)
        )

    def forward(self, input):
        return self.main(input)


OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])


def train_qnet(opt, env, optimizer_spec,
        exploration=LinearSchedule(1000000, 0.1),
        stopping_criterion=None,
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.99,
        learning_starts=50000,
        learning_freq=4,
        frame_history_len=4,
        target_update_freq=10000,
        grad_norm_clipping=10):
    
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete

    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = env.observation_space.shape
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_shape = (img_h, img_w, frame_history_len * img_c)
    num_actions = env.action_space.n

    qnet = QNet(input_shape[-1], num_actions)
    optimizer = optimizer_spec.constructor(qnet.parameters(), lr=optimizer_spec.lr_schedule.value(0), **optimizer_spec.kwargs)
    if opt.cuda:
        qnet = qnet.cuda()
    tnet = deepcopy(qnet)
    if opt.cuda:
        tnet = tnet.cuda()
    
    # construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    ###############
    # RUN ENV     #
    ###############
    num_param_updates = 0
    mean_episode_reward      = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 10000

    for t in itertools.count():
        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env, t):
            break

        #####
        cur_idx = replay_buffer.store_frame(last_obs)
        if (np.random.random() < exploration.value(t)):
            cur_act = np.random.randint(0, num_actions)
        else:
            cur_obs = replay_buffer.encode_recent_observation()
            cur_obs_th = torch.from_numpy(cur_obs[None,:].transpose((0, 3, 1, 2))).float()/255.0
            if opt.cuda:
                cur_obs_th = cur_obs_th.cuda()
            cur_qvs_th = tnet(Variable(cur_obs_th))
            cur_qvs = cur_qvs_th.cpu().data.numpy()
            cur_act = np.argmax(cur_qvs[0])
        nxt_obs, cur_rew, cur_done, cur_info = env.step(cur_act)
        replay_buffer.store_effect(cur_idx, cur_act, cur_rew, cur_done)
        last_obs = env.reset() if cur_done else nxt_obs
        #####

        if (t > learning_starts and
            t % learning_freq == 0 and
            replay_buffer.can_sample(batch_size)):
            
            obs_t_batch, act_t_batch, rew_t_batch, obs_tp1_batch, done_t_mask = \
                replay_buffer.sample(batch_size)
            
            cur_obs_th = torch.from_numpy(obs_t_batch.transpose((0, 3, 1, 2))).float()/255.0
            nxt_obs_th = torch.from_numpy(obs_tp1_batch.transpose((0, 3, 1, 2))).float()/255.0
            act_th = torch.from_numpy(act_t_batch)
            rew_th = torch.from_numpy(rew_t_batch).float()
            msk_th = torch.from_numpy(done_t_mask).float()
            if opt.cuda:
                cur_obs_th, nxt_obs_th = cur_obs_th.cuda(), nxt_obs_th.cuda()
                act_th, rew_th, msk_th = act_th.cuda(), rew_th.cuda(), msk_th.cuda()

            targets = tnet(Variable(nxt_obs_th))
            targets, _ = torch.max(targets, -1)
            targets = rew_th + gamma * targets.data * (1.0 - msk_th)

            qvals = qnet(Variable(cur_obs_th))
            preds = qvals.gather(-1, Variable(act_th.unsqueeze(-1).long())).squeeze(-1)
            criterion = nn.SmoothL1Loss()
            loss = criterion(preds, Variable(targets))
            loss.backward()
            nn.utils.clip_grad_norm(qnet.parameters(), grad_norm_clipping)
            lr = optimizer_spec.lr_schedule.value(t)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            # print(optimizer.state_dict())
            optimizer.step()
            error = loss.data[0]

            num_param_updates += 1

            if num_param_updates % target_update_freq == 0:
                tnet = deepcopy(qnet)
                if opt.cuda:
                    tnet = tnet.cuda()
                num_param_updates = 0

            

            #####

        ### 4. Log progress
        episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
        if t % LOG_EVERY_N_STEPS == 0:
            print("Timestep %d" % (t,))
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            print("learning_rate %f" % optimizer_spec.lr_schedule.value(t))
            sys.stdout.flush()
