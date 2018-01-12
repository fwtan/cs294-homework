import numpy as np
import gym
import logz
import scipy.signal
import os
import time
import inspect
from multiprocessing import Process
from copy import deepcopy
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class PolicyNet(nn.Module):
    def __init__(self, no, na, n_layers, size, discrete):
        super(PolicyNet, self).__init__()

        self.n_layers = n_layers
        self.activation = nn.Tanh()
        if discrete:
            self.output_activation = nn.Softmax(dim=1)
            self.sigma = None
        else:
            self.output_activation = None
            self.sigma = nn.Parameter(torch.rand(1))

        self.first = nn.Linear(no, size)
        self.last  = nn.Linear(size, na)

        for i in range(n_layers-1):
            setattr(self, 'hidden_%02d'%i, nn.Linear(size, size))

    def forward(self, O):
        X = self.first(O)
        X = self.activation(X)
        for i in range(self.n_layers-1):
            X = getattr(self, 'hidden_%02d'%i)(X)
            X = self.activation(X)
        Y = self.last(X)
        if self.output_activation is not None:
            Y = self.output_activation(Y)
        return Y, self.sigma


def pathlength(path):
    return len(path["reward"])


def train_PG(opt):
    start = time.time()

    # Configure output directory for logging
    logz.configure_output_dir(opt.logdir)

    # Log experimental parameters
    logz.save_config(opt)

    # Set random seeds
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.seed)
    
    # Make the gym environment
    env = gym.make(opt.env_name)
    
    # Is this env continuous, or discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Maximum length for episodes
    max_path_length = opt.max_path_length or env.spec.max_episode_steps

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    # Policy net, the underlying model is a mlp
    policy_net = PolicyNet(ob_dim, ac_dim, opt.n_layers, opt.size, discrete)
    optimizer = optim.Adam(policy_net.parameters(), lr=opt.learning_rate)
    
    # Neural network baseline (the mean reward to reduce the variance of the gradient)
    if opt.nn_baseline:
        baseline_net = PolicyNet(ob_dim, 1, opt.n_layers, opt.size, False)
        baseline_optimizer = optim.Adam(baseline_net.parameters(), lr=opt.learning_rate)
     
    total_timesteps = 0
    for itr in range(opt.n_iter):
        print("********** Iteration %i ************"%itr)
        # Collect paths until we have enough timesteps

        # policy net turns to evaluation mode
        policy_net.eval()

        # the batch size of this iteration
        # yes, the batch size for each iteration varies
        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            obs, acs, rewards = [], [], []
            animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and opt.render)
            steps = 0
            while True:
                if animate_this_episode:
                    env.render()
                    time.sleep(0.05)
                
                # collect observations
                obs.append(ob)

                # run the policy net to collect the actions
                obs_th = torch.from_numpy(ob[None,:]).float()
                if opt.cuda:
                    obs_th = obs_th.cuda()
                
                if discrete:
                    probs, _ = policy_net(Variable(obs_th))
                    # multinomial sampling
                    m = torch.distributions.Categorical(probs)
                    ac = m.sample().data.numpy()
                else:
                    mean, sigma = policy_net(Variable(obs_th))
                    ac = sigma * (mean + Variable(torch.randn(mean.size())))
                    ac = ac.data.numpy()
                ac = ac[0]
                acs.append(ac)

                # collect rewards
                ob, rew, done, _ = env.step(ac)
                rewards.append(rew)

                steps += 1
                if done or steps > max_path_length:
                    break

            path = {"observation" : np.array(obs), 
                    "reward" : np.array(rewards), 
                    "action" : np.array(acs)}
            paths.append(path)

            timesteps_this_batch += pathlength(path)

            if timesteps_this_batch > opt.batch_size:
                break
        total_timesteps += timesteps_this_batch


        # Build arrays for observation, action for the policy gradient update by concatenating 
        # across paths
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])

        # discount for reward computation
        gamma = opt.discount
        # trajectory rewards
        q_n = []
        if opt.reward_to_go:
            for path in paths:
                n_samples = len(path["reward"])
                tr = 0.0
                pr = []
                for k in range(n_samples-1, -1, -1):
                    cr = path["reward"][k]
                    tr = gamma * tr + cr
                    pr.append(tr)
                q_n.extend(pr[::-1])
        else:
            for path in paths:
                n_samples = len(path["reward"])
                tr = 0.0
                for k in range(n_samples-1, -1, -1):
                    cr = path["reward"][k]
                    tr = gamma * tr + cr
                pr = np.ones((n_samples,)) * tr
                q_n.extend(pr.tolist())
        q_n = np.array(q_n)

        # If the neural network baseline is used
        # The predicted mean rewards should be subtracted from
        # the Q values
        if opt.nn_baseline:
            baseline_net.eval()
            obs_th = torch.from_numpy(ob_no).float()
            if opt.cuda:
                obs_th = obs_th.cuda()
            b_n, _ = baseline_net(Variable(obs_th))
            b_n = torch.squeeze(b_n, dim=-1)
            b_n = b_n.data.numpy()
            adv_n = q_n - b_n
        else:
            adv_n = q_n.copy()

        # Normalize the advantages
        # an empirical way to reduce the variance of the gradient
        # may or may not help, just an option to try

        if not(opt.dont_normalize_advantages):
            mu = np.mean(adv_n)
            sigma = np.std(adv_n)
            adv_n = (adv_n - mu)/sigma
        
        # Pytorch tensors 
        batch_size = ob_no.shape[0]
        ob_no_th = torch.from_numpy(ob_no).float()
        ac_na_th = torch.from_numpy(ac_na).float()
        adv_n_th = torch.from_numpy(adv_n).float()
        
        if opt.cuda:
            ob_no_th = ob_no_th.cuda()
            ac_na_th = ac_na_th.cuda()
            adv_n_th = adv_n_th.cuda()

        if opt.nn_baseline:
            # train the baseline network
            q_mu = np.mean(q_n)
            q_sigma = np.std(q_n)
            q_n_th = torch.from_numpy((q_n - q_mu)/q_sigma).float() 
            if opt.cuda:
                q_n_th = q_n_th.cuda()

            baseline_net.train()
            baseline_criterion = nn.MSELoss()
            
            baseline_net.zero_grad()
            pred, _ = baseline_net(Variable(ob_no_th))
            pred = torch.squeeze(pred, dim=-1)
            baseline_loss = baseline_criterion(pred, Variable(q_n_th))
            baseline_loss.backward()
            baseline_optimizer.step()
            baseline_error = baseline_loss.data[0]

        policy_net.train()
        policy_net.zero_grad()
        if discrete:
            probs, _ = policy_net(Variable(ob_no_th))
            # multinomial sampling, a biased exploration
            m = torch.distributions.Categorical(probs)
            loss = -m.log_prob(Variable(ac_na_th)) * Variable(adv_n_th)
            loss = loss.mean()
        else:
            means, sigma = policy_net(Variable(ob_no_th))
            diff = (means - Variable(ac_na_th, requires_grad=False)/(sigma+1e-8)) ** 2
            loss = torch.mean(torch.sum(diff, -1))

        loss.backward()
        optimizer.step()
        error = loss.data[0]
            
        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("AverageReturn", np.mean(returns))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        logz.log_tabular("Loss", error)
        if opt.nn_baseline:
            logz.log_tabular("BaselineLoss", baseline_error)
        logz.dump_tabular()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000) # directions?
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.) # length of the episode
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=1)
    parser.add_argument('--size', '-s', type=int, default=32)
    parser.add_argument('--cuda', '-gpu', action='store_true')
    args = parser.parse_args()

    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None
    opt = deepcopy(args)
    opt.max_path_length = max_path_length

    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        opt.seed = seed
        opt.logdir = osp.join(logdir,'%d'%seed)
        print('Running experiment with seed %d'%seed)
        def train_func():
            train_PG(opt)
        p = Process(target=train_func, args=tuple())
        p.start()
        p.join()
        

if __name__ == "__main__":
    main()
