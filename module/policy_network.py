# -*- coding: utf-8 -*-
import os
import math

import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

class PolicyNetwork(torch.nn.Module):
    def __init__(self, rbm, learning_rate=5e-5):
        super(PolicyNetwork, self).__init__()
        self.input_dim = rbm.weights.shape[0]
        self.latent_dim = rbm.weights.shape[1]
        self.linear1 = torch.nn.Linear(self.input_dim, self.latent_dim)
        self.linear2 = torch.nn.Linear(self.latent_dim, self.input_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.update_weights(rbm)
        
    def update_weights(self, rbm):
        self.linear1.weight.data = torch.t(rbm.weights.data.clone())
        self.linear1.bias.data = rbm.hidden_bias.data.clone()
        self.linear2.weight.data = rbm.weights.data.clone()
        self.linear2.bias.data = rbm.visible_bias.data.clone()

    def forward(self, state, mask):
        x = torch.sigmoid(self.linear1(state))
        x = torch.sigmoid(self.linear2(x))
        x = F.relu(x-mask)
        return x
    
    def sample_k_ind(self, k, dim):
        ind = []
        for i in range(dim):
            temp_ind = []
            while(True):
                new_ind = int(np.random.normal(0, k/2.4))
                if new_ind not in temp_ind and new_ind>=0 and new_ind<self.input_dim:
                    temp_ind.append(new_ind)
                if len(temp_ind) == k:
                    break
            ind.append(temp_ind)
        return np.array(ind)
    
    def f(self, x):
        x = F.relu(x)
        return x

    def get_action(self, state, k):
        mask = torch.FloatTensor(np.sign(state))
        state = torch.from_numpy(state).float()
        scores = self.forward(Variable(state), mask)
        detached_scores = scores.detach().numpy()
        sampled_rank = self.sample_k_ind(k, detached_scores.shape[0])
        sampled_ind = np.take_along_axis((-detached_scores).argsort(), sampled_rank, 1)

        temp = scores.view(scores.shape[0],1,-1)-scores.gather(1,torch.LongTensor(sampled_ind)).view(scores.shape[0],-1,1)
        appx_ranks = torch.sum(temp, 2)
        appx_ranks = appx_ranks - torch.min(appx_ranks, 1, keepdim=True)[0]
        appx_ranks = appx_ranks/torch.max(appx_ranks, 1, keepdim=True)[0]*torch.FloatTensor(sampled_rank)
        appx_probs = (math.sqrt(2)/(math.sqrt(2*math.pi*(k/2.4)**2)))*torch.exp(-(appx_ranks**2)/(2*(k/2.4)**2))
        
        action_probs = torch.prod(appx_probs, 1)
        log_probs = torch.log(action_probs)
        return sampled_ind, log_probs, appx_ranks
    
    def get_reward(self, actions, interaction, neg_reward):
        reward = np.array([interaction[j][2]/5 if interaction[j][1] in actions[j] else neg_reward for j in range(len(actions))])
        return reward
    
    def update_policy(self, reward, log_probs):
        reward = torch.FloatTensor(reward)
        policy_gradient = torch.mean(-log_probs*reward)
        self.optimizer.zero_grad()
        policy_gradient.backward()
        self.optimizer.step()
        return policy_gradient
    
    def train(self, train_interaction, epochs, batch_size, k, neg_reward):
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(0, len(train_interaction)-batch_size, batch_size):
                interaction = train_interaction[i:i+batch_size]
                state = np.array([interaction[j][0] for j in range(len(interaction))])/5
                actions, log_probs, _ = self.get_action(state, k)
                reward = self.get_reward(actions, interaction, neg_reward)
                loss = self.update_policy(reward, log_probs)
                if torch.isnan(loss):
                    print("NaN Loss")
                    break
                epoch_loss += loss
            print('Epoch Loss (epoch=%d): %.4f' % (epoch, epoch_loss))
            
    def train_epoch_first(self, train_interaction, epochs, batch_size, k, neg_reward):
        for i in range(0, len(train_interaction)-batch_size, batch_size):
            for epochs in range(epochs):
                interaction = train_interaction[i:i+batch_size]
                state = np.array([interaction[j][0] for j in range(len(interaction))])/5
                actions, log_probs, _ = self.get_action(state, k)
                reward = self.get_reward(actions, interaction, neg_reward)
                loss = self.update_policy(reward, log_probs)
                if torch.isnan(loss):
                    print("NaN Loss")
                    break
            
    def save_model(self, modelpath):
        torch.save(self.state_dict(), os.path.join(modelpath, "policy_net.pt"))
        
    def load_model(self, modelpath, modelname):
        self.load_state_dict(torch.load(os.path.join(modelpath, modelname)))