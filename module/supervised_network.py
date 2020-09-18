# -*- coding: utf-8 -*-
import os

import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

class SupervisedNetwork(torch.nn.Module):
    def __init__(self, rbm, learning_rate=5e-5):
        super(SupervisedNetwork, self).__init__()
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

    def forward(self, state):
        x = torch.sigmoid(self.linear1(state))
        x = torch.sigmoid(self.linear2(x))
        return x

    def f(self, x):
        x = F.relu(x)
        return x
    
    def get_loss(self, state, next_inds, alls):
        # alls - computes gradient across the entire parameters
        states = torch.from_numpy(state).float()
        scores = self.forward(Variable(states))
        if alls:
            action_probs = scores
            for i in range(len(next_inds)):
                state[i,next_inds[i]] = 1
            log_loss = -torch.mean(torch.FloatTensor(state)*torch.log(action_probs))
        else:
            action_probs = scores[torch.arange(scores.size(0)), next_inds]
            log_loss = -torch.mean(torch.log(action_probs))
        return log_loss

    def minimize_loss(self, log_loss):    
        policy_gradient = log_loss
        self.optimizer.zero_grad()
        policy_gradient.backward()
        self.optimizer.step()
        
    def train(self, train_interaction, epochs, batch_size, alls):
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(0, len(train_interaction)-batch_size, batch_size):
                interaction = train_interaction[i:i+batch_size]
                state = np.array([interaction[j][0] for j in range(len(interaction))])/5
                next_inds = np.array([[interaction[j][1]] for j in range(len(interaction))])
                loss = self.get_loss(state, next_inds, alls)
                if torch.isnan(loss):
                    print("NaN Loss")
                    break
                epoch_loss += loss
            print('Epoch Loss (epoch=%d): %.4f' % (epoch, epoch_loss))
            
    def train_epoch_first(self, train_interaction, epochs, batch_size, alls):
        for i in range(0, len(train_interaction)-batch_size, batch_size):
            interaction = train_interaction[i:i+batch_size]
            state = np.array([interaction[j][0] for j in range(len(interaction))])/5
            next_inds = np.array([interaction[j][1] for j in range(len(interaction))])
            for epoch in range(epochs):
                loss = self.get_loss(state, next_inds, alls)
                if torch.isnan(loss):
                    print("NaN Loss")
                    break
            
    def save_model(self, modelpath):
        torch.save(self.state_dict(), os.path.join(modelpath, "supervied_net.pt"))
        
    def load_model(self, modelpath, modelname):
        self.load_state_dict(torch.load(os.path.join(modelpath, modelname)))