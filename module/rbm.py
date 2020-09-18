# -*- coding: utf-8 -*-
import json

import torch
import numpy as np
import torch.nn as nn

class RBM(nn.Module):
    def __init__(self, num_visible, num_hidden, k, learning_rate=1e-2, momentum_coefficient=0.5, weight_decay=1e-4):
        super(RBM, self).__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k
        self.learning_rate = learning_rate
        self.momentum_coefficient = momentum_coefficient
        self.weight_decay = weight_decay

        self.weights = torch.randn(num_visible, num_hidden) * 0.1
        self.visible_bias = torch.ones(num_visible) * 0.5
        self.hidden_bias = torch.zeros(num_hidden)

        self.weights_momentum = torch.zeros(num_visible, num_hidden)
        self.visible_bias_momentum = torch.zeros(num_visible)
        self.hidden_bias_momentum = torch.zeros(num_hidden)

    def sample_hidden(self, visible_probabilities):
        hidden_activations = torch.matmul(visible_probabilities, self.weights) + self.hidden_bias
        hidden_probabilities = self._sigmoid(hidden_activations)
        return hidden_probabilities

    def sample_visible(self, hidden_probabilities):
        visible_activations = torch.matmul(hidden_probabilities, self.weights.t()) + self.visible_bias
        visible_probabilities = self._sigmoid(visible_activations)
        return visible_probabilities

    def contrastive_divergence(self, input_data):
        # Positive phase
        positive_hidden_probabilities = self.sample_hidden(input_data)
        positive_hidden_activations = (positive_hidden_probabilities >= self._random_probabilities(self.num_hidden)).float()
        positive_associations = torch.matmul(input_data.t(), positive_hidden_activations)

        # Negative phase
        hidden_activations = positive_hidden_activations
        for step in range(self.k):
            visible_probabilities = self.sample_visible(hidden_activations)
            hidden_probabilities = self.sample_hidden(visible_probabilities)
            hidden_activations = (hidden_probabilities >= self._random_probabilities(self.num_hidden)).float()

        negative_visible_probabilities = visible_probabilities
        negative_hidden_probabilities = hidden_probabilities
        negative_associations = torch.matmul(negative_visible_probabilities.t(), negative_hidden_probabilities)

        # Update parameters
        self.weights_momentum *= self.momentum_coefficient
        self.weights_momentum += (positive_associations - negative_associations)

        self.visible_bias_momentum *= self.momentum_coefficient
        self.visible_bias_momentum += torch.sum(input_data - negative_visible_probabilities, dim=0)

        self.hidden_bias_momentum *= self.momentum_coefficient
        self.hidden_bias_momentum += torch.sum(positive_hidden_probabilities - negative_hidden_probabilities, dim=0)

        batch_size = input_data.size(0)

        self.weights += self.weights_momentum * self.learning_rate / batch_size
        self.visible_bias += self.visible_bias_momentum * self.learning_rate / batch_size
        self.hidden_bias += self.hidden_bias_momentum * self.learning_rate / batch_size
        self.weights -= self.weights * self.weight_decay  # L2 weight decay

        # Compute reconstruction error
        error = torch.mean((input_data - negative_visible_probabilities)**2)
        return error

    def _sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def _random_probabilities(self, num):
        random_probabilities = torch.rand(num)
        return random_probabilities

    def predict(self,input_data):
        q_h0 = self._sigmoid(torch.matmul(input_data, self.weights) + self.hidden_bias)
        prediction = self._sigmoid(torch.matmul(q_h0, self.weights.t()) + self.visible_bias)
        return prediction
    
    def train(self, train_data, epochs, batch_size):
        train_data = torch.FloatTensor(train_data)
        for epoch in range(epochs):
            epoch_error = 0.0
            for i in range(0, len(train_data), batch_size): 
                vk = train_data[i:i+batch_size]
                batch_error = self.contrastive_divergence(vk)
                epoch_error += batch_error
            print('Epoch Error (epoch=%d): %.4f' % (epoch, epoch_error))
            
    def save(self, sfile):
        state_dict = {}
        state_dict['num_visible'] = self.num_visible
        state_dict['num_hidden'] = self.num_hidden
        state_dict['k'] = self.k
        state_dict['learning_rate'] = self.learning_rate
        state_dict['momentum_coefficient'] = self.momentum_coefficient
        state_dict['weight_decay'] = self.weight_decay

        state_dict['weights'] = self.weights.numpy().tolist()
        state_dict['visible_bias'] = self.visible_bias.numpy().tolist()
        state_dict['hidden_bias'] = self.hidden_bias.numpy().tolist()

        state_dict['weights_momentum'] = self.weights_momentum.numpy().tolist()
        state_dict['visible_bias_momentum'] = self.visible_bias_momentum.numpy().tolist()
        state_dict['hidden_bias_momentum'] = self.hidden_bias_momentum.numpy().tolist()
        
        with open(sfile, 'w') as fp:
            json.dump(state_dict, fp)
            
    def load(self, sfile):
        with open(sfile, 'r') as fp:
            state_dict = json.load(fp)
            
        self.num_visible = state_dict['num_visible'] = self.num_visible
        self.num_hidden = state_dict['num_hidden'] = self.num_hidden
        self.k = state_dict['k'] = self.k
        self.learning_rate = state_dict['learning_rate'] = self.learning_rate
        self.momentum_coefficient = state_dict['momentum_coefficient'] = self.momentum_coefficient
        self.weight_decay = state_dict['weight_decay'] = self.weight_decay

        self.weights = torch.FloatTensor(np.array(state_dict['weights']))
        self.visible_bias = torch.FloatTensor(np.array(state_dict['num_visible']))
        self.hidden_bias = torch.FloatTensor(np.array(state_dict['hidden_bias']))

        self.weights_momentum = torch.FloatTensor(np.array(state_dict['weights_momentum']))
        self.visible_bias_momentum = torch.FloatTensor(np.array(state_dict['visible_bias_momentum']))
        self.hidden_bias_momentum = torch.FloatTensor(np.array(state_dict['hidden_bias_momentum']))