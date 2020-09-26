# -*- coding: utf-8 -*-
import os
import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt

from module.rbm import RBM
from module.policy_network import PolicyNetwork

def hit_rate(test_output, next_books, k):
    top_k = (-test_output).argsort()[:,:k]
    hit = 0
    total = len(test_output)
    for i in range(len(test_output)):
        if next_books[i] in top_k[i]:
            hit += 1
    return hit/total
    
def arhr(test_output, next_books):
    temp = -test_output
    order = temp.argsort()
    ranks = order.argsort()+1
    arhr = np.sum(1/ranks[np.arange(len(next_books)),next_books])/len(next_books)
    return arhr

if __name__ == "__main__":
    datapath = "../data"
    modelpath = "../model"
    
    # open data
    with open(os.path.join(datapath, "train_matrix"), "rb") as fp:
        train_matrix = pickle.load(fp)
    train_matrix = train_matrix.astype(float) / 5
    with open(os.path.join(datapath, "valid_interaction"), "rb") as fp:
        valid_interaction = pickle.load(fp)
    with open(os.path.join(datapath, "train_interaction"), "rb") as fp:
        train_interaction = pickle.load(fp)
    with open(os.path.join(datapath, "test_interaction"), "rb") as fp:
        test_interaction = pickle.load(fp)
        
    test_input = np.array([i[0] for i in test_interaction]).astype(float)
    test_tensor = torch.FloatTensor(test_input) / 5
    test_next_books = [i[1] for i in test_interaction]
        
    # train RBM
    num_visible = train_matrix.shape[1]
    num_hidden = 128
    k = 5
    epochs = 100
    batch_size = 20
    rbm = RBM(num_visible, num_hidden, k)
    rbm.train(train_matrix, epochs, batch_size)
    
    # train policy network with different k
    epochs = 5
    batch_size = 20
    learning_rate = 0.0001
    neg_reward = -0.05
    ks = [2, 4, 6, 8, 10]
    
    policy_nets = {}
    HR10 = []
    HR25 = []
    ARHR = []
    
    for k in ks:
        policy_net = PolicyNetwork(rbm, learning_rate)
        policy_net.train_epoch_first(train_interaction, epochs, batch_size, k, neg_reward)
        policy_nets[k] = policy_net
        
        prediction = policy_net.forward(test_tensor, np.sign(test_tensor)).detach().numpy()
        prediction = prediction * np.where(test_tensor==0, 1, 0)
        
        HR10.append(hit_rate(prediction, test_next_books, 10))
        HR25.append(hit_rate(prediction, test_next_books, 25))
        ARHR.append(arhr(prediction, test_next_books))
        
    # graph results
    fig = plt.figure(1)
    axes = plt.gca()
    plt.plot(ks, HR10)         
    plt.xlabel('hit rate')
    plt.ylabel('k')
    plt.legend(['HR10'])
    
    fig = plt.figure(2)
    axes = plt.gca()
    plt.plot(ks, HR25)         
    plt.xlabel('hit rate')
    plt.ylabel('k')
    plt.legend(['HR25'])
    
    fig = plt.figure(3)
    axes = plt.gca()
    plt.plot(ks, ARHR)         
    plt.xlabel('arhr')
    plt.ylabel('k')
    plt.legend(['ARHR'])