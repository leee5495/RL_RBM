# -*- coding: utf-8 -*-
import os
import pickle

import torch
import numpy as np

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
    datapath = "./data"
    modelpath = "./model"
    
    # open data
    with open(os.path.join(datapath, "train_matrix"), "rb") as fp:
        train_matrix = pickle.load(fp)
    train_matrix = train_matrix.astype(float) / 5
    with open(os.path.join(datapath, "valid_interaction"), "rb") as fp:
        valid_interaction = pickle.load(fp)
    with open(os.path.join(datapath, "train_interaction"), "rb") as fp:
        train_interaction = pickle.load(fp)
        
    # train RBM
    num_visible = train_matrix.shape[1]
    num_hidden = 128
    k = 5
    epochs = 100
    batch_size = 20
    rbm = RBM(num_visible, num_hidden, k)
    rbm.train(train_matrix, epochs, batch_size)
    
    # train policy network
    epochs = 5
    batch_size = 20
    k = 5
    learning_rate = 0.0001
    neg_reward = -0.05
    policy_net = PolicyNetwork(rbm, learning_rate)
    policy_net.train_epoch_first(train_interaction, epochs, batch_size, k, neg_reward)
    
    # validate
    valid_input = np.array([i[0] for i in valid_interaction]).astype(float)
    valid_tensor = torch.FloatTensor(valid_input) / 5
    valid_next_books = [i[1] for i in valid_interaction]
    
    test_output = policy_net.forward(valid_tensor, np.sign(valid_tensor)).detach().numpy()
    test_output = test_output * np.where(valid_tensor==0, 1, 0)
    HR_10 = hit_rate(test_output, valid_next_books, 10)
    HR_25 = hit_rate(test_output, valid_next_books, 25)
    ARHR = arhr(test_output, valid_next_books)
    print("HR_10 : {:.02f}".format(HR_10))
    print("HR_25 : {:.02f}".format(HR_25))
    print("ARHR:   {:.02f}".format(ARHR))
    
    # save model
    policy_net.save_model(modelpath)