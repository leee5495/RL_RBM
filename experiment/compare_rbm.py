# -*- coding: utf-8 -*-
import os
import pickle

import torch
import numpy as np
from tabulate import tabulate
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
    with open(os.path.join(datapath, "test_interaction"), "rb") as fp:
        test_interaction = pickle.load(fp)
    
    # load model
    num_visible = train_matrix.shape[1]
    num_hidden = 128
    k = 5
    rbm = RBM(num_visible, num_hidden, k)
    policy_net = PolicyNetwork(rbm)
    policy_net.load_model(modelpath, "policy_net.pt")
    
    # validate
    test_input = np.array([i[0] for i in test_interaction]).astype(float)
    test_tensor = torch.FloatTensor(test_input) / 5
    test_next_books = [i[1] for i in test_interaction]
    
    rlrbm_prediction = policy_net.forward(test_tensor, np.sign(test_tensor)).detach().numpy()
    rlrbm_prediction = rlrbm_prediction * np.where(test_tensor==0, 1, 0)
    
    rlrbm_HR10 = hit_rate(rlrbm_prediction, test_next_books, 10)
    rlrbm_HR25 = hit_rate(rlrbm_prediction, test_next_books, 25)
    rlrbm_arhr = arhr(rlrbm_prediction, test_next_books)
    
    # control: rbm with num_hidden = 128
    num_visible = train_matrix.shape[1]
    num_hidden = 128
    k = 5
    epochs = 100
    batch_size = 20
    rbm = RBM(num_visible, num_hidden, k)
    rbm.train(train_matrix, epochs, batch_size)
    
    rbm_prediction = rbm.predict(test_tensor).detach().numpy()
    rbm_prediction = rbm_prediction * np.where(test_tensor==0, 1, 0)
    
    rbm_HR10 = hit_rate(rbm_prediction, test_next_books, 10)
    rbm_HR25 = hit_rate(rbm_prediction, test_next_books, 25)
    rbm_arhr = arhr(rbm_prediction, test_next_books)
    
    # print tabulated result
    table = tabulate([['HR10', rlrbm_HR10, rbm_HR10], 
                      ['HR25', rlrbm_HR25, rbm_HR25],
                      ['ARHR', rlrbm_arhr, rbm_arhr]],
                     headers=['RLRBM', 'RBM'],
                     tablefmt='orgtbl')
    print(table)

    # graph HR on each k
    rlrbm_HR = [hit_rate(rlrbm_prediction, test_next_books, i) for i in range(10)]
    rbm_HR = [hit_rate(rbm_prediction, test_next_books, i) for i in range(10)]
    
    plt.plot(rlrbm_HR)
    plt.plot(rbm_HR)
    plt.ylabel('Hit Rate')
    plt.xlabel('k')
    plt.legend(['RLRBM', 'RBM'])
    plt.show()