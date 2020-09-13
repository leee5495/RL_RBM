# -*- coding: utf-8 -*-
import os
import re
import pickle

import pandas as pd
import numpy as np

if __name__ == "__main__":
    datapath = "./data"
    # open raw data
    with open(os.path.join(datapath, "u.data")) as fin:
        file = fin.read()
    file = re.split('\t|\n|::', file)
    file = np.reshape(np.array(file[:-1]), [-1,4]).astype(int) - [[1,1,0,0]]
    
    # make ratings dataframe and divide to train, interaction, test
    user_item_df = pd.DataFrame(file, columns=['user_id', 'item_id', 'rating', 'time']).sort_values(['time'])
    train_ind = int(user_item_df.shape[0]*0.7)
    interaction_ind = int(user_item_df.shape[0]*0.9)
    train_df = user_item_df.iloc[:train_ind]
    interaction_df = user_item_df.iloc[train_ind:interaction_ind]
    test_df = user_item_df.iloc[interaction_ind:]
    
    # make user-item matrix for train, interaction, test
    num_users = max(user_item_df.user_id.unique().tolist())+1
    num_item = max(user_item_df.item_id.unique().tolist())+1
    rating_matrix = np.zeros((num_users, num_item), dtype=np.int8)
    train_matrix = np.zeros((num_users, num_item), dtype=np.int8)
    train_interaction = []
    temp_test_interaction = []
    
    for group in train_df.groupby('user_id'):  
        user_id = group[0]
        item_ids = group[1].item_id.tolist()
        ratings = group[1].rating.tolist()
        for i in range(len(item_ids)):
            rating_matrix[user_id][item_ids[i]] = ratings[i]
            train_matrix[user_id][item_ids[i]] = ratings[i]
    
    # make interaction data - list of tuples (current_rating_vector, new_rated_item, rating)
    for index, row in interaction_df.iterrows():
        user_id = row['user_id']
        item_id = row['item_id']
        rating = row['rating']
        train_interaction.append((rating_matrix[user_id].copy(), item_id, rating))
        rating_matrix[user_id][item_id] = rating
    
    # save test ratings matrix - ratings up to test inds
    with open(os.path.join(datapath, "rating_matrix"), "wb") as fp:
        pickle.dump(rating_matrix, fp)
    
    # make test interaction data - list of tuples (current_rating_vector, new_rated_item, rating)
    for index, row in test_df.iterrows():
        user_id = row['user_id']
        item_id = row['item_id']
        rating = row['rating']
        temp_test_interaction.append((rating_matrix[user_id].copy(), item_id, rating))
        rating_matrix[user_id][item_id] = rating
            
    valid_interaction = temp_test_interaction[int(len(temp_test_interaction)*0.5):]
    test_interaction = temp_test_interaction[:int(len(temp_test_interaction)*0.5)]
    train_matrix = train_matrix[~np.all(train_matrix == 0, axis=1)]
    
    # save preprocessed data
    with open(os.path.join(datapath, "train_matrix"), "wb") as fp:
        pickle.dump(train_matrix, fp)
    with open(os.path.join(datapath, "train_interaction"), "wb") as fp:
        pickle.dump(train_interaction, fp)
    with open(os.path.join(datapath, "valid_interaction"), "wb") as fp:
        pickle.dump(valid_interaction, fp)
    with open(os.path.join(datapath, "test_interaction"), "wb") as fp:
        pickle.dump(test_interaction, fp)