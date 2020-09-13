# Enhanced Collaborative Filtering with Reinforcement Learning
Experimentation code for Enhanced Collaborative Filtering with Reinforcement Learning

![image](https://github.com/leee5495/RL_RBM/blob/master/misc/%EB%8F%84%ED%98%95.png)

### Data
- download data from: [MovieLens 100K Dataset](https://grouplens.org/datasets/movielens/100k/)

### preprocess.py
preprocesses MovieLens 100K data
- `datapath`: path to the downloaded MovieLens 100K data (also saves the preprocessed data)
- output
  - `train_matrix`: user-item rating matrix up to 70% of rating data
  - `rating_matrix`: user-item rating matrix up to 90% of rating data
  - `train_interaction`: interaction data given in list of tuples - (current_rating_vec, next_rated_item, next_rating)
  - `valid_interaction`: interaction data for validation (first half of last 10% of rating data)
  - `test_interaction`: interaction data for test (last half of last 10% of rating data)

### train.py
trains RLRBM using the preprocessed MovieLens 100K data
- `datapath`: path to the preprocessed data
- `modelpath`: path to save the trained policy network
- hyperparameters
  - RBM hyperparameters
    ```
    k = number of contrastive divergence steps
    epochs = number of train epochs
    batch_size = size of each batch
    ```
  - policy network hyperparameters
    ```
    epochs = number of train epochs
    batch_size = size of each batch
    k = number of items to sample for each recommendation action
    learning_rate = learning rate of the optimizer
    neg_reward = reward to give if the next true item is not in the recommended items
    ```
