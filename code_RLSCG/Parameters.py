import numpy as np

## seed
seed = 0


## parameters about neural network

batch_size = 32
capacity = 20000
hidden_dim = 32
epochs = 5 ## 
embedding_size = 32
cons_num_features = 3
vars_num_features = 9



## parameters of RL algorithm

epsilon = 0.9
min_epsilon = 1e-2
min_epsilon_ratio = 0.8
decaying_epsilon = True

action_pool_size = 20 ## solution pool
max_episode_num = 440  #439 ## as there are 440 training instances## as there are 440 training instances
capacity = 2000 ## so that the experience is relatively new



## parameter index
model_index = 0




lr = 1e-3     # 1e-3, 3e-4 , 1e-4
gamma = 0.9   # 0.9，0.95，0.99
step_penalty = 100     # 100，10，1
alpha_obj_weight = 300   # 300, 100, 0



