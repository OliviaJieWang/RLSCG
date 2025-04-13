
import platform
import torch

print("Python architecture:", platform.machine())
print("Torch version:", torch.__version__)

import os



import matplotlib
# matplotlib.use('Agg')
import os.path
import numpy as np
from read_data import *
import Parameters 
from DQN import *
from test import *
from read_data import *
import random



seed_ = Parameters.seed
np.random.seed(seed_)
random.seed(seed_)



epsilon_ = Parameters.epsilon
decaying_epsilon_ = Parameters.decaying_epsilon
gamma_ = Parameters.gamma
alpha_ = Parameters.alpha_obj_weight
max_episode_num_ = Parameters.max_episode_num
min_epsilon_ = Parameters.min_epsilon
min_epsilon_ratio_ = Parameters.min_epsilon_ratio
capacity_ =  Parameters.capacity
hidden_dim_ = Parameters.hidden_dim
batch_size_ = Parameters.batch_size
epochs_ = Parameters.epochs
embedding_size_ = Parameters.embedding_size
cons_num_features_ = Parameters.cons_num_features
vars_num_features_ = Parameters.vars_num_features
learning_rate_ = Parameters.lr

display_ = True

model_index_ = Parameters.model_index

#### training and saving the data for plotting and model weights (weights and data are saved inside .learning)
schedule_train_name = "Name_files/Scheduled_train.txt"

DQN = DQNAgent(env = None, capacity = capacity_, hidden_dim = hidden_dim_, batch_size = batch_size_, epochs = epochs_, embedding_size = embedding_size_, 
			   cons_num_features = cons_num_features_, vars_num_features = vars_num_features_, learning_rate = learning_rate_)


# DQN_RLCG = DQNAgent(env = None, capacity = capacity_, hidden_dim = hidden_dim_, batch_size = batch_size_, epochs = epochs_, embedding_size = embedding_size_,
# 			   cons_num_features = cons_num_features_, vars_num_features = vars_num_features_, learning_rate = learning_rate_)


import warnings
TRAIN =True

if TRAIN:

	warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
	total_times, episode_rewards, num_episodes =  DQN.learning(schedule_train_name, epsilon = epsilon_, decaying_epsilon = decaying_epsilon_, gamma = gamma_, 
	                learning_rate = learning_rate_, max_episode_num = max_episode_num_, display = display_, min_epsilon = min_epsilon_, min_epsilon_ratio = min_epsilon_ratio_,model_index = model_index_)    



TEST = True
if TEST:

	### here, what only matters is the parameters weight
	DQN_test = DQNAgent(env = None, capacity = capacity_, hidden_dim = hidden_dim_, batch_size = batch_size_, epochs = epochs_, embedding_size = embedding_size_,
				   cons_num_features = cons_num_features_, vars_num_features = vars_num_features_, learning_rate = learning_rate_)

	DQN_test_RLCG = DQNAgent(env=None, capacity=capacity_, hidden_dim=hidden_dim_, batch_size=batch_size_, epochs=epochs_, embedding_size= embedding_size_,
						cons_num_features= 2, vars_num_features=9, learning_rate=learning_rate_)

	adaptive_alpha0 =0.5 # 0.5
	DATA = general_compare(DQN_test,DQN_test_RLCG,0,50, adaptive_alpha0)


