
import random
import numpy as np
from utility import *
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler
from read_data import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch




class Agent(object):
  '''Base Class of Agent
  '''
  def __init__(self, initial_env=None, capacity = 10000):
      self.env = initial_env # the evironment would be one cutting stock object
      ## add the env and available action will be added in the learning_method 
      # self.A = self.env.available_action

      self.A = []
      self.experience = Experience(capacity = capacity)
      # S record the current super state for the agent
      # self.S = self.get_aug_state()   

      self.S = []

  def get_aug_state(self):

      import numpy as np
      from copy import deepcopy
      from sklearn.preprocessing import MinMaxScaler


      current_pai = np.array(self.env.Shadow_Price, dtype=float)
      # pai_in = np.array(self.env.pai_in, dtype=float)
      # pai_diff = current_pai - pai_in


      pai_diff = current_pai - np.array(self.env.previous_pai,
                                        dtype=float) if self.env.previous_pai is not None else np.zeros_like(
          current_pai)



      edge_indices = np.array(self.env.edge_indices)  # 转换为 numpy 数组

      RC_all = np.array(self.env.RC, dtype=float).reshape(-1, 1)
      In_Cons_Num_all = np.array(self.env.In_Cons_Num, dtype=float).reshape(-1, 1)
      ColumnSol_Val_all = np.array(self.env.ColumnSol_Val, dtype=float).reshape(-1, 1)
      stay_in_all = np.array(self.env.stay_in, dtype=float).reshape(-1, 1)
      stay_out_all = np.array(self.env.stay_out, dtype=float).reshape(-1, 1)
      just_left_all = np.array(self.env.just_left, dtype=float).reshape(-1, 1)
      just_enter_all = np.array(self.env.just_enter, dtype=float).reshape(-1, 1)
      associated_alpha_all = np.array(self.env.associated_alpha, dtype=float).reshape(-1, 1)
      mispricing_indicator_all = np.array(self.env.mispricing_indicator, dtype=float).reshape(-1, 1)

      column_features = np.hstack([
          RC_all,
          In_Cons_Num_all,
          ColumnSol_Val_all,
          stay_in_all,
          stay_out_all,
          just_left_all,
          just_enter_all,
          associated_alpha_all,
          mispricing_indicator_all
      ])


      Shadow_Price = np.array(self.env.Shadow_Price, dtype=float).reshape(-1, 1)
      In_Cols_Num = np.array(self.env.In_Cols_Num, dtype=float).reshape(-1, 1)
      pai_diff = pai_diff.reshape(-1, 1)
      cons_features = np.hstack([Shadow_Price, In_Cols_Num, pai_diff])


      def scale_features(data):
          scaler = MinMaxScaler()
          return scaler.fit_transform(data)

      for j in range(column_features.shape[1]):
          column_features[:, j:j + 1] = scale_features(column_features[:, j:j + 1])
      for j in range(cons_features.shape[1]):
          cons_features[:, j:j + 1] = scale_features(cons_features[:, j:j + 1])


      aug_state = (cons_features, edge_indices, column_features)
      return aug_state  # action_info不再需要




  def act(self, chosen_alpha):
      ## get the current super state 
      s0_augmented = self.S



      r, is_done, mispricing = self.env.step(chosen_alpha)

      s1_augmented = self.get_aug_state()
      # total_1 = action_info_1[0]

      trans = Transition(s0_augmented, r, is_done, s1_augmented, chosen_alpha, mispricing)

      # trans = Transition(s0_augmented, r, is_done, s1_augmented, action_info_0, total_0, total_1, chosen_alpha)
      total_reward = self.experience.push(trans)
      self.S = s1_augmented

      return s1_augmented, r, is_done, total_reward,mispricing



  def _decayed_epsilon(self,epsilon,cur_episode: int,
                            min_epsilon: float,
                            minus_epsilon: float,
                            target_episode: int) -> float: 
      # slope = (min_epsilon - max_epsilon) / (target_episode)
      # intercept = max_epsilon

      epsilon_next = max(min_epsilon, epsilon-minus_epsilon)

      # return max(min_epsilon, slope * cur_episode + intercept)
      return epsilon_next
                      


  def sample(self, batch_size = 32):

      return self.experience.sample(batch_size)

  @property
  def total_trans(self):

      return self.experience.total_trans
  
  def last_episode_detail(self):
      self.experience.last_episode.print_detail()



