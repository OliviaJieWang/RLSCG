
# # DQN.py

import Parameters
import numpy as np
from agents import Agent
from net import BipartiteGNN,BipartiteGNN_RLCG
from copy import deepcopy
import random
import torch
import matplotlib.pyplot as plt
import os
import read_data

class DQNAgent(Agent):
    '''
    '''

    def __init__(self, env,
                 capacity,
                 hidden_dim,
                 batch_size,
                 epochs,
                 embedding_size,
                 cons_num_features,
                 vars_num_features,
                 learning_rate):

        super(DQNAgent, self).__init__(env, capacity)
        self.embedding_size = embedding_size
        self.cons_num_features = cons_num_features
        self.vars_num_features = vars_num_features
        self.lr = learning_rate


        self.behavior_Q = BipartiteGNN(
            embedding_size=self.embedding_size,
            cons_num_features=self.cons_num_features,
            vars_num_features=self.vars_num_features,
            learning_rate=self.lr
        )

        self.target_Q = BipartiteGNN(
            embedding_size=self.embedding_size,
            cons_num_features=self.cons_num_features,
            vars_num_features=self.vars_num_features,
            learning_rate=self.lr
        )
        self._update_target_Q()

        self.batch_size = batch_size
        self.epochs = epochs

        self.alphas = np.linspace(0, 0.95, 20)

        self.target_update_freq = 10  #
        self.train_step_counter = 0

    def _update_target_Q(self):

        self.target_Q.load_state_dict(self.behavior_Q.state_dict())

    def policy(self, s, allowed_alphas,epsilon=None):
        # print(">>> DQNAgent.policy is called!")
        # total_added, Alphas = action_info



        Q_s = self.behavior_Q(s)  # shape [num_vars,1]

        Q_s = Q_s.detach().numpy().flatten()

        if epsilon is not None and np.random.random() < epsilon:
            chosen_alpha = random.choice(self.alphas)
            print('random, alpha',chosen_alpha)
        # else:
        #     print(Q_s)
        #     idx = np.argmax(Q_s)
        #     # print('idx,alphas',idx,self.alphas)
        #     chosen_alpha = self.alphas[idx]
        #
        #     print('Q, alpha',chosen_alpha)
        #
        #
        # return chosen_alpha
        else:

            allowed_indices = [i for i, alpha in enumerate(self.alphas) if alpha in allowed_alphas]
            print('allowed_indices',allowed_indices)
            if allowed_indices:
                Q_s_allowed = Q_s[allowed_indices]
                max_idx = np.argmax(Q_s_allowed)
                chosen_idx = allowed_indices[max_idx]
                chosen_alpha = self.alphas[chosen_idx]
            else:

                chosen_alpha = random.choice(self.alphas)
            # print('Q, alpha', chosen_alpha)
        return chosen_alpha

    def follow_policy(self, s, allowed_alphas,epsilon=None):
        # print(">>> DQNAgent.policy is called!")
        # total_added, Alphas = action_info



        Q_s = self.behavior_Q(s)  # shape [num_vars,1]

        # Q_s_for_action = Q_s[-total_added:].detach().numpy().flatten()
        Q_s = Q_s.detach().numpy().flatten()
        #
        # if epsilon is not None and np.random.random() < epsilon:
        #     chosen_alpha = random.choice(self.alphas)
        #     print('random, alpha',chosen_alpha)
        # else:
        #     print(Q_s)
        #     idx = np.argmax(Q_s)
        #     # print('idx,alphas',idx,self.alphas)
        #     chosen_alpha = self.alphas[idx]
        #
        #     print('Q, alpha',chosen_alpha)
        #
        #
        # return chosen_alpha
        # else:

        allowed_indices = [i for i, alpha in enumerate(self.alphas) if alpha in allowed_alphas]
        # print('allowed_indices',allowed_indices)
        if allowed_indices:
            Q_s_allowed = Q_s[allowed_indices]
            max_idx = np.argmax(Q_s_allowed)
            chosen_idx = allowed_indices[max_idx]
            chosen_alpha = self.alphas[chosen_idx]
        else:

            chosen_alpha = random.choice(self.alphas)
        # print('Q, alpha', chosen_alpha)
        return chosen_alpha



    def _learn_from_memory(self, gamma, learning_rate):
        trans_pieces = self.sample(self.batch_size)
        states_0 = [x.s0 for x in trans_pieces]  # states_0
        states_1 = [x.s1 for x in trans_pieces]
        rewards = np.array([x.reward for x in trans_pieces])
        is_dones = np.array([x.is_done for x in trans_pieces])
        mispricings = np.array([x.mispricing for x in trans_pieces])  #
        chosen_alphas = [x.chosen_alpha for x in trans_pieces]

        data = []
        labels = []

        for i in range(len(states_0)):
            data.append(states_0[i])
            Q_s0 = self.target_Q(states_0[i]).detach().squeeze(-1)  # [α_num]
            y = Q_s0.clone()  # 复制当前 Q 值，用于后续更新。

            alpha_idx = np.where(self.alphas == chosen_alphas[i])[0][0]
            if is_dones[i]:
                Q_target = rewards[i]
            elif mispricings[i]:
                Q_target = rewards[i]
            else:
                Q_s1_max = torch.max(self.target_Q(states_1[i])).item()
                Q_target = rewards[i] + gamma * Q_s1_max

            y[alpha_idx] = Q_target

            labels.append(y.numpy())

        loss_val = 0.0
        for _ in range(self.epochs):
            loss_ = self.behavior_Q.train_or_test(data, labels, None, chosen_alphas, None, train=True)
            loss_val += loss_

        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_freq == 0:
            self._update_target_Q()

        return loss_val / self.epochs


    def learning_method(self, instance, gamma, learning_rate, epsilon, display):
        self.env = instance
        self.S = self.get_aug_state()
        s0_aug = self.S # 初始状态

        time_in_episode, total_reward = 0, 0
        is_done = False
        loss = 0

        while not is_done:
            allowed_alphas = self.env.allowed_alphas  # 从环境获取当前允许的 alpha
            chosen_alpha = self.policy(s0_aug, allowed_alphas,epsilon)
            s1_aug, r, is_done, total_reward, mispricing = self.act(chosen_alpha)

            if self.total_trans > self.batch_size:
                for _ in range(self.epochs):
                    loss += self._learn_from_memory(gamma, learning_rate)

            time_in_episode += 1
            if not mispricing:
                s0_aug = s1_aug

        if time_in_episode * self.epochs > 0:
            loss /= (time_in_episode * self.epochs)
        if display:
            print(f"epsilon:{epsilon:.2f}, loss:{loss:.2f}, {self.experience.last_episode}")
        return time_in_episode, total_reward



    def learning(self, name_file, epsilon=0.85, decaying_epsilon=True, gamma=0.9,
                 learning_rate=3e-4, max_episode_num=439, display=False, min_epsilon=1e-2, min_epsilon_ratio=0.8,
                 model_index=0):


        total_time, episode_reward, num_episode = 0, 0, 0
        total_times, episode_rewards, num_episodes = [], [], []

        # 获取所有实例的名称
        file_names = read_data.instance_name(name_file)
        print('len(file_name)', len(file_names))

        # max_episode_num now set to be 480 as there are 480 instances uploaded

        # print("max_episode_num is",max_episode_num)
        for i in range(max_episode_num):
            if epsilon is None:
                epsilon = 1e-10
            elif decaying_epsilon:
                # epsilon = 1.0 / (1 + num_episode)
                epsilon = self._decayed_epsilon(epsilon,cur_episode=num_episode + 1,
                                                min_epsilon=min_epsilon,
                                                minus_epsilon= 0.005,
                                                target_episode= 300 )   #int(max_episode_num * min_epsilon_ratio))
            print('epsilon',epsilon)
            #### read_file
            cut_stock_instance = read_data.instance_train(i, name_file)
            # name_ = name_file[i]
            # optimal_val = df.loc[df['Name'] == "BPP_50_50_0.1_0.8_0.txt", "Best LB"].item()

            if cut_stock_instance == "not found":
                print("########### NOT FOUND ###############")
                continue;

            cut_stock_instance.initialize()

            time_in_episode, episode_reward = self.learning_method(cut_stock_instance, \
                                                                   gamma=gamma, learning_rate=learning_rate,
                                                                   epsilon=epsilon, display=display)
            # total_time += time_in_episode
            num_episode += 1

            total_times.append(time_in_episode)
            episode_rewards.append(episode_reward)
            num_episodes.append(num_episode)

            print("Episode: " + str(num_episode) + " takes " + str(time_in_episode) + " steps with epsilon " + str(epsilon))

            if num_episode % 80 == 0:
                model_save_name = 'Model_' + str(model_index) + "_" + str(num_episode) + '.pt'
                path_model_check = F'check_points/Model/{model_save_name}'
                self.target_Q.save_state(path_model_check)

                path_data_check = 'check_points/Data/'
                np.save(path_data_check + "model_" + str(model_index) + "_total_steps_" + str(num_episode),
                        np.asarray(total_times))

                fig, axs = plt.subplots(2)
                fig.suptitle('Vertically stacked subplots steps, reward')
                axs[0].plot(num_episodes, total_times)
                axs[1].plot(num_episodes, episode_rewards)
                plt.savefig("./save_graph/training_plots/" + str(num_episode) + ".png")

            #
            # current_instance_name = file_names[i]
            # print('current_instance_name', current_instance_name)
            #

            # parameter_folder = os.path.join('instances', 'Scheduled_train')
            # parameter_file_path = os.path.join(parameter_folder, current_instance_name)
            #

            # dual_vars_folder = os.path.join('instances', 'Dual_Vars')
            # os.makedirs(dual_vars_folder, exist_ok=True)


            # dual_var_filename = f"{os.path.splitext(current_instance_name)[0]}_dual_var.txt"
            # dual_var_save_path = os.path.join(dual_vars_folder, dual_var_filename)


            # dual_vars = cut_stock_instance.Shadow_Price


            # np.savetxt(dual_var_save_path, dual_vars, fmt='%.6f')
            # print(f"Saved dual variables for instance {current_instance_name} to {dual_var_save_path}")

        path_data = 'save_data/training_data/'
        model_save_name = 'Model_' + str(model_index) + '.pt'
        path_model = F'save_models/{model_save_name}'

        self.target_Q.save_state(path_model)
        np.save(path_data + "model_" + str(model_index) + "_total_steps", np.asarray(total_times))

        return total_times, episode_rewards, num_episodes


