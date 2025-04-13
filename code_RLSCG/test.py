



## where the functions used for testing are stored;
## these functions are used for generating the data needed in  plotting_functions.py testing functions

import numpy as np
import matplotlib.pyplot as plt
from env import *
from DQN import *
from read_data import *
from gurobipy import GRB
import gurobipy as gp
import time

import json


import os
import numpy as np
import time
import traceback
from copy import deepcopy

def follow_policy_RLCG(DQN,action_info, s):
    '''DQN selects an action
    '''
    total_added, Actions = action_info
    Q_s = DQN.target_Q(s)
    Q_s_for_action = Q_s[-total_added::]
    # rand_value = np.random.random()


    # idx = int(np.argmax(Q_s_for_action))


    idx = torch.argmax(Q_s_for_action).item()
    # print('idx',idx)
    # print('Actions',Actions)

    return Actions[idx]


def general_compare(DQN_test, DQN_test_RLCG, model_index, prob_size,adaptive_alpha0):
    """
    run Greedy, RLCG, RLSCG

    :return: (Greedy, RLCG, RLSCG) result
    """
    prob_size = int(prob_size)
    name = "TestName" + str(prob_size)
    name_file = os.path.join("Name_files", f"{name}.txt")

    names = instance_name(name_file)
    total_length = len(names)

    Greedy = []
    adaptive = []
    RL = []
    RLCG = []


    print("#####################")
    print(f"Starts testing for model {model_index} for instance size {prob_size}")
    print()

    for i in range(total_length):
        print(f"\n🔹 Running instance {i + 1} out of {total_length} ...")
        # Greedy
        try:
            print('🔹 Starting Greedy method...')
            time1 = time.time()
            cut1 = instance_test(i, name_file)
            reward, is_done = cut1.initialize_adaptive()
            while not is_done:

                reward, is_done,_= cut1.step(0,False)
            history_opt_g = cut1.objVal_history
            time2 = time.time()
            obj_greedy = history_opt_g[-1]
            steps_g = len(history_opt_g)
            print(f" Greedy takes {steps_g} steps to reach obj {obj_greedy} with time {time2 - time1:.4f} seconds")

            Greedy.append((history_opt_g, steps_g, time2 - time1, obj_greedy))


        except Exception as e:
            print(f" Exception in Greedy for instance {i}: {e}")
            print(traceback.format_exc())
            continue
        # RLSCG
        try:

            time3 = time.time()
            cut2 = instance_test(i, name_file)
            reward, is_done = cut2.initialize()

            model_save_name = f'Model_{model_index}.pt'
            path_model = os.path.join('save_models', model_save_name)
            DQN_test.target_Q.restore_state(path_model)
            DQN_test.behavior_Q.restore_state(path_model)

            DQN_test.env = cut2
            DQN_test.S = DQN_test.get_aug_state()

            while not is_done:
                # action_info = DQN_test.S[1]
                s = DQN_test.S
                # action = follow_policy(DQN_test, action_info, s)

                allowed_alphas = cut2.allowed_alphas
                selected_alpha = DQN_test.follow_policy( s,allowed_alphas)

                reward, is_done,_ = cut2.step(selected_alpha)



                DQN_test.S = DQN_test.get_aug_state()
            history_opt_rl = cut2.objVal_history
            time4 = time.time()
            obj_RL = history_opt_rl[-1]
            steps_RL = len(history_opt_rl)
            print(f" RL takes {steps_RL} steps to reach obj {obj_RL} with time {time4 - time3:.4f} seconds")

            RL.append((history_opt_rl, steps_RL, time4 - time3, obj_RL))

        except Exception as e:
            print(f" Exception in RL for instance {i}: {e}")
            print(traceback.format_exc())
            continue

            # RL- RLCG -to compare
        try:

            time5 = time.time()
            cut3 = instance_test(i, name_file)
            reward, is_done = cut3.initialize_RLCG()

            model_save_name = f'Model_RLCG_2022.pt'
            path_model = os.path.join('save_models', model_save_name)
            DQN_test_RLCG.target_Q.restore_state(path_model)
            DQN_test_RLCG.behavior_Q.restore_state(path_model)

            DQN_test_RLCG.env = cut3
            DQN_test_RLCG.S = DQN_test_RLCG.get_aug_state_RLCG()

            while not is_done:
                action_info = DQN_test_RLCG.S[1]
                s = DQN_test_RLCG.S[0]
                # action = follow_policy(DQN_test, action_info, s)

                action = follow_policy_RLCG(DQN_test_RLCG, action_info, s)
                # print('RL action',selected_alpha)

                reward, is_done = cut3.step_RLCG(action)

                # reward, is_done = cut2.step(action, False)
                DQN_test_RLCG.S = DQN_test_RLCG.get_aug_state_RLCG()
            history_opt_rlcg = cut3.objVal_history
            time6 = time.time()
            obj_RLCG = history_opt_rlcg[-1]
            steps_RLCG = len(history_opt_rlcg)
            print(f" RLCG takes {steps_RLCG} steps to reach obj {obj_RLCG} with time {time6 - time5:.4f} seconds")

            RLCG.append((history_opt_rlcg, steps_RLCG, time6 - time5, obj_RLCG))


        except Exception as e:
            print(f" Exception in RLCG for instance {i}: {e}")
            print(traceback.format_exc())
            continue




    complete_data = (Greedy, RL, RLCG,adaptive)
    path = 'save_data/testing_data/'
    os.makedirs(path, exist_ok=True)

    import pickle


    with open(os.path.join(path, f'testing_result_model_{model_index}_size{prob_size}.pkl'), 'wb') as f:
        pickle.dump(complete_data, f)


    return complete_data
