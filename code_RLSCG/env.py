#### The cutting stock object is the environment

from gurobipy import GRB
import gurobipy as gp
import numpy as np
import pandas as pd
import Parameters
import random
from copy import deepcopy
# from read_data import read_optimal_solutions
import os

# path = 'given_solutions/solutions.xlsx'
global df_optiaml_val
# df_optiaml_val = pd.read_excel(path)

path = 'given_solutions/solutions.csv'
df_optiaml_val = pd.read_csv(path)


class CuttingStock(object):
    def __init__(self, customer_count, customer_length,
                roll_length,name_):
        # each instance corresponds to self.state = self.env.reset() in learning_method in agent.py

        # state: curent graph (connection, current column nodes, current constraint node and their features)
        #### static info: problem defination, same info used for initialization this instance
        self.name = name_ ## used to get optimal value
        self.n = len(customer_count)
        self.m = sum(customer_count)
        self.order_lens = customer_length
        self.demands = customer_count
        self.roll_len = roll_length
        self.optimal_val = df_optiaml_val.loc[df_optiaml_val['Name'] == self.name, "Best LB"].item()
        # self.distinct_order_count = roll_count

        #### dynamic info (info needed to for state + reward), get from CG iterations from solving current RMP and PP:  
        self.objVal_history = []
        self.total_steps = 0

        ## action with their reduced cost (stored as tuple) ([all the patterns],[the reduced cost for all those patterns])
        # self.available_action = ()
        self.current_patterns = self.generate_initial_patterns()

        self.count_convergence = 0

        self.edge_indices = [[], []]

        '''  
        Info for column and constraint node features, stored using list,length will change
            column 
                      number of constraint participation
                      current solution value (if not in the basis, 0 -> int or not)
                      columnIsNew

                      column incompatibility degree --> check this

             constraint : shadow price
                          number of columns contributing to the constraint
        '''  
        ## for all the columns (size change)
        self.RC = []
        self.In_Cons_Num = []
        self.ColumnSol_Val = []
        self.ColumnIs_Basic = []
        self.Waste = []
        ## for all the variable that are in the basis, count the number of times it's in basis, otherwise 0
        self.stay_in = []
        self.stay_out = []
        ## 1-> just left the basis in last iteration, 0 not just left
        self.just_left = []
        self.just_enter = []

        ## 1-> is action node, 0 -> .. useless as we can do this at get_aug_state
        # self.action_node = []



        ## for all the constraints (size fixed)
        self.Shadow_Price = []
        self.In_Cols_Num = []


        ## constants
        self.pool_size = Parameters.action_pool_size
        self.alpha_obj_weight = Parameters.alpha_obj_weight

        self.step_penalty = Parameters.step_penalty


        # 新增属性
        self.pai_star = None
        self.pai_in = None
        self.pai_out = None
        self.alphas = np.linspace(0, 0.95, 20)

        self.alpha_misprice = 0.95



        self.num_misprice = 0

        self.previous_pai = None
        self.associated_alpha = []
        self.mispricing_indicator = []
        self.mispricing_penalty = 10
        self.convergence_reward = 50.0
        self.step_penalty = 5  #


        self.mispricing_flag = False

    @property
    def allowed_alphas(self):
        return [alpha for alpha in self.alphas if alpha <= self.alpha_misprice]



    def summarize(self):
        print("Problem instance with ", self.n, " orders and those orders include in total ", self.m, "rolls")
        print("-"*47)
        print("\nOrders:\n")
        for i, order_len in enumerate(self.order_lens):
            print("\tOrder ", i, ": length= ", order_len, " demand=", self.demands[i])
        print("\nRoll Length: ", self.roll_len)

    def generate_initial_patterns(self):
        patterns = []
        for i in range(self.n):
            pattern_ = list(np.zeros(self.n).astype(int))
            pattern_[i] = int(self.roll_len/self.order_lens[i])
            patterns.append(pattern_)
        return patterns

    def solve_subproblem_return_actions(self, duals, chosen_alpha=None):
        #     Groundset = range (10)


        self.pai_out = duals.copy()


        alphas_used = [alpha for alpha in self.alphas if alpha <= self.alpha_misprice]

        if chosen_alpha is not None:


            alpha = chosen_alpha
            #
            # print('pai_in', self.pai_in)
            # print('duals', duals)
            pai_sep = alpha * self.pai_in + (1 - alpha) * duals




            subproblem = gp.Model("subproblem")
            order_range = range(self.n)
            subproblem.Params.OutputFlag = 0
            subproblem.setParam(GRB.Param.PoolSolutions, 1)  # 只需要一个解
            subproblem.Params.LogToConsole = 0


            x = subproblem.addVars(order_range, vtype=GRB.INTEGER, name="x")
            subproblem.addConstr(sum(self.order_lens[i] * x[i] for i in order_range) <= self.roll_len)
            subproblem.setObjective(1 - gp.quicksum(pai_sep[i] * x[i] for i in order_range), GRB.MINIMIZE)

            subproblem.optimize()

            sol = []
            for e in range(self.n):
                sol.append(int(x[e].Xn))
            rc = subproblem.objVal

            if rc < -1e-2:

                return subproblem, [sol], [rc], alphas_used

            else:

                return subproblem, [], [], alphas_used
        else:

            columns_to_select = []
            reduced_costs = []
            # alphas_used = []


            # print('alphas',self.alphas)
            for alpha in self.alphas:
                # print('alphas')
                pai_sep = alpha * self.pai_in + (1 - alpha) * duals

                subproblem = gp.Model("subproblem")


                order_range = range(self.n)
            # subproblem = gp.Model("subproblem")

            # Limit how many solutions to collect
                subproblem.Params.LogToConsole = 0

                subproblem.setParam(GRB.Param.PoolSolutions, self.pool_size)

                # Limit the search space by setting a gap for the worst possible solution
                # that will be accepted
                subproblem.setParam(GRB.Param.PoolGap, 0.10)
                # do a systematic search for the k- best solutions
                subproblem.setParam(GRB.Param.PoolSearchMode, 2)
                subproblem.Params.OutputFlag = 0
                # decision variables
                x = subproblem.addVars(order_range,
                                       vtype=GRB.INTEGER,
                                       obj=duals,
                                       name="x")
                # direction of optimization (min or max)
                subproblem.modelSense = GRB.MAXIMIZE
                # Length constraint
                subproblem.addConstr(sum(self.order_lens[i] * x[i] for i in order_range) <= self.roll_len)
                subproblem.optimize()


                if subproblem.SolCount >= 1:
                    sol = []
                    subproblem.setParam(GRB.Param.SolutionNumber, 0)
                    for e in range(self.n):
                        sol.append(int(x[e].Xn))
                    rc = 1 - np.dot(pai_sep, sol)
                    if rc < -1e-2:
                        columns_to_select.append(sol)
                        reduced_costs.append(rc)
                        # alphas_used.append(alpha)

        return subproblem, columns_to_select, reduced_costs, alphas_used

    def solve_subproblem(self, duals, chosen_alpha):
        #     Groundset = range (10)


        self.pai_out = duals.copy()

        if chosen_alpha is not None:

            # print('subproblem, chosen alpha', chosen_alpha_idx)

            alpha = chosen_alpha
            pai_sep = alpha * self.pai_in + (1 - alpha) * duals

            subproblem = gp.Model("subproblem")
            order_range = range(self.n)
            subproblem.Params.OutputFlag = 0
            subproblem.setParam(GRB.Param.PoolSolutions, 1)  # 只需要一个解
            subproblem.Params.LogToConsole = 0

            x = subproblem.addVars(order_range, vtype=GRB.INTEGER, obj=duals, name="x")
            subproblem.modelSense = GRB.MAXIMIZE
            subproblem.addConstr(sum(self.order_lens[i] * x[i] for i in order_range) <= self.roll_len)
            subproblem.optimize()

            sol = []
            for e in range(self.n):
                sol.append(int(x[e].Xn))
            rc = 1 - np.dot(pai_sep, sol)
            if rc < -1e-2:
                return subproblem, [sol], [rc], [alpha]
            else:
                return subproblem, [], [], []

    # get the constraint participation for each col node and col participation for each cons node
    ## use current patterns to count the non-zeros in the pattern matrix

    ### be careful about pointer thing (need to do the copy, otherwise previous stored value may change due to the update)
    def update_col_con_number(self,patterns):
        pa = np.asarray(patterns)
        self.In_Cons_Num = np.count_nonzero(pa, axis=1)
        self.In_Cols_Num = np.count_nonzero(pa, axis=0)
    


    def define_master_problem(self, patterns):
        
        n_pattern = len(patterns)
        pattern_range = range(n_pattern)
        order_range = range(self.n)
        patterns = np.array(patterns, dtype=int)
        master_problem = gp.Model("master problem")

        master_problem.Params.OutputFlag = 0

        master_problem.Params.LogToConsole = 0
        
        # decision variables
        lambda_ = master_problem.addVars(pattern_range,
                                        vtype=GRB.CONTINUOUS,
                                        obj=np.ones(n_pattern),
                                        name="lambda")
        
        # direction of optimization (min or max)
        master_problem.modelSense = GRB.MINIMIZE
        
        # demand satisfaction constraint
        for i in order_range:
            master_problem.addConstr(sum(patterns[p,i]*lambda_[p] for p in pattern_range) == self.demands[i],
                                    "Demand[%d]" %i)
        master_problem.Params.LogToConsole = 0
        return master_problem




    def basic_or_not(self):
        ## use the current solution value for each column, return whether it'sin basis or not
        sol = np.asarray(self.ColumnSol_Val)
        is_basic = abs(sol - 0)>=0.001
        integer_map = map(int, is_basic)
        integer_list = list(integer_map)
        return np.asarray(integer_list)

    def initialize(self):



        dual_file = f'instances/Dual_Vars/{self.name.rstrip(".txt")}_dual_var.txt'
        dual_file_pred = f'instances/Dual_Vars/{self.name.rstrip(".txt")}_dual_var_pred.txt'

        if os.path.exists(dual_file):
            with open(dual_file) as f:
                self.pai_star = np.array([float(line.strip()) for line in f])
                print('dual var')
        elif os.path.exists(dual_file_pred):
            with open(dual_file_pred) as f:
                self.pai_star = np.array([float(line.strip()) for line in f])
                print('dual var predict ')
        else:
            self.pai_star = np.zeros(self.n)



        # self.pai_star = np.zeros(self.n)


        self.pai_in = self.pai_star.copy()
        self.instance_size = self.get_instance_size()

        self.total_steps = 0
        ## this is for taking the first step without having any actions (solving first CG iteration)
        patterns = self.current_patterns

        for i in range(len(patterns)):
            self.Waste.append(self.roll_len-np.dot(np.asarray(patterns[i]),np.asarray(self.order_lens)))
        
        self.update_col_con_number(patterns)

        ## update how many constraints each columnn is in; how many columns each constraint contains
        master_problem = self.define_master_problem(patterns)
        master_problem.optimize()

        self.ColumnSol_Val = np.asarray(master_problem.x)

        self.ColumnIs_Basic = np.asarray(master_problem.vbasis)+np.ones(len(patterns))
        # self.ColumnIs_Basic = self.basic_or_not()

        self.objVal_history.append(master_problem.objVal)
        dual_variables = np.array([constraint.pi for constraint in master_problem.getConstrs()])

        ## get the rc for all the initial patterns
        for pattern in patterns:
            rc = 1-sum(dual_variables[i] * pattern[i] for i in range(len(dual_variables)))
            self.RC.append(rc)
        self.Shadow_Price = dual_variables
        self.previous_pai = dual_variables.copy()  # 初始化previous_pai
        # print('check')
        subproblem,columns_to_select,reduced_costs,alpha_used = self.solve_subproblem_return_actions(dual_variables)


        self.associated_alpha = [-1] * len(patterns)  # 初始列α为-1
        self.mispricing_indicator = [0] * len(patterns)

        for col_idx, pattern in enumerate(self.current_patterns):
            for cons_idx, val in enumerate(pattern):
                if val != 0:  # 只记录非零连接
                    self.edge_indices[0].append(cons_idx)  # 约束节点
                    self.edge_indices[1].append(col_idx)  # 列节点




        reward = 0

        # self.available_action = (columns_to_select,reduced_costs,alpha_used)
        self.stay_in = list(np.zeros(len(patterns)))
        self.stay_out = list(np.zeros(len(patterns)))
        self.just_left = list(np.zeros(len(patterns)))
        self.just_enter = list(np.zeros(len(patterns)))
   
        is_done = False
        return reward, is_done




    def step(self, chosen_alpha, Train=True):

        # mispricing = False

        last_basis = np.array(self.ColumnIs_Basic[:])

        last_basis = np.append(last_basis, 0)

        #    available_action = (columns_candidates, reduced_costs, alpha_used)
        # last_columns_to_select, last_reduced_cost, _ = deepcopy(self.available_action)


        if self.total_steps ==0:
            master_problem = self.define_master_problem(self.current_patterns)
            master_problem.optimize()
            dual_variables = np.array([constraint.pi for constraint in master_problem.getConstrs()])

        #
        else:
            dual_variables = self.pai_out

            if self.mispricing_flag ==True:

                self.pai_in = chosen_alpha * self.pai_in + (1 - chosen_alpha) * self.pai_out

            else:
                self.pai_in = self.pai_star.copy()

        # print('self.total_steps',self.total_steps)
        subproblem, columns_to_select, reduced_costs, alpha_used = self.solve_subproblem_return_actions(
                dual_variables, chosen_alpha)
        # print('reduced_costs',reduced_costs)

        reward = 0
        is_done = False
        mispricing = False

        if len(columns_to_select) > 0:  # Case 1

            # print('reduced_costs[0]',reduced_costs[0])

            self.current_patterns.append(columns_to_select[0])
            self.Waste.append(self.roll_len - np.dot(np.array(columns_to_select[0]), np.array(self.order_lens)))
            self.associated_alpha.append(chosen_alpha)
            self.mispricing_indicator.append(0)

            self.update_col_con_number(self.current_patterns)
            master_problem = self.define_master_problem(self.current_patterns)
            master_problem.optimize()

            new_col_idx = len(self.current_patterns) - 1  # 新列的索引
            for cons_idx, val in enumerate(columns_to_select[0]):
                if val != 0:  # 只记录非零连接
                    self.edge_indices[0].append(cons_idx)
                    self.edge_indices[1].append(new_col_idx)

            self.ColumnSol_Val = np.asarray(master_problem.x)
            self.ColumnIs_Basic = np.asarray(master_problem.vbasis) + np.ones(len(self.current_patterns))
            dual_variables_next = np.array([constraint.pi for constraint in master_problem.getConstrs()])
            self.pai_out = dual_variables_next
            self.previous_pai = dual_variables.copy()

            while len(self.ColumnIs_Basic) < len(last_basis):
                self.ColumnIs_Basic = np.append(self.ColumnIs_Basic, 0)
            difference = last_basis - self.ColumnIs_Basic
            num_existing = len(difference) - 1
            self.just_left = [0] * num_existing
            self.just_enter = [0] * num_existing
            for i in range(num_existing):
                if difference[i] == 1:
                    self.just_left[i] = 1
                    self.stay_in[i] = 0
                elif difference[i] == -1:
                    self.just_enter[i] = 1
                    self.stay_out[i] = 0
                elif difference[i] == 0:
                    if last_basis[i] == 1:
                        self.stay_in[i] += 1
                    else:
                        self.stay_out[i] += 1
            self.just_left.append(0)
            self.stay_out.append(0)
            self.stay_in.append(0)
            self.just_enter.append(1 if self.ColumnIs_Basic[-1] == 1 else 0)

            self.objVal_history.append(master_problem.objVal)
            self.RC.append(reduced_costs[0])
            self.Shadow_Price = dual_variables_next

            delta_t = self.objVal_history[-2] - self.objVal_history[-1]

            if delta_t ==0 and chosen_alpha !=0: # mispricing

                mispricing = True
                reward = -self.mispricing_penalty
                if self.current_patterns:  #
                    self.mispricing_indicator[-1] = 1
                self.alpha_misprice = min(self.alpha_misprice, chosen_alpha)
                # print('self.alpha_misprice', self.alpha_misprice)

                self.mispricing_flag = True


            else: # test if mispricing
                reward = delta_t - self.step_penalty

        else:  # Case 2

            _, check_columns, check_rcs, _ = self.solve_subproblem_return_actions(dual_variables, chosen_alpha=0)
            if len(check_columns) > 0:  # Mispricing
                mispricing = True
                reward = -self.mispricing_penalty
                if self.current_patterns:
                    self.mispricing_indicator[-1] = 1

                self.alpha_misprice = min(self.alpha_misprice, chosen_alpha)
                # print('self.alpha_misprice', self.alpha_misprice)

                self.mispricing_flag = True

            else:  # True convergence
                is_done = True
                reward = self.convergence_reward

        self.total_steps += 1
        return reward, is_done, mispricing

    def get_instance_size(self):

        return len(self.pai_star)

