import numpy as np
import matplotlib.pyplot as plt
import os

model_index = 0  # model idex
prob_size = 200   # problem size


file_path = f'save_data/testing_data/testing_result_model_{model_index}_size{prob_size}.pkl'
complete_data = np.load(file_path, allow_pickle=True)




Greedy_data, RL_data, RLCG,adaptive = complete_data

greedy_times = [instance[2] for instance in Greedy_data]
rl_times = [instance[2] for instance in RL_data]
print('rl_times',rl_times)
rlcg_times = [instance[2] for instance in RLCG]
adaptive_times = [instance[2] for instance in adaptive]

avg_greedy = np.mean(greedy_times)
avg_rl = np.mean(rl_times)
avg_rlcg = np.mean(rlcg_times)
avg_adaptive = np.mean(adaptive_times)

std_greedy = np.std(greedy_times)
std_rl = np.std(rl_times)
std_rlcg = np.std(rlcg_times)
std_adaptive = np.std(adaptive_times)


print(f"Greedy average time: {avg_greedy:.4f} ± {std_greedy:.4f} s")
print(f"RLSCG     average time: {avg_rl:.4f} ± {std_rl:.4f} s")
print(f"RLCG     average time: {avg_rlcg:.4f} ± {std_rlcg:.4f} s")

methods = ['Greedy', 'RL', 'RLCG']    #
means = [avg_greedy, avg_rl, avg_rlcg]   #
stds = [std_greedy, std_rl, std_rlcg]      #

plt.figure(figsize=(8, 6))
plt.bar(methods, means, yerr=stds, capsize=10, color=['skyblue', 'lightgreen', 'salmon'])
plt.title(f'Running Time Comparison (Size = {prob_size})')
plt.ylabel('Time (seconds)')
plt.grid(axis='y', linestyle='--')
plt.show()


greedy_steps = [instance[1] for instance in Greedy_data]
rl_steps = [instance[1] for instance in RL_data]
rlcg_steps = [instance[1] for instance in RLCG]
adaptive_steps = [instance[1] for instance in adaptive]


avg_greedy = np.mean(greedy_steps)
avg_rl = np.mean(rl_steps)
avg_rlcg = np.mean(rlcg_steps)

std_greedy = np.std(greedy_steps)
std_rl = np.std(rl_steps)
std_rlcg = np.std(rlcg_steps)

print(f"Greedy iteration: {avg_greedy:.4f} ± {std_greedy:.4f} ")
print(f"RLSCG     iteration: {avg_rl:.4f} ± {std_rl:.4f} ")
print(f"RLCG    iteration: {avg_rlcg:.4f} ± {std_rlcg:.4f} ")


# Boxplot: Exclude 'adaptive'
plt.figure(figsize=(8, 6))
plt.boxplot([greedy_steps, rl_steps, rlcg_steps], labels=['Greedy', 'RL', 'RLCG'], showfliers=True)
plt.title(f'Iteration Comparison (Size = {prob_size})')
plt.ylabel('Iteration Count')
plt.grid(axis='y', linestyle='--')
# plt.ylim(30, 200)
plt.show()


print("All RL times:")
for i, t in enumerate(rl_times):
    print(f"Instance {i}: {t}")
