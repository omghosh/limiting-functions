
import numpy as np 
import pandas as pd
from functions import *
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 12

environment_dict['Salt'] = [env for env in environment_dict['Salt'] if env != 'Batch3_Salt_NS_fitness']
n_folds = 1000



bc_counts, fitness_df, grants_df_with_barcode_df, full_df = create_full_fitness_dataframe()
np.random.seed(100)

organized_perturbation_fitness_df = create_delta_fitness_matrix(batches, fitness_df, environment_dict)
this_fitness = organized_perturbation_fitness_df

# these_mutants = mutant_dict['Original Training'] + mutant_dict['Original Testing']
these_mutants = [mut for mut in mutant_dict['Evo2D'] if mut in mutant_dict['anc: WT']]
print(len(these_mutants))
stderr_threshold = 0.3
std_error_matrix = this_fitness.loc[these_mutants, [col.replace('fitness', 'stderror') for col in all_conds]]
mutants_to_cull = std_error_matrix[std_error_matrix > stderr_threshold].dropna(how='all').index
num_mutants_to_cull = len(mutants_to_cull)
these_mutants = [mut for mut in these_mutants if mut not in mutants_to_cull]


plt.figure(figsize=(7, 6), dpi=300)
for base_env in environment_dict.keys():
    if base_env == 'Salt':
        these_envs = environment_dict[base_env]
        # remove 'NS' from the list of environments
        these_envs = [env for env in these_envs if 'NS' not in env]
    else:
        these_envs = environment_dict[base_env]
    fitness_matrix = this_fitness.loc[these_mutants, these_envs]
    U,S,Vt = np.linalg.svd(fitness_matrix.values)

    # find noise matrix 
    max_var_exp_list = []
    max_var_exp_for_each_component = []
    min_var_exp_for_each_component = []
    for i in range(n_folds):
        noise_df = pd.DataFrame(np.random.normal(0, (organized_perturbation_fitness_df.loc[fitness_matrix.index, [col.replace('fitness', 'stderror') for col in fitness_matrix.columns]].values), fitness_matrix.shape), columns=fitness_matrix.columns)
        noise_matrix = np.random.normal(0, this_fitness.loc[fitness_matrix.index,  [col.replace('fitness', 'stderror') for col in fitness_matrix.columns]])
        U_n,S_n,Vt_n = np.linalg.svd(noise_df)
        max_var_exp_list.append((S_n**2/np.sum(S**2))[0])
    max_var_exp = np.mean(max_var_exp_list)
    plt.axhline(max_var_exp, color=env_color_dict[base_env], linestyle=':', alpha=0.5)
    # how many dimensions fall above the limit of detection

    num_above_limit = np.sum((S**2)/np.sum(S**2) > max_var_exp)
    # index of the first value above the limit of detection
    first_above_limit = np.argmax((S**2)/np.sum(S**2) > max_var_exp) +1

    plt.semilogy((S[:16]**2)/np.sum(S[:16]**2), 'o-', label = base_env, color=env_color_dict[base_env], markerfacecolor = 'none', markeredgecolor=env_color_dict[base_env])
    # plot dots for only the components above the limit of detection
    plt.semilogy(np.arange(len(S))[np.where((S**2)/np.sum(S**2) > max_var_exp)], (S**2/np.sum(S**2))[np.where((S**2)/np.sum(S**2) > max_var_exp)], 'o', color=env_color_dict[base_env])

    # plot horizontal line at the limit of detection 
    plt.axhline(max_var_exp, color=env_color_dict[base_env], linestyle=':', alpha=0.5, label = f'{base_env} Limit of Detection')

plt.xlim(-0.5, 15.25)
plt.ylim(10**-3, 1)


# plt.title(f'Original Mutants, evolved in 2Day, minus {num_mutants_to_cull} mutants\n of {len(these_mutants)} with std error > {stderr_threshold}')
plt.title('Ancestor: WT, Evolution Condition: 2 Day', fontsize=14)
plt.xlabel('Number of Dimensions')
plt.ylabel('Fraction of Variance Explained')
# plt.legend(fontsize = 10)
plt.xticks(ticks=np.arange(0, 16, 2), labels=np.arange(1, 17, 2)) 
plt.savefig(f'plots/fig4a.png')
# plt.show()
