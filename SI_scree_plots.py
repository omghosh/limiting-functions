
import numpy as np 
import pandas as pd
from functions import *
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
plt.rcParams['font.family'] = 'Geneva'
plt.rcParams['font.size'] = 8

environment_dict['Salt'] = [env for env in environment_dict['Salt'] if env != 'Batch3_Salt_NS_fitness']

bc_counts, fitness_df, grants_df_with_barcode_df, full_df = create_full_fitness_dataframe()
np.random.seed(100)

n_folds = 1000
organized_perturbation_fitness_df = create_delta_fitness_matrix(batches, fitness_df, environment_dict)
this_fitness = organized_perturbation_fitness_df

mut_keys = ['anc: GPB2', 'anc: IRA1_NON', 'anc: IRA1_MIS', 'anc: CYR1', 'anc: TOR1', 'anc: WT']
fig, axs = plt.subplots(4, 3, figsize=(8, 12), dpi=300)

for e,anc in enumerate(mut_keys):
    for evo_cond in ['Evo1D', 'Evo2D']:
        if evo_cond == 'Evo1D':
            row = e//3
            col = e%3
        else:
            row = e//3 + 2
            col = e%3
        
        these_mutants = mutant_dict[anc]
        these_mutants = [mut for mut in these_mutants if mut in mutant_dict[evo_cond]]
        stderr_threshold = 0.3
        std_error_matrix = this_fitness.loc[these_mutants, [col.replace('fitness', 'stderror') for col in all_conds]]
        if anc == 'anc: IRA1_NON':
            mutants_to_cull = []
        else:
            mutants_to_cull = std_error_matrix[std_error_matrix > stderr_threshold].dropna(how='all').index
        num_mutants_to_cull = len(mutants_to_cull)
        these_mutants = [mut for mut in these_mutants if mut not in mutants_to_cull]

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
            axs[row,col].axhline(max_var_exp, color=env_color_dict[base_env], linestyle=':', alpha=0.5)
            # how many dimensions fall above the limit of detection
            num_above_limit = np.sum((S**2)/np.sum(S**2) > max_var_exp)
            # index of the first value above the limit of detection
            first_above_limit = np.argmax((S**2)/np.sum(S**2) > max_var_exp) +1

            axs[row,col].semilogy((S[:16]**2)/np.sum(S[:16]**2), 'o-', label = base_env, color=env_color_dict[base_env], markerfacecolor = 'none', markeredgecolor=env_color_dict[base_env])
            # plot dots for only the components above the limit of detection
            axs[row,col].semilogy(np.arange(len(S))[np.where((S**2)/np.sum(S**2) > max_var_exp)], (S**2/np.sum(S**2))[np.where((S**2)/np.sum(S**2) > max_var_exp)], 'o', color=env_color_dict[base_env])

            # plot horizontal line at the limit of detection 
            axs[row,col].axhline(max_var_exp, color=env_color_dict[base_env], linestyle=':', alpha=0.5, label = f'{base_env} Limit of Detection')

        axs[row,col].set_xlim(-0.5, 15.25)
        axs[row,col].set_ylim(10**-3, 1)
        axs[row,col].set_title(f'{anc}, EC: {evo_cond}', fontsize=8)
        axs[row,col].set_xticks(ticks=np.arange(0, 16, 2), labels=np.arange(1, 17, 2)) 
fig.supylabel('Fraction of Variance Explained', fontsize=12)
fig.supxlabel('Number of Dimensions', fontsize=12)
plt.tight_layout()
plt.savefig(f'plots/SI/screeplots.png')
