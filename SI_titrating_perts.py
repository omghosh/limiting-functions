import numpy as np 
import pandas as pd
from functions import *
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
environment_dict['Salt'] = [env for env in environment_dict['Salt'] if env != 'Batch3_Salt_NS_fitness']
env_color_dict = {'2Day': (0.60, 0.73, 0.61), '1Day': (0.49, 0.38, 1), 'Salt': (1, 0.59, 0.55)}

bc_counts, fitness_df, grants_df_with_barcode_df, full_df = create_full_fitness_dataframe()
these_muts = mutant_dict['Original Training'] + mutant_dict['Original Testing']
np.random.seed(100)
# for each base and for each set of mutants, get that matrix 
# subsample down to a preset number of mutants and environments 
# do svd on that subsampled matrix 
# plot the fraction of variance explained 
n_folds = 100
organized_perturbation_fitness_df = create_delta_fitness_matrix(batches, fitness_df, environment_dict)
this_fitness = organized_perturbation_fitness_df


overall_detection_limit = 0


# these_mutants = mutant_dict['Original Training'] + mutant_dict['Original Testing']



num_permutations = 100
for mut_group in ['anc: IRA1_NON', 'anc: IRA1_MIS', 'anc: GPB2', 'anc: TOR1', 'anc: CYR1']:
    for evo_cond in ['Evo1D', 'Evo2D']:
        if mut_group == 'original':
            these_mutants = mutant_dict['Original Training'] + mutant_dict['Original Testing']
        else:
            these_mutants = mutant_dict[mut_group]

        these_mutants = [mut for mut in these_mutants if mut in mutant_dict[evo_cond]]
        if len(these_mutants) == 0:
            print(f'No mutants in {mut_group} for {evo_cond}')
            continue

        if mut_group == 'IRA1_NON':
            mutants_to_cull = []
        else:
            stderr_threshold = 0.3
            std_error_matrix = this_fitness.loc[these_mutants, [col.replace('fitness', 'stderror') for col in all_conds]]
            mutants_to_cull = std_error_matrix[std_error_matrix > stderr_threshold].dropna(how='all').index
        num_mutants_to_cull = len(mutants_to_cull)
        these_mutants = [mut for mut in these_mutants if mut not in mutants_to_cull]
        if len(these_mutants) == 0:
            print(f'No mutants in {mut_group} for {evo_cond} after culling')
            continue

        plt.figure(figsize=(8, 6))    
        for q in range(16):
            print(f'Going up to {q} environments')
            for p in range(num_permutations):
                for base_env in environment_dict.keys():
                    these_envs = environment_dict[base_env]
                    # randomly permute the environments
                    np.random.shuffle(these_envs)
                    these_envs=these_envs[:q+1]

                    fitness_matrix = this_fitness.loc[these_mutants, these_envs]
                    U,S,Vt = np.linalg.svd(fitness_matrix.values)
                    # find noise matrix 
                    max_var_exp_list = []
                    for i in range(n_folds):
                        # print(f'Starting fold {i}')
                        noise_df = pd.DataFrame(np.random.normal(0, (organized_perturbation_fitness_df.loc[fitness_matrix.index, [col.replace('fitness', 'stderror') for col in fitness_matrix.columns]].values), fitness_matrix.shape), columns=fitness_matrix.columns)
                        noise_matrix = np.random.normal(0, this_fitness.loc[fitness_matrix.index,  [col.replace('fitness', 'stderror') for col in fitness_matrix.columns]])
                        U_n,S_n,Vt_n = np.linalg.svd(noise_df)
                        max_var_exp_list.append((S_n**2/np.sum(S**2))[0])
                    max_var_exp = np.mean(max_var_exp_list)
                    num_above_limit = np.sum((S**2)/np.sum(S**2) > max_var_exp)
                    num_above_limit += np.random.normal(0, 0.1)
                    # add some jitter to the x axis
                    num_envs = len(these_envs) + np.random.normal(0, 0.1)        
                    plt.scatter(num_envs, num_above_limit, color=env_color_dict[base_env], alpha=0.3)
        plt.xlabel('Number of environments')
        plt.ylabel('Number of components above detection limit')
        plt.title(f'Titrating in perturbations, {mut_group}, {evo_cond}')
        plt.savefig(f'plots/SI/titrate_in_perturbations_{mut_group}_{evo_cond}.png', dpi=300)
        plt.close()

