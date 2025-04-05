
import numpy as np 
import pandas as pd
from functions import *
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
env_color_dict = {'2Day': (0.77, 0.84, 0.75), '1Day': (0.55, 0.6, 0.98), 'Salt': (1, 0.59, 0.55)}
environment_dict['Salt'] = [env for env in environment_dict['Salt'] if env != 'Batch3_Salt_NS_fitness']

bc_counts, fitness_df, grants_df_with_barcode_df, full_df = create_full_fitness_dataframe()
directory = 'pruned_delta_scree_plots'
these_muts = mutant_dict['Original Training'] + mutant_dict['Original Testing']
np.random.seed(100)
if not os.path.exists(f'../plots/{directory}'):
    os.makedirs(f'../plots/{directory}')
# for each base and for each set of mutants, get that matrix 
# subsample down to a preset number of mutants and environments 
# do svd on that subsampled matrix 
# plot the fraction of variance explained 
n_folds = 100
organized_perturbation_fitness_df = create_delta_fitness_matrix(batches, fitness_df, environment_dict)
this_fitness = organized_perturbation_fitness_df
dimensionality_results_evo1D = pd.DataFrame(columns = environment_dict.keys(), index = mutant_dict.keys())
dimensionality_results_evo2D = pd.DataFrame(columns =  environment_dict.keys(), index = mutant_dict.keys())



overall_detection_limit = 0


these_mutants = mutant_dict['Original Training'] + mutant_dict['Original Testing']

stderr_threshold = 0.3
std_error_matrix = this_fitness.loc[these_mutants, [col.replace('fitness', 'stderror') for col in all_conds]]
mutants_to_cull = std_error_matrix[std_error_matrix > stderr_threshold].dropna(how='all').index
num_mutants_to_cull = len(mutants_to_cull)
these_mutants = [mut for mut in these_mutants if mut not in mutants_to_cull]
plt.figure(figsize=(8, 6), dpi=300)
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
        print(f'Starting fold {i}')
        noise_df = pd.DataFrame(np.random.normal(0, (organized_perturbation_fitness_df.loc[fitness_matrix.index, [col.replace('fitness', 'stderror') for col in fitness_matrix.columns]].values), fitness_matrix.shape), columns=fitness_matrix.columns)

        noise_matrix = np.random.normal(0, this_fitness.loc[fitness_matrix.index,  [col.replace('fitness', 'stderror') for col in fitness_matrix.columns]])

        U_n,S_n,Vt_n = np.linalg.svd(noise_df)

        max_var_exp_list.append((S_n**2/np.sum(S**2))[0])
        max_var_exp_for_each_component.append((S_n**2/np.sum(S**2)))
        min_var_exp_for_each_component.append((S_n**2/np.sum(S**2)))

    # fill between the max and min of the noise matrix
        
    min_var_exp_for_each_component = np.min(min_var_exp_for_each_component, axis=0)
    max_var_exp_for_each_component = np.max(max_var_exp_for_each_component, axis=0)

    plt.fill_between(np.arange(len(S)), min_var_exp_for_each_component, max_var_exp_for_each_component, color=env_color_dict[base_env], alpha=0.3)
        

    max_var_exp = np.mean(max_var_exp_list)
    # how many dimensions fall above the limit of detection

    dimensionality_results_evo2D.loc['Original', base_env] = np.sum((S**2)/np.sum(S**2) > max_var_exp)

    num_above_limit = np.sum((S**2)/np.sum(S**2) > max_var_exp)
    # index of the first value above the limit of detection
    first_above_limit = np.argmax((S**2)/np.sum(S**2) > max_var_exp) +1

    plt.semilogy((S**2)/np.sum(S**2), '-', label = base_env, color=env_color_dict[base_env])
    # plot dots for only the components above the limit of detection
    plt.semilogy(np.arange(len(S))[np.where((S**2)/np.sum(S**2) > max_var_exp)], (S**2/np.sum(S**2))[np.where((S**2)/np.sum(S**2) > max_var_exp)], 'o', color=env_color_dict[base_env])

    # plot horizontal line at the limit of detection 
    # plt.axhline(max_var_exp, color=env_color_dict[base_env], linestyle=':', alpha=0.5)

fitness_matrix = this_fitness.loc[these_mutants, all_conds]
U,S,Vt = np.linalg.svd(fitness_matrix.values)
# find noise matrix
max_var_exp_list = []
for i in range(n_folds):

    noise_matrix = np.random.normal(0, this_fitness.loc[fitness_matrix.index,  [col.replace('fitness', 'stderror') for col in fitness_matrix.columns]])
    U_n,S_n,Vt_n = np.linalg.svd(noise_matrix)
    # plot the fraction of variance explained
    # plt.semilogy((S_n**2)/np.sum(S**2), ':', color='purple', alpha=0.1)
    max_var_exp_list.append((S_n**2/np.sum(S**2))[0])
max_var_exp = np.mean(max_var_exp_list)

plt.xlim(-1, 15)
plt.ylim(10**-3, 1)



plt.title(f'Original Mutants, evolved in 2Day, minus {num_mutants_to_cull} mutants\n of {len(these_mutants)} with std error > {stderr_threshold}')
plt.xlabel('k')
plt.ylabel('Fraction of Variance Explained')
plt.legend()
# plt.show()
plt.savefig(f'../plots/nice_scree_plot_April2025.png')

dimensionality_results_evo1D = dimensionality_results_evo1D.dropna()
dimensionality_results_evo2D = dimensionality_results_evo2D.dropna()


def summary(dimensionality_results_evo1D,dimensionality_results_evo2D, metric ):

    for evo_cond in ['Evo1D', 'Evo2D']:
        if evo_cond == 'Evo1D':
            dimensionality_results = dimensionality_results_evo1D
        else:
            dimensionality_results = dimensionality_results_evo2D
        genotypes = dimensionality_results.index
        environments = environment_dict.keys()
        values = dimensionality_results.values
        # # Set the figure size
        plt.figure(figsize=(8, 8), dpi=300)

        # Number of genotypes and environments
        n_genotypes = len(genotypes)
        n_env = len(environments)

        # Set the positions on the x-axis for each environment group within each genotype
        bar_width = 0.2  # width of each bar
        indices = np.arange(n_genotypes)

        # Plotting each environment as a separate bar within each genotype group
        for i, env in enumerate(environments):
            if evo_cond == 'Evo1D' and env == '1Day':
                plt.bar(indices + i * bar_width, values[:, i], width=bar_width, label=env, color=env_color_dict[env])
            if evo_cond == 'Evo2D' and env == '2Day':
                plt.bar(indices + i * bar_width, values[:, i], width=bar_width, label=env, color=env_color_dict[env])
            else:
                plt.bar(indices + i * bar_width, values[:, i], width=bar_width, label=env, color=env_color_dict[env], alpha = 0.5)

        # # Set x-axis labels and ticks
        plt.xticks(indices + bar_width, genotypes)  # Positioning labels in the center of grouped bars
        plt.xlabel('Ancestral Genotypes')
        plt.ylabel('Inferred Dimensionality')
        plt.title(f'Evolved in {evo_cond}, {metric} Dimensionality')

        # Add legend
        plt.legend(title="Environments")

        # Show the plot
        plt.tight_layout()
        plt.show()
        # plt.savefig(f'../plots/{directory}/{evo_cond}_dimensionality_barplot.png')
        plt.close()


