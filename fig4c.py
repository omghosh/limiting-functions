import numpy as np 
import pandas as pd
from functions import *
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os
import sys
import seaborn as sns
sns.set_style("darkgrid")

plt.rcParams['font.family'] = 'Geneva'
plt.rcParams['font.size'] = 12
organized_perturbation_fitness_df = create_delta_fitness_matrix(batches, fitness_df, environment_dict)

environment_dict['Salt'] = [env for env in environment_dict['Salt'] if env != 'Batch3_Salt_NS_fitness']

bc_counts, fitness_df, grants_df_with_barcode_df, full_df = create_full_fitness_dataframe()
for metric in ['ind_detection_limit', 'ind_entropy', 'overall_detection_limit', 'overall_entropy']:
    directory = f'delta_dimensionality_summary_{metric}'
    these_muts = mutant_dict['Original Training'] + mutant_dict['Original Testing']

    n_folds = 1000
    this_fitness = organized_perturbation_fitness_df
    dimensionality_results_evo1D = pd.DataFrame(columns = environment_dict.keys(), index = mutant_dict.keys())
    dimensionality_results_evo2D = pd.DataFrame(columns =  environment_dict.keys(), index = mutant_dict.keys())



    for mutant_list in mutant_dict.keys():
        print(f'Starting {mutant_list}')
        if not 'anc' in mutant_list:
            continue
        evo1D = [mut for mut in mutant_dict[mutant_list] if mut in mutant_dict['Evo1D']]
        evo2D = [mut for mut in mutant_dict[mutant_list] if mut in mutant_dict['Evo2D']]
        overall_detection_limit=0

        for evo_cond in ['Evo1D', 'Evo2D']:
            if evo_cond == 'Evo1D':
                dimensionality_results = dimensionality_results_evo1D
            else:
                dimensionality_results = dimensionality_results_evo2D
            stderr_threshold = 0.3
            these_mutants = [mut for mut in mutant_dict[mutant_list] if mut in mutant_dict[evo_cond]]
            std_error_matrix = this_fitness.loc[these_mutants, [col.replace('fitness', 'stderror') for col in all_conds]]
            if mutant_list == 'anc: IRA1_NON':
                mutants_to_cull = []
            else:
                mutants_to_cull = std_error_matrix[std_error_matrix > stderr_threshold].dropna(how='all').index
            num_mutants_to_cull = len(mutants_to_cull)

            if len(these_mutants) == 0:
                continue
            if len(these_mutants) == len(mutants_to_cull):
                continue


            s_values = {}
            for base_env in environment_dict.keys():
                print(f'Starting {mutant_list} {evo_cond} {base_env}')
                these_envs = environment_dict[base_env]
                fitness_matrix = this_fitness.loc[[mut for mut in these_mutants if mut not in mutants_to_cull], these_envs]
                U,S,Vt = np.linalg.svd(fitness_matrix.values)
                s_values[base_env] = S
                # find noise matrix 
                max_var_exp_list = []
                for i in range(n_folds):
                    noise_matrix = np.random.normal(0, this_fitness.loc[fitness_matrix.index,  [col.replace('fitness', 'stderror') for col in fitness_matrix.columns]])
                    U_n,S_n,Vt_n = np.linalg.svd(noise_matrix)
                    max_var_exp_list.append((S_n**2/np.sum(S**2))[0])

                max_var_exp = np.mean(max_var_exp_list)
                if max_var_exp > overall_detection_limit:
                    overall_detection_limit = max_var_exp

                #number of components that explain variance above the limit of detection
                print(f'{base_env} has {np.sum((S**2)/np.sum(S**2) > max_var_exp)} components above the limit of detection')
                if metric == 'ind_detection_limit':
                    dim = np.sum((S**2)/np.sum(S**2) > max_var_exp)
                    # add some jitter to the points
                    dim = dim + np.random.normal(0, 0.1)
                    dimensionality_results.loc[mutant_list, base_env] = (dim)
                elif metric == 'ind_entropy':
                    s_vector =S[S**2/np.sum(S**2) > max_var_exp]
                    dimensionality_results.loc[mutant_list, base_env] = -np.sum(s_vector**2/np.sum(s_vector**2)*np.log(s_vector**2/np.sum(s_vector**2)))
            if metric == 'overall_detection_limit':
                for base_env in environment_dict.keys():
                    # how many components are above the overall detection limit 
                    this_S = s_values[base_env]
                    dim = np.sum((this_S**2)/np.sum(this_S**2) > overall_detection_limit)
                    dim = dim + np.random.normal(0, 0.1)
                    dimensionality_results.loc[mutant_list, base_env] = dim
            elif metric == 'overall_entropy':
                for base_env in environment_dict.keys():
                    this_S = s_values[base_env]
                    s_vector = this_S[this_S**2/np.sum(this_S**2) > overall_detection_limit]
                    dimensionality_results.loc[mutant_list, base_env] = -np.sum(s_vector**2/np.sum(s_vector**2)*np.log(s_vector**2/np.sum(s_vector**2)))

    dimensionality_results_evo1D = dimensionality_results_evo1D.dropna()
    dimensionality_results_evo2D = dimensionality_results_evo2D.dropna()


    dimensionality_results = dimensionality_results_evo1D
    genotypes = dimensionality_results_evo2D.index
    environments = environment_dict.keys()
    values = dimensionality_results.values

    ancestor_colors =          {'anc: GPB2':(0.2, 0.63, 0.17), ##33a02c',  # dark green
                    'anc: IRA1_NON': (0.12,0.47,0.71), #'#1f78b4', # dark blue
                    'anc: IRA1_MIS': (0.65,0.81, 0.89), #'#a6cee3', # dark blue
                 'anc: CYR1': (0.79, 0.7,0.84 ), # '#cab2d6', # light purple
                    'anc: TOR1':(0.85,0.53,0.75 )}
    
    gene_colors = {'anc: WT': (0.5, 0.5, 0.5), ' ': (1, 1, 1)}
    gene_colors.update(ancestor_colors)
    # Set the figure size
    plt.figure(figsize=(7, 6), dpi=300)
    for g,genotype in enumerate(genotypes):
        if genotype in (dimensionality_results_evo1D.index):
            for env in environments: 
                if env == evo_cond:
                    y1 = dimensionality_results_evo1D.loc[genotype, '2Day']
                    y2 = dimensionality_results_evo1D.loc[genotype, 'Salt']
                    x= dimensionality_results_evo1D.loc[genotype, env]
                    plt.scatter(x, y1, color = gene_colors[genotype], label = genotype, s = 100, alpha = 0.75)
                    plt.scatter(x, y2, color = gene_colors[genotype],  label = genotype, s = 100, alpha = 0.75)

    for g, genotype in enumerate(dimensionality_results_evo2D.index):
        if genotype in dimensionality_results_evo2D.index:
            for env in dimensionality_results_evo2D.keys():
                if env == '2Day':
                    y1 = dimensionality_results_evo2D.loc[genotype, '1Day']
                    y2 = dimensionality_results_evo2D.loc[genotype, 'Salt']
                    x = dimensionality_results_evo2D.loc[genotype, env]

                    plt.scatter(x, y1, color = gene_colors[genotype], label = genotype,  s = 100,  alpha=0.75)
                    plt.scatter(x, y2, color = gene_colors[genotype],  label = genotype, s = 100, alpha=0.75)
    

    min_dim = np.min([np.min(dimensionality_results_evo1D['2Day']), np.min(dimensionality_results_evo1D['Salt']), np.min(dimensionality_results_evo1D['1Day'])])
    min_dim = np.min([min_dim, np.min(dimensionality_results_evo2D['2Day']), np.min(dimensionality_results_evo2D['Salt']), np.min(dimensionality_results_evo2D['1Day'])])
    max_dim = np.max([np.max(dimensionality_results_evo1D['2Day']), np.max(dimensionality_results_evo1D['Salt']), np.max(dimensionality_results_evo1D['1Day'])])
    max_dim = np.max([max_dim, np.max(dimensionality_results_evo2D['2Day']), np.max(dimensionality_results_evo2D['Salt']), np.max(dimensionality_results_evo2D['1Day'])])
    plt.plot([min_dim-0.5, max_dim+0.5], [min_dim-0.5, max_dim+0.5], 'k--')
    # add legend with just one copy of each genotype
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title = 'Genotypes')
    plt.xlabel('Evolution Base Dimensionality', fontsize = 16)
    plt.ylabel('Non-Evolution Base Dimensionality', fontsize=16)
    plt.xlim(min_dim-0.5, max_dim+0.5)
    plt.ylim(min_dim-0.5, max_dim+0.5)

    # plt.title(f'{metric} Dimensionality')
    plt.tight_layout()
    if metric == 'ind_detection_limit':
        plt.savefig(f'plots/fig4c.png')
    else:
        plt.savefig(f'plots/SI/{metric}_dimensionality_scatterplot.png')
    # plt.show()
