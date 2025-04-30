
import numpy as np 
import pandas as pd
from functions import *
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy import stats
import sys
from matplotlib.lines import Line2D


plt.rcParams['font.family'] = 'Geneva'
plt.rcParams['font.size'] = 16


bc_counts, fitness_df, grants_df_with_barcode_df, full_df = create_full_fitness_dataframe()
organized_perturbation_fitness_df= create_delta_fitness_matrix(batches, fitness_df, environment_dict)

# remove all Li_WT
organized_perturbation_fitness_df.loc[mutant_dict['anc: Li_WT']]
organized_perturbation_fitness_df.drop(mutant_dict['anc: Li_WT'], inplace=True)

all_perts  = list(set([col.split('_')[2] for col in organized_perturbation_fitness_df.columns]))


# get rep-rep delta deviation
oneday_bases_fitness = fitness_df[['Batch1_1Day_30_fitness', 'Batch2_1Day_1.5_fitness', 'Batch3_1Day_M3_fitness', 'Batch4_1Day_M3b4_fitness']]
oneday_bases_error = fitness_df[[col.replace('fitness', 'stderror') for col in oneday_bases_fitness]]

# take a weighted averge of the base fitnesses and calculate new standard error
weights = 1/oneday_bases_error**2
oneday_base_avg = pd.Series((oneday_bases_fitness.values*weights.values).sum(axis = 1)/weights.values.sum(axis = 1), index = oneday_bases_fitness.index)
oneday_base_avg_error = 1/weights.sum(axis = 1)**0.5


salt_base_avg = fitness_df[['Batch1_Salt_30_fitness', 'Batch2_Salt_1.5_fitness', 'Batch3_Salt_M3_fitness']].mean(axis = 1)
twoday_base_avg = fitness_df[[ 'Batch2_2Day_1.5_fitness', 'Batch3_2Day_M3_fitness', 'Batch4_2Day_M3b4_fitness']].mean(axis = 1)

for bc_of_interest in ['CGCTAAAGACATAATGTGGTTTGTTG_TCCATAATTGGGAATTGGATTTTGGC']: # mutant_dict['Original Training']:
    # Identify the gene associated with this mutant
    if bc_of_interest not in grants_df_with_barcode_df['barcode'].values:
        gene = ''
    else:
        gene = grants_df_with_barcode_df[grants_df_with_barcode_df['barcode'] == bc_of_interest]['gene'].values[0]
    fig, ax1 = plt.subplots(figsize=(8, 6), dpi = 300)
    ax2 = ax1.twinx()  # Create a second y-axis

    r1_values = []
    r2_values = []

    for pert in all_perts:
        print(f'Working on {pert}')
        oneday = organized_perturbation_fitness_df.loc[bc_of_interest, [col for col in all_conds if f'{pert}_' in col and '1Day' in col]]
        twoday = organized_perturbation_fitness_df.loc[bc_of_interest,[col for col in all_conds if f'{pert}_' in col and '2Day' in col]]
        salt = organized_perturbation_fitness_df.loc[bc_of_interest,[col for col in all_conds if f'{pert}_' in col and 'Salt' in col]]

        oneday_error = organized_perturbation_fitness_df.loc[bc_of_interest, [col.replace('fitness', 'stderror') for col in all_conds if f'{pert}_' in col and '1Day' in col]]
        twoday_error = organized_perturbation_fitness_df.loc[bc_of_interest,[col.replace('fitness', 'stderror') for col in all_conds if f'{pert}_' in col and '2Day' in col]]
        salt_error = organized_perturbation_fitness_df.loc[bc_of_interest,[col.replace('fitness', 'stderror') for col in all_conds if f'{pert}_' in col and 'Salt' in col]]
        if len(twoday) != 0 and len(salt) != 0:
            ax2.errorbar(salt, twoday, xerr = salt_error, yerr = twoday_error, fmt = 'o', color = env_color_dict['2Day'], label = '2Day vs Salt')
            print(f'Salt: {salt.values} 2Day: {twoday.values}')


        # plot oneady vs salt first 
        if len(oneday) != 0 and len(salt) != 0:
            ax1.errorbar(salt, oneday , xerr = salt_error, yerr = oneday_error, fmt = 'o',  color = env_color_dict['1Day'], label = '1Day vs Salt')
        
            oneday_r1_fitness = fitness_df[[col for col in fitness_df.columns if f'{pert}-R1_fitness' in col and '1Day' in col]]
            oneday_r1_stderror = fitness_df[[col for col in fitness_df.columns if f'{pert}-R1_stderror' in col and '1Day' in col]]
            oneday_r2_fitness=fitness_df[[col for col in fitness_df.columns if f'{pert}-R2_fitness' in col and '1Day' in col]]
            oneday_r2_stderror = fitness_df[[col for col in fitness_df.columns if f'{pert}-R2_stderror' in col and '1Day' in col]]
            r1_delta_fitness = oneday_r1_fitness.loc[bc_of_interest]-oneday_base_avg.loc[bc_of_interest]
            r2_delta_fitness = oneday_r2_fitness.loc[bc_of_interest]-oneday_base_avg.loc[bc_of_interest]
            r1_delta_error = (oneday_r1_stderror.loc[bc_of_interest]**2+oneday_base_avg_error.loc[bc_of_interest]**2)**0.5
            r2_delta_error = (oneday_r2_stderror.loc[bc_of_interest]**2+oneday_base_avg_error.loc[bc_of_interest]**2)**0.5
            r1_values.extend(r1_delta_fitness.values)
            r2_values.extend(r2_delta_fitness.values)
            ax2.errorbar(r1_delta_fitness, r2_delta_fitness, xerr = r1_delta_error, yerr= r2_delta_error, fmt='d', color='gray', alpha=0.75, label='Rep1 vs Rep2')

        print(f'Salt: {salt.values} 1Day: {oneday.values}')
        # plot twoday vs salt

    # Formatting
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax1.axvline(x=0, color='k', linestyle='-', alpha=0.5)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax2.axvline(x=0, color='k', linestyle='-', alpha=0.5)

    ax1.set_xlabel(f'Effect of perturbation in Salt base, ' r'$\delta X_{p}^{Salt}$')
    ax1.set_ylabel(f'Effect of perturbation in alternate base')#  ,  color=env_color_dict['1Day'])
    # ax2.set_ylabel(f'Effect of perturbation on 2 Day base', color=env_color_dict['2Day'], rotation = 270)

    # ax1.tick_params(axis='y', colors=env_color_dict['1Day'])
    # ax2.tick_params(axis='y', colors=env_color_dict['Salt'])

    # Make both y-axes symmetric
    ax1_ylim = max(abs(val) for val in ax1.get_ylim())
    ax2_ylim = max(abs(val) for val in ax2.get_ylim())
    shared_ylim = max(ax1_ylim, ax2_ylim)
    ax1.set_ylim(-shared_ylim, shared_ylim)
    ax2.set_ylim(-shared_ylim, shared_ylim)
    ax2.set_yticklabels([])

    # make x-axis symmetric
    ax1_xlim = max(abs(val) for val in ax1.get_xlim())
    ax1.set_xlim(-1, ax1_xlim)


    plt.title(f'Gene: {gene} Delta Fitness Salt vs Other Bases')

    # Get the current axis limits
    x_min, x_max = ax1.get_xlim()
    y1_min, y1_max = ax1.get_ylim()
    y2_min, y2_max = ax2.get_ylim()

    # Find the full range needed for the 1:1 line
    # This ensures the line covers all visible areas of the plot
    min_val = min(x_min, y1_min, y2_min)
    max_val = max(x_max, y1_max, y2_max)

    # Plot the 1:1 line across the full range
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

    # Create custom legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=env_color_dict['1Day'], markersize=8, label=r'$\delta X_{p}^{1 Day}$' ' vs ' r'$\delta X_{p}^{Salt}$'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=env_color_dict['2Day'], markersize=8, label=r'$\delta X_{p}^{2 Day}$' ' vs ' r'$\delta X_{p}^{Salt}$'),
        Line2D([0], [0], marker='d', color='w', markerfacecolor='gray', markersize=8, label='Rep1 vs Rep2'),
        Line2D([0], [0], linestyle='--', color='k', label='1:1 Line', alpha = 0.5)
    ]
    
    ax1.legend(handles=legend_elements, loc='upper right', framealpha=0.7, 
               ncol=1, columnspacing=1, handletextpad=0.5)
    
    plt.tight_layout()
    plt.savefig(f'plots/fig3b.png')
    plt.close()








