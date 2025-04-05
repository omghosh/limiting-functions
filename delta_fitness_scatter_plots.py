
import numpy as np 
import pandas as pd
from functions import *
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy import stats
import sys
# sns.set(style='darkgrid')
# 2 day color (0.77, 0.84, 0.75)
# (0.63, 0.68,0.62 )
env_color_dict = {'2Day': (0.77, 0.84, 0.75), '1Day': (0.55, 0.6, 0.98), 'Salt': (1, 0.59, 0.55)}
    # Statistical tests for both comparisons
# def test_deviation_from_perfect_correlation(x, y):
#     # 1. Pearson correlation test
#     r, p_pearson = stats.pearsonr(x, y)
    
#     # 2. Test if slope is significantly different from 1
#     # and intercept different from 0 using linear regression
#     slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
#     # 3. Calculate residuals from 1:1 line
#     residuals = (y - x)
    
#     # 4. One-sample t-test on residuals (testing if mean differs from 0)
#     t_stat, p_residuals = stats.ttest_1samp(residuals, 0)
    
#     return {
#         'correlation': r,
#         'p_correlation': p_pearson,
#         'slope': slope,
#         'intercept': intercept, 
#         'p_slope': p_value,
#         'mean_residual': np.mean(residuals),
#         'p_residuals': p_residuals
#     }




bc_counts, fitness_df, grants_df_with_barcode_df, full_df = create_full_fitness_dataframe()
organized_perturbation_fitness_df= create_delta_fitness_matrix(batches, fitness_df, environment_dict)

# remove all Li_WT
organized_perturbation_fitness_df.loc[mutant_dict['anc: Li_WT']]
organized_perturbation_fitness_df.drop(mutant_dict['anc: Li_WT'], inplace=True)

all_perts  = list(set([col.split('_')[2] for col in organized_perturbation_fitness_df.columns]))

# bc_of_interest = bc_counts[bc_counts['gene']=='IRA1']['barcode']

# gene = 'IRA1'
# for bc_of_interest in bc_counts[bc_counts['gene']==gene]['barcode']:

# get rep-rep delta deviation
oneday_bases_fitness = fitness_df[['Batch1_1Day_30_fitness', 'Batch2_1Day_1.5_fitness', 'Batch3_1Day_M3_fitness', 'Batch4_1Day_M3b4_fitness']]
oneday_bases_error = fitness_df[[col.replace('fitness', 'stderror') for col in oneday_bases_fitness]]

# take a weighted averge of the base fitnesses and calculate new standard error
weights = 1/oneday_bases_error**2
oneday_base_avg = pd.Series((oneday_bases_fitness.values*weights.values).sum(axis = 1)/weights.values.sum(axis = 1), index = oneday_bases_fitness.index)
oneday_base_avg_error = 1/weights.sum(axis = 1)**0.5
print(oneday_base_avg_error)
print(oneday_base_avg)

salt_base_avg = fitness_df[['Batch1_Salt_30_fitness', 'Batch2_Salt_1.5_fitness', 'Batch3_Salt_M3_fitness']].mean(axis = 1)
twoday_base_avg = fitness_df[[ 'Batch2_2Day_1.5_fitness', 'Batch3_2Day_M3_fitness', 'Batch4_2Day_M3b4_fitness']].mean(axis = 1)


for bc_of_interest in ['CGCTAAAGACATAATGTGGTTTGTTG_TCCATAATTGGGAATTGGATTTTGGC']: # mutant_dict['Original Training']:
    # Identify the gene associated with this mutant
    if bc_of_interest not in grants_df_with_barcode_df['barcode'].values:
        gene = ''
    else:
        gene = grants_df_with_barcode_df[grants_df_with_barcode_df['barcode'] == bc_of_interest]['gene'].values[0]
    fig, ax1 = plt.subplots(figsize=(8, 6))
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

    ax1.set_xlabel(f'Salt Delta Fitness')
    ax1.set_ylabel(f'1 Day Delta Fitness') #, color=env_color_dict['1Day'])
    ax2.set_ylabel(f'2Day Delta Fitness') #, color=env_color_dict['Salt'])

    # ax1.tick_params(axis='y', colors=env_color_dict['1Day'])
    # ax2.tick_params(axis='y', colors=env_color_dict['Salt'])

    # Make both y-axes symmetric
    ax1_ylim = max(abs(val) for val in ax1.get_ylim())
    ax2_ylim = max(abs(val) for val in ax2.get_ylim())
    shared_ylim = max(ax1_ylim, ax2_ylim)
    ax1.set_ylim(-shared_ylim, shared_ylim)
    ax2.set_ylim(-shared_ylim, shared_ylim)

    # make x-axis symmetric
    ax1_xlim = max(abs(val) for val in ax1.get_xlim())
    ax1.set_xlim(-1, ax1_xlim)



    plt.title(f'Gene: {gene} Delta Fitness Salt vs Other Bases')
    # fig.tight_layout()

    # After creating all your error bar plots but before plt.show():

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

    # Keep your existing axis limits (don't change them)
    # The line will be clipped to the visible area automatically

    print(np.array(r1_values), np.array(r2_values))
    print(f'Pearsons r: {stats.pearsonr(np.array(r1_values), np.array(r2_values))}')


    # # Collect the data points for testing
    # oneday_values = []
    # twoday_values = []
    # salt_values_1 = []
    # salt_values_2 = []
    # rep1_values = []
    # rep2_values = []


    # for pert in all_perts:
    #     # Get 1-day vs 2-day data
    #     oneday = organized_perturbation_fitness_df.loc[bc_of_interest, 
    #             [col for col in all_conds if f'{pert}_' in col and '1Day' in col]]
    #     salt1 = organized_perturbation_fitness_df.loc[bc_of_interest,
    #             [col for col in all_conds if f'{pert}_' in col and 'Salt' in col]]
    #     rep1 = fitness_df[[col for col in fitness_df.columns if f'{pert}-R1_fitness' in col and '1Day' in col]]
    #     rep2 = fitness_df[[col for col in fitness_df.columns if f'{pert}-R2_fitness' in col and '1Day' in col]]
        
    #     if len(oneday) > 0 and len(salt1) > 0:
    #         oneday_values.extend(oneday.values)
    #         salt_values_1.extend(salt1.values)
    #     if len(rep1) > 0 and len(rep2) > 0:
    #         rep1_values.extend(rep1.loc[bc_of_interest].values)
    #         rep2_values.extend(rep2.loc[bc_of_interest].values)
        
    #     # Get salt vs 2-day data
    #     salt2 = organized_perturbation_fitness_df.loc[bc_of_interest,
    #         [col for col in all_conds if f'{pert}_' in col and 'Salt' in col]]

    #     twoday = organized_perturbation_fitness_df.loc[bc_of_interest,
    #             [col for col in all_conds if f'{pert}_' in col and '2Day' in col]]
        
    #     if len(salt2) > 0 and len(twoday) > 0:
    #         salt_values_2.extend(salt2.values)
    #         twoday_values.extend(twoday.values)
    #         # if salt.values[0]>0.5:
    #             # print(f'{pert} salt {salt.values} twoday {twoday2.values[0]}')
    #             # print(bc_of_interest)

    # # Run the tests
    # results_salt_1day = test_deviation_from_perfect_correlation(np.array(salt_values_1), np.array(oneday_values))
    # results_salt_2day = test_deviation_from_perfect_correlation(np.array(salt_values_2), np.array(twoday_values))
    # results_rep1 = test_deviation_from_perfect_correlation(np.array(rep1_values), np.array(rep2_values))
    # # Print results
    # print(f'\nGene: {gene} Barcode: {bc_of_interest}')
    # print("\n1 Day vs Salt comparison:")
    # print(f"Correlation: {results_salt_1day['correlation']:.3f} (p={results_salt_1day['p_correlation']:.4f})")
    # print(f"Slope: {results_salt_1day['slope']:.3f} (p={results_salt_1day['p_slope']:.4f})")
    # print(f"Mean deviation from 1:1 line: {results_salt_1day['mean_residual']:.3f} (p={results_salt_1day['p_residuals']:.4f})")

    # print("\nSalt vs 2 Day comparison:")
    # print(f"Correlation: {results_salt_2day['correlation']:.3f} (p={results_salt_2day['p_correlation']:.4f})")
    # print(f"Slope: {results_salt_2day['slope']:.3f} (p={results_salt_2day['p_slope']:.4f})")
    # print(f"Mean deviation from 1:1 line: {results_salt_2day['mean_residual']:.3f} (p={results_salt_2day['p_residuals']:.4f})")

    # print("\nRep1 vs Rep2 comparison:")
    # print(f"Correlation: {results_rep1['correlation']:.3f} (p={results_rep1['p_correlation']:.4f})")
    # print(f"Slope: {results_rep1['slope']:.3f} (p={results_rep1['p_slope']:.4f})")
    # print(f"Mean deviation from 1:1 line: {results_rep1['mean_residual']:.3f} (p={results_rep1['p_residuals']:.4f})")
    # # include legend for just one of each colored point 
    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    by_label = dict(zip(labels, handles))
    by_label.update(dict(zip(labels2, handles2)))

    ax1.legend(by_label.values(), by_label.keys(), loc='upper left')



    plt.tight_layout()
    plt.savefig(f'../plots/delta_fitness_scatter_plots/{gene}_{bc_of_interest}_delta_fitness_scatterplot.png')
    plt.show()
    plt.close()








