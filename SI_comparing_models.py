import numpy as np
import matplotlib.pyplot as plt
from functions import *
from statsmodels.multivariate.cancorr import CanCorr
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
# Assuming all preprocessing and computations (which_fitness, which_mutants, this_fitness, etc.) are done as in your code.
bc_counts, fitness_df, grants_df_with_barcode_df, full_df = create_full_fitness_dataframe()

organized_perturbation_fitness_df = create_delta_fitness_matrix(batches, fitness_df, environment_dict)

env_color_dict = {'2Day': (0.77, 0.84, 0.75), '1Day': (0.55, 0.6, 0.98), 'Salt': (1, 0.59, 0.55)}
env_color_list = [(0.77, 0.84, 0.75), (0.55, 0.6, 0.98), (1, 0.59, 0.55),(0.33, 0.79, 0.63),(0.91,0.77, 0.47) , (0.8,0.53, 1)]


r2_results_from_gene_predictions = {}
r2_results_from_bcv_predictions = {}
r2_results_from_delta_fitness_avg = {}




#### make the plots 3 in a row, clone the y axis (same ylims) 


##############################
# Prediction from Gene ###
##########################


original_training = mutant_dict['Original Training']
original_testing = mutant_dict['Original Testing']
print('Starting gene predictions')
sorted_genes = []
for base_env in environment_dict.keys():
    for focal_env in environment_dict[base_env]:
        if focal_env == 'Batch1_Salt_32Baffle_fitness':
            continue
        print(f'Starting {focal_env}')
        for which_fitness in ['delta']: #, 'original']:
            mutant_colors = []
            truths = []
            preds_gene = []
            preds_delta =[]
            if which_fitness == 'delta':
                this_fitness = organized_perturbation_fitness_df
            else:
                this_fitness = fitness_df
            perturbation = focal_env.split('_')[2]
        # get other envs with the same perturbation
            matched_envs = [env for env in all_conds if f'{perturbation}_' in env and env != focal_env]

            for mut in original_testing:
                true_fitness = this_fitness.loc[mut, focal_env]
                gene = bc_counts[bc_counts['barcode'] == mut]['gene'].values[0]
                # all bcs for the gene within original training set 
                gene_bcs = bc_counts[bc_counts['gene'] == gene]['barcode'].values
                gene_bcs = list(set(gene_bcs).intersection(set(original_training)))
                if gene in mutant_colorset.keys():
                    mutant_colors.append(mutant_colorset[gene])
                else:
                    mutant_colors.append((0.5, 0.5, 0.5))
                gene_fitness_avg = this_fitness.loc[gene_bcs, focal_env].mean()
                predicted_fitness = gene_fitness_avg
                truths.append(true_fitness)
                preds_gene.append(predicted_fitness)
                matched_fitness_vals = this_fitness.loc[mut, matched_envs]
                # if matched fitness vals is empty, then add nan 
                if matched_fitness_vals.shape[0] == 0:
                    predicted_delta_fitness = np.nan
                else:
                    predicted_delta_fitness = matched_fitness_vals.mean(axis = 0)
                preds_delta.append(predicted_delta_fitness)
            
            print(len(truths), len(preds_gene), len(preds_delta))



            df = pd.DataFrame({'truth': truths, 'pred_delta': preds_delta, 'pred_gene': preds_gene})
            # drop rows with NaN, and drop those indices from the colors

            mutant_colors_dropped = [mutant_colors[i] for i in range(len(mutant_colors)) if not np.isnan(df['pred_gene'].iloc[i])]
            df = df.dropna()

            # get pearson correlation coefficient
            pearson_r_gene = np.corrcoef(df['truth'], df['pred_gene'])[0, 1]
            r2_gene = pearson_r_gene**2

            pearson_r_delta = np.corrcoef(df['truth'], df['pred_delta'])[0, 1]
            r2_delta = pearson_r_delta**2

            # r2_gene = r2_score(df['truth'], df['pred'])
            r2_results_from_gene_predictions[focal_env] = r2_gene

            r2_results_from_delta_fitness_avg[focal_env] = r2_delta

        
            all_predictions, prediction_dfs, truth = bicross_validation(original_training, original_testing, [cond for cond in environment_dict[base_env] if cond != focal_env], [focal_env], this_fitness)
            best_rank, r2_bcv_from_model = find_best_rank(all_predictions, truth, prediction_metric = 'r2')


            r2_results_from_bcv_predictions[focal_env]= r2_bcv_from_model

            fig, axs = plt.subplots(1, 3, figsize=(14, 4))

            axs[0].scatter(df['pred_gene'], df['truth'] , c=mutant_colors_dropped, alpha=0.5)
            axs[0].text(0.1, 0.9, f'R2: {r2_gene:.2f}', transform=axs[0].transAxes)
            axs[0].set_xlabel('Predicted Fitness')
            axs[0].set_ylabel('True Fitness')
            axs[0].set_title('Gene Fitness Prediction')
            xmin_gene, xmax_gene = axs[0].get_xlim()
            ymin_gene, ymax_gene = axs[0].get_ylim()
            min_val_gene = min(xmin_gene, ymin_gene)
            max_val_gene = max(xmax_gene, ymax_gene)

            axs[1].scatter(df['pred_delta'], df['truth'], c=mutant_colors_dropped, alpha=0.5)
            axs[1].text(0.1, 0.9, f'R2: {r2_delta:.2f}', transform=axs[1].transAxes)
            axs[1].set_xlabel('Predicted Fitness')
            # axs[1].set_ylabel('True Fitness')
            axs[1].set_title('Delta Fitness Prediction')
            xmin_delta, xmax_delta = axs[1].get_xlim()
            ymin_delta, ymax_delta = axs[1].get_ylim()
            min_val_delta = min(xmin_delta, ymin_delta)
            max_val_delta = max(xmax_delta, ymax_delta)


            # plot the predictions for the best rank
            axs[2].scatter(all_predictions[best_rank], truth, c=mutant_colors, alpha=0.5)
            xmin_bcv, xmax_bcv = axs[1].get_xlim()
            ymin_bcv, ymax_bcv = axs[1].get_ylim()

            min_val_bcv = min(xmin_bcv, ymin_bcv)
            max_val_bcv = max(xmax_bcv, ymax_bcv)

            axs[2].set_xlabel('Predicted Fitness')
            # axs[2].set_ylabel('True Fitness')
            axs[2].set_title('BCV Fitness Prediction')
            plt.suptitle(f'Fitness Prediction in {focal_env}')
            axs[2].text(0.1, 0.9, f'R2: {r2_bcv_from_model:.2f}', transform=axs[2].transAxes)

            # get same lims for all plots
            overall_min = min(min_val_gene, min_val_delta, min_val_bcv)
            overall_max = max(max_val_gene, max_val_delta, max_val_bcv)
            for i in range(3):
                axs[i].plot([overall_min, overall_max], [overall_min, overall_max], 'k--')
                axs[i].set_xlim(overall_min, overall_max)
                axs[i].set_ylim(overall_min, overall_max)
            
            # plt.tight_layout()
            plt.savefig(f'plots/SI/model_comparisons/{focal_env}_{which_fitness}.png')
            # plt.show()
            plt.close()
            print(f'Succeeded for {focal_env}')


# ##########################
#             # Prediction from delta fitness in other bases (average) 

# #####################
# print('Starting delta fitness predictions')
# sorted_genes = []
# for base_env in environment_dict.keys():
#     for focal_env in environment_dict[base_env]:
#         # if focal_env != 'Batch2_2Day_1.5_fitness':
#         #     continue
#         print(f'Starting {focal_env}')
#         which_fitness = 'delta'
#         mutant_colors = []
#         truths = []
#         preds = []
#         if which_fitness == 'delta':
#             this_fitness = organized_perturbation_fitness_df
#         else:
#             this_fitness = fitness_df

#         perturbation = focal_env.split('_')[2]
#         # get other envs with the same perturbation
#         matched_envs = [env for env in all_conds if f'{perturbation}_' in env and env != focal_env]

#         for mut in original_testing:
#             true_fitness = this_fitness.loc[mut, focal_env]
#             gene = bc_counts[bc_counts['barcode'] == mut]['gene'].values[0]
#             # all bcs for the gene within original training set 
#             if gene not in sorted_genes:
#                 sorted_genes.append(gene)
#             gene_bcs = bc_counts[bc_counts['gene'] == gene]['barcode'].values
#             gene_bcs = list(set(gene_bcs).intersection(set(original_training)))
#             if gene in mutant_colorset.keys():
#                 mutant_colors.append(mutant_colorset[gene])
#             else:
#                 mutant_colors.append((0.5, 0.5, 0.5))
#             matched_fitness_vals = this_fitness.loc[mut, matched_envs]
#             predicted_delta_fitness = matched_fitness_vals.mean(axis = 0)
#             truths.append(true_fitness)
#             preds.append(predicted_delta_fitness)

#         df = pd.DataFrame({'truth': truths, 'pred': preds})
#         # drop rows with NaN, and drop those indices from the colors

#         mutant_colors_dropped = [mutant_colors[i] for i in range(len(mutant_colors)) if not np.isnan(df['pred'].iloc[i])]
#         df = df.dropna()
#         if df.shape[0]==0:
#             continue
#         pearson_r_delta = np.corrcoef(df['truth'], df['pred'])[0, 1]
#         r2_delta = pearson_r_delta**2
#         # r2_delta = r2_score( df['pred'], df['truth'])
#         fig, axs = plt.subplots(2, 1, figsize=(5, 10))
#         r2_results_from_delta_fitness_avg[focal_env] = r2_delta

#         # get single color legend for mutants (one of each color, labeled by gene)
#         axs[0].scatter( df['pred'].values,df['truth'].values, c=mutant_colors_dropped, alpha=0.5)
#         # print the r2 value on the plot
#         axs[0].text(0.1, 0.9, f'R2: {r2_delta:.2f}', transform=axs[0].transAxes)
#         axs[0].set_xlabel('Predicted Fitness')
#         axs[0].set_ylabel('True Fitness')
#         axs[0].set_title('Delta Fitness Prediction')
#         xmin, xmax = axs[0].get_xlim()
#         ymin, ymax = axs[0].get_ylim()
#         min_val = min(xmin, ymin)
#         max_val = max(xmax, ymax)
#         axs[0].plot([min_val, max_val], [min_val, max_val], 'k--')
#         all_predictions, prediction_dfs, truth = bicross_validation(original_training, original_testing, [cond for cond in environment_dict[base_env] if cond != focal_env], [focal_env], this_fitness)

#         best_rank, r2_bcv_2 = find_best_rank(all_predictions, truth, prediction_metric = 'r2')

#         # plot the predictions for the best rank
#         axs[1].scatter(all_predictions[best_rank],truth, c=mutant_colors, alpha=0.5)
#         xmin, xmax = axs[1].get_xlim()
#         ymin, ymax = axs[1].get_ylim()

#         min_val = min(xmin, ymin)
#         max_val = max(xmax, ymax)

#         axs[1].plot([min_val, max_val], [min_val, max_val], 'k--')

#         axs[1].set_xlabel('Predicted Fitness')
#         axs[1].set_ylabel('True Fitness')
#         axs[1].set_title('BCV Fitness Prediction')
#         plt.suptitle(f'Fitness Prediction in {focal_env}')
#         axs[1].text(0.1, 0.9, f'R2: {r2_bcv_2:.2f}', transform=axs[1].transAxes)
#         plt.tight_layout()
#         plt.savefig(f'../plots/delta_fitness_vs_bcv/{focal_env}_{which_fitness}.png')
#         # plt.show()
#         plt.close()


def create_gene_legend(sorted_genes, figsize=(8, 2), ncol=4, title="Gene Legend"):
    """
    Creates a standalone legend figure showing each gene with its corresponding color.
    
    Parameters:
    -----------
    mutant_colorset : dict
        Dictionary mapping gene names to their color tuples.
    figsize : tuple, optional
        Size of the legend figure. Default is (8, 2).
    ncol : int, optional
        Number of columns in the legend. Default is 4.
    title : str, optional
        Title for the legend. Default is "Gene Legend".
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing only the legend.
    """
    # Create a figure with no axes
    fig = plt.figure(figsize=figsize)
    
    # Create legend handles
    legend_elements = []
    for gene in sorted_genes:
        color = mutant_colorset[gene]
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                   markersize=10, label=gene, alpha = 0.8)
        )
    
    # Add a handle for gray items (if they exist in your plots)
    if (0.5, 0.5, 0.5) not in mutant_colorset.values():
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=(0.5, 0.5, 0.5), 
                   markersize=10, label='Other genes')
        )
    
    # Add the legend to the figure
    fig.legend(handles=legend_elements, loc='center', ncol=ncol, 
               frameon=True, title=title)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig
# Create the legend figure
legend_fig = create_gene_legend(sorted_genes, title='Gene Legend')
# plt.show()
plt.savefig('plots/SI/gene_legend.png')
plt.close()



# Setup your dataframe as before
results_df = pd.DataFrame([r2_results_from_bcv_predictions, r2_results_from_gene_predictions, r2_results_from_delta_fitness_avg]).T
results_df.columns = ['r2_bcv', 'r2_gene', 'r2_delta_avg']
results_df['environment'] = results_df.index
results_df['base_environment'] = results_df['environment'].apply(lambda x: x.split('_')[1])
results_df['base_color'] = results_df['base_environment'].map(env_color_dict)

# Reset index to make environment a regular column
results_df = results_df.reset_index(drop=True)

# Convert to tidy/long format
results_df_tidy = results_df.melt(
    id_vars=['environment', 'base_environment', 'base_color'], 
    value_vars=['r2_bcv', 'r2_gene', 'r2_delta_avg'],
    var_name='model', 
    value_name='r2'
)

# Create a figure
plt.figure(figsize=(8, 6))

sns.boxplot(
    y='model',  # Model is on y-axis 
    x='r2',     # R2 is on x-axis
    data=results_df_tidy, 
    width=0.5, 
    fliersize=0,
    boxprops={'facecolor': 'none'},   # This makes the boxes transparent
    medianprops={'color': 'black'},   # Ensure median line is visible
    whiskerprops={'color': 'black'},  # Ensure whiskers are visible
    capprops={'color': 'black'}       # Ensure caps are visible
)
# Pivot the data for line connections
pivot_df = results_df_tidy.pivot_table(
    index=['environment', 'base_environment', 'base_color'],
    columns='model',
    values='r2'
).reset_index()

# Get the y-positions for each model (note: for horizontal, we use y-positions)
# We need to map the model names to their positions (0, 1, 2)
model_positions = {model: i for i, model in enumerate(['r2_bcv', 'r2_gene' , 'r2_delta_avg'])}

# Get unique environments and colors
unique_environments = results_df_tidy['base_environment'].unique()
color_palette = [env_color_dict.get(env, 'gray') for env in unique_environments]

# Now add the points (horizontal stripplot)
sns.stripplot(
    y='model',  # Now model is on y-axis
    x='r2',     # and r2 is on x-axis
    data=results_df_tidy,
    hue='base_environment',
    jitter=0.1,
    alpha=0.75,
    palette=color_palette,
    size=8
)

plt.title('Model Comparison (R² by Environment)')
plt.xlabel('R² Score')  # x and y labels are swapped
plt.ylabel('Model')
plt.legend(title='Base Environment')
plt.tight_layout()
plt.savefig('plots/SI/model_comparison.png')
# plt.show()
