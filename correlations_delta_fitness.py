import numpy as np 
import pandas as pd
from functions import *
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
from scipy.stats import pearsonr, spearmanr

bc_counts, fitness_df, grants_df_with_barcode_df, full_df = create_full_fitness_dataframe()

organized_perturbation_fitness_df = create_delta_fitness_matrix(batches, fitness_df, environment_dict)


# Define your modularity function
def calculate_modularity(matrix, labels):
    A = matrix.to_numpy()
    total_weight = np.sum(A)
    degrees = np.sum(A, axis=1)
    Q = 0
    for i in range(len(A)):
        for j in range(len(A)):
            if labels[matrix.columns[i]] == labels[matrix.columns[j]]:
                expected_weight = (degrees[i] * degrees[j]) / total_weight
                Q += A[i, j] - expected_weight
    Q /= total_weight
    return Q

# all_perturbations = ['30', '1.5', 'M3', 'M3b4', '50uMParomomycin', '10uMH89', 'Raf', '4uMH89', 'RafBaffle', '1.4', '1.4Baffle', '10uMParomomycin', '1.8', '30Baffle', 'NS', '28', '1.8Baffle', '0.5%EtOH', 'SucBaffle', '32', 'Suc', '32Baffle']

all_perturbations = ['30', '1.5', 'M3', 'M3b4', '30Baffle', '28', '32', '32Baffle', '1.4', '1.4Baffle', '1.8', '1.8Baffle', 'Suc', 'SucBaffle', 'Raf', 'RafBaffle', 'NS', '0.5%EtOH', '10uMH89', '4uMH89', '10uMParomomycin', '50uMParomomycin']
pert_label_mapping = {'32Baffle': '32°C, Baffle', 'M3b4': 'Batch 4 Base', 'M3': 'Batch 3 Base','1.5': 'Batch 2 Base', '30': 'Batch 1 Base', '50uMParomomycin': '50uM Paro', '10uMH89': '10uM H89',
                      'Raf':'Raffinose','4uMH89': '4uM H89' ,'RafBaffle':'Raffinose, Baff', '1.4':'1.4% Gluc', '1.4Baffle':'1.4% Gluc, Baff','10uMParomomycin': '10uM Paro' ,'1.8':'1.8% Gluc',
                      '0.5%EtOH': '0.5% Ethanol', 'SucBaffle':'Sucrose, Baff', '32': '32°C', 'Suc':'Sucrose', '28': '28°C', 'NS':'No Shake', '30Baffle':'30°C, Baff', '1.8Baffle':'1.8% Gluc, Baff'}
all_perturbations_labels = []
for pert in all_perturbations:
    if pert in pert_label_mapping.keys():
        all_perturbations_labels.append(pert_label_mapping[pert])
    else:
        all_perturbations_labels.append(pert)



base_organized_pearson_similarity = pd.DataFrame(index = twoday_conds+oneday_conds+salt_conds, columns = twoday_conds+oneday_conds+salt_conds)
# organized_perturbation_fitness_df = organized_perturbation_fitness_df.applymap(replace_extinct)
# only look at original mutants
# conds_organized_by_perturbation = [col for col in organized_perturbation_fitness_df.columns if 'fitness' in col]
conds_organized_by_perturbation = []
for pert in all_perturbations:
    for base in ['2Day', '1Day', 'Salt']:
        for col in organized_perturbation_fitness_df.columns:
            if f'{base}_{pert}_fitness' in col:
                conds_organized_by_perturbation.append(col)

print(conds_organized_by_perturbation)
organized_pearson_similarity = pd.DataFrame(index =conds_organized_by_perturbation, columns = conds_organized_by_perturbation)
# organized_perturbation_fitness_df = organized_perturbation_fitness_df.loc[mutant_dict['Original Training']+mutant_dict['Original Testing']]
for col1 in conds_organized_by_perturbation:
    for col2 in conds_organized_by_perturbation:
        if np.linalg.norm(organized_perturbation_fitness_df[col1]) > 0 and np.linalg.norm(organized_perturbation_fitness_df[col2]) > 0:
            organized_pearson_similarity.loc[col1,col2] = pearsonr(organized_perturbation_fitness_df[col1],organized_perturbation_fitness_df[col2])[0]


perturbation_partitions = {}
for col in organized_pearson_similarity.columns:
    pert = col.split('_')[2]  # Extract the perturbation type from the column name
    # Map the column name directly to its partition
    if pert not in perturbation_partitions:
        perturbation_partitions[col] = pert
    else:
        perturbation_partitions[col] = pert
## Calculate modularity for the perturbation partition
perturbation_modularity = calculate_modularity(organized_pearson_similarity, perturbation_partitions)
print(f'Perturbation modularity: {perturbation_modularity}')
# Define the borders between partitions
partition_borders = []
current_partition = None
for idx, col in enumerate(organized_pearson_similarity.columns):
    pert = col.split('_')[2]  # Extract the perturbation type
    if pert != current_partition:
        # Add border if we're transitioning to a new partition
        if current_partition is not None:
            partition_borders.append(idx)
        current_partition = pert
# Draw heatmap
        # Calculate the midpoints for each partition for labeling
partition_midpoints = []
partition_labels = []
current_partition = None
start_idx = 0

for idx, col in enumerate(organized_pearson_similarity.columns):
    pert = col.split('_')[2]  # Extract the perturbation type
    if pert != current_partition:
        if current_partition is not None:
            # Calculate midpoint for the current partition
            midpoint = (start_idx + idx - 1) / 2
            partition_midpoints.append(midpoint)
            partition_labels.append(pert_label_mapping[current_partition])
        current_partition = pert
        start_idx = idx

# Add the last partition's midpoint and label
midpoint = (start_idx + len(organized_pearson_similarity.columns) - 1) / 2
partition_midpoints.append(midpoint)
partition_labels.append(pert_label_mapping[current_partition])

plt.figure(figsize=(10,8))  # Adjust figure size if necessary
sns.heatmap(organized_pearson_similarity.astype(float), cmap='BrBG', center=0, cbar_kws={'shrink': 0.7})

print(organized_pearson_similarity.loc[[col for col in organized_pearson_similarity.columns if '0.5%EtOH' in col],[col for col in organized_pearson_similarity.columns if '0.5%EtOH' in col]])

# Add borders between partitions
for border in partition_borders:
    plt.axhline(border, color='gray', linewidth=1, linestyle='--')  # Horizontal line
    plt.axvline(border, color='gray', linewidth=1, linestyle='--')  # Vertical line

# Set title
plt.title(f'Pearson Correlation, organized by perturbation (Q = {perturbation_modularity:.2f})')



# Update x-axis and y-axis ticks to show partition labels
plt.xticks(partition_midpoints, partition_labels, rotation=90, fontsize=10)
plt.yticks(partition_midpoints, partition_labels, rotation=0, fontsize=10)


# Show and save the plot
plt.tight_layout()
plt.savefig(f'../plots/pearson_correlations_delta_fitness_organized_by_perturbation_abs_with_borders.png')
plt.show()

plt.close()
# # Draw heatmap



for col1 in twoday_conds+oneday_conds+salt_conds:
    for col2 in twoday_conds+oneday_conds+salt_conds:
        if np.linalg.norm(organized_perturbation_fitness_df[col1]) > 0 and np.linalg.norm(organized_perturbation_fitness_df[col2]) > 0:
            base_organized_pearson_similarity.loc[col1,col2] = pearsonr(organized_perturbation_fitness_df[col1],organized_perturbation_fitness_df[col2])[0] 
# drop empty columns and rows
base_organized_pearson_similarity = base_organized_pearson_similarity.dropna(how='all', axis = 0)
base_organized_pearson_similarity = base_organized_pearson_similarity.dropna(how='all', axis=1)

base_partitions = {}
for col in base_organized_pearson_similarity.columns:
    base = col.split('_')[1]  # Extract the perturbation type from the column name
    # Map the column name directly to its partition
    if base not in base_partitions:
        base_partitions[col] = base
    else:
        base_partitions[col] = base

# # Calculate modularity for the base partition
base_modularity = calculate_modularity(base_organized_pearson_similarity, base_partitions)
print(f'Base modularity: {base_modularity}')

base_borders = []
current_partition = None
for idx, col in enumerate(base_organized_pearson_similarity.columns):
    base = col.split('_')[1]  # Extract the base type
    if base != current_partition:
        # Add border if we're transitioning to a new partition
        if current_partition is not None:
            base_borders.append(idx)
        current_partition = base

    # Calculate the midpoints for each base partition for labeling
base_midpoints = []
base_labels = []
current_partition = None
start_idx = 0

for idx, col in enumerate(base_organized_pearson_similarity.columns):
    base = col.split('_')[1]  # Extract the base type
    if base != current_partition:
        if current_partition is not None:
            # Calculate midpoint for the current partition
            midpoint = (start_idx + idx - 1) / 2
            base_midpoints.append(midpoint)
            base_labels.append(current_partition)
        current_partition = base
        start_idx = idx

# Add the last partition's midpoint and label
midpoint = (start_idx + len(base_organized_pearson_similarity.columns) - 1) / 2
base_midpoints.append(midpoint)
base_labels.append(current_partition)

# Draw heatmap
plt.figure(figsize=(10, 8))  # Adjust figure size if necessary
sns.heatmap(base_organized_pearson_similarity.astype(float), cmap='BrBG', center=0, cbar_kws={'shrink': 0.7})

# Add borders between partitions
for border in base_borders:
    plt.axhline(border, color='gray', linewidth=1, linestyle='--')  # Horizontal line
    plt.axvline(border, color='gray', linewidth=1, linestyle='--')  # Vertical line

# Set title
plt.title(f'Pearson Correlation, organized by base (Q = {base_modularity:.2f})')



# Update x-axis and y-axis ticks to show base labels at midpoints
plt.xticks(base_midpoints, base_labels, rotation=0, fontsize=10)
plt.yticks(base_midpoints, base_labels, rotation=0, fontsize=10)


# Show and save the plot
plt.tight_layout()
# plt.savefig(f'../plots/pearson_correlations_delta_fitness_organized_by_base_abs_with_borders.png')
plt.show()

plt.close()

# # fig, axs = plt.subplots(2, 2, figsize=(10,10))

# # # where is the highest correlation that is not the same perturbation
# # max_correlation = 0
# # max_correlation_perturbation = ''
# # for col in base_organized_pearson_similarity.columns:
# #     for row in base_organized_pearson_similarity.index:
# #         if col != row:
#             if base_organized_pearson_similarity.loc[row,col] > max_correlation:
#                 max_correlation = base_organized_pearson_similarity.loc[row,col]
#                 max_correlation_perturbation = (row,col)
# print(max_correlation, max_correlation_perturbation)
# axs[0,0].scatter(organized_perturbation_fitness_df[max_correlation_perturbation[0]], organized_perturbation_fitness_df[max_correlation_perturbation[1]], alpha=0.5)
# axs[0,0].set_title(f'{max_correlation_perturbation[0]} vs {max_correlation_perturbation[1]}')



# # what is lowest correlation that is not the same perturbation
# min_correlation = 1
# min_correlation_perturbation = ''
# for col in base_organized_pearson_similarity.columns:
#     for row in base_organized_pearson_similarity.index:
#         if col != row:
#             if base_organized_pearson_similarity.loc[row,col] < min_correlation:
#                 min_correlation = base_organized_pearson_similarity.loc[row,col]
#                 min_correlation_perturbation = (row,col)
# print(min_correlation, min_correlation_perturbation)
# axs[0,1].scatter(organized_perturbation_fitness_df[min_correlation_perturbation[0]], organized_perturbation_fitness_df[min_correlation_perturbation[1]], alpha=0.5)
# axs[0,1].set_title(f'{min_correlation_perturbation[0]} vs {min_correlation_perturbation[1]}')


# # what is highest correlation that is the same perturbation but nto on the diagonal 
# max_same_correlation = 0
# max_same_correlation_perturbation = ''
# for col in base_organized_pearson_similarity.columns:
#     for row in base_organized_pearson_similarity.index:
#         if col != row and col.split('_')[2] == row.split('_')[2]:
#             if base_organized_pearson_similarity.loc[row,col] > max_same_correlation:
#                 max_same_correlation = base_organized_pearson_similarity.loc[row,col]
#                 max_same_correlation_perturbation = (row,col)
# print(max_same_correlation, max_same_correlation_perturbation)
# axs[1,0].scatter(organized_perturbation_fitness_df[max_same_correlation_perturbation[0]], organized_perturbation_fitness_df[max_same_correlation_perturbation[1]], alpha=0.5)
# axs[1,0].set_title(f'{max_same_correlation_perturbation[0]} vs {max_same_correlation_perturbation[1]}')

# # what is lowest correlation that is the same perturbation but not on the diagonal
# min_same_correlation = 1
# min_same_correlation_perturbation = ''
# for col in base_organized_pearson_similarity.columns:
#     for row in base_organized_pearson_similarity.index:
#         if col != row and col.split('_')[2] == row.split('_')[2]:
#             if base_organized_pearson_similarity.loc[row,col] < min_same_correlation:
#                 min_same_correlation = base_organized_pearson_similarity.loc[row,col]
#                 min_same_correlation_perturbation = (row,col)
# print(min_same_correlation, min_same_correlation_perturbation)
# axs[1,1].scatter(organized_perturbation_fitness_df[min_same_correlation_perturbation[0]], organized_perturbation_fitness_df[min_same_correlation_perturbation[1]], alpha=0.5)
# axs[1,1].set_title(f'{min_same_correlation_perturbation[0]} vs {min_same_correlation_perturbation[1]}')

# plt.tight_layout()
# plt.savefig(f'../plots/scatter_correlations_delta_fitness.png')
# plt.close()

