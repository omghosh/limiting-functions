import numpy as np
import pandas as pd
from numpy import random 
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from functions import *
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# sns.set_style('darkgrid')
plt.rcParams['font.family'] = 'Helvetica'

plt.rcParams['font.size'] = 14
bc_counts, fitness_df, grants_df_with_barcode_df, full_df = create_full_fitness_dataframe()
fitness_df = fitness_df.applymap(replace_extinct)
base_perturbations = ['30','1.5', 'M3', 'M3b4']
all_perturbations=[]
z_scores_dict = {}
external_z_scores_dict = {}
# there are several ways we can calculate a z score 
# the way grant did it was to look at the batches of the evolution condition for the sigma per mutant
for i, base_env in enumerate(environment_dict.keys()):
    print(f'Starting {base_env}')
    home_conds = []
    for cond in environment_dict[base_env]:
        perturbation = cond.split('_')[2]
        all_perturbations.append(perturbation)
        if perturbation in base_perturbations:
            home_conds.append(cond)
    
    means = fitness_df[home_conds].mean(axis=1)
    stds_observed = fitness_df[home_conds].std(axis=1)
    stds_inferred = fitness_df[[condition.replace('fitness', 'stderror') for condition in home_conds]].mean(axis=1)

    for cond in environment_dict[base_env]:
        z_scores = np.abs(fitness_df[cond] - means)/np.sqrt(stds_observed**2 + stds_inferred**2)
        z_scores_dict[cond] = z_scores
    for external_cond in all_conds:
        if external_cond not in environment_dict[base_env]:
            z_scores = np.abs(fitness_df[external_cond] - means)/np.sqrt(stds_observed**2 + stds_inferred**2)
            external_z_scores_dict[f'{external_cond}_from_{base_env}'] = z_scores


z_scores_df = pd.DataFrame(z_scores_dict)
# get average z score for each perturbation
z_scores_avgs = z_scores_df.mean()
z_scores_stds = z_scores_df.std()
sorted_zscores_avgs = pd.Series()
sorted_zscores_stds = pd.Series()

for env in environment_dict.keys():
    these_envs = [cond for cond in z_scores_avgs.index if env in cond]
    base_replicates = [cond for cond in these_envs if cond.split('_')[2] in (base_perturbations)]
    these_sorted_zscores_bases = z_scores_avgs[base_replicates].sort_values(ascending=True)
    these_sorted_zscores=pd.concat([these_sorted_zscores_bases,z_scores_avgs[[env for env in these_envs if env not in base_replicates]].sort_values(ascending=True)])
    sorted_zscores_avgs = pd.concat([sorted_zscores_avgs, these_sorted_zscores])
    these_sorted_zscores_stds = z_scores_stds[these_envs].loc[these_sorted_zscores.index]
    sorted_zscores_stds = pd.concat([sorted_zscores_stds, these_sorted_zscores_stds])

twoday_zscore_avgs = {}
oneday_zscore_avgs = {}
salt_zscore_avgs = {}
twoday_zscore_stds = {}
oneday_zscore_stds = {}
salt_zscore_stds = {}

for cond in sorted_zscores_avgs.index:
    base = cond.split('_')[1]
    perturbation = cond.split('_')[2]
    if base == '2Day':
        if perturbation in twoday_zscore_avgs.keys():
            perturbation = f'{perturbation}_2'
        twoday_zscore_avgs[perturbation] = sorted_zscores_avgs[cond]
        twoday_zscore_stds[perturbation] = sorted_zscores_stds[cond]
    elif '1Day' in cond:
        if perturbation in oneday_zscore_avgs.keys():
            perturbation = f'{perturbation}_2'
        oneday_zscore_avgs[perturbation] = sorted_zscores_avgs[cond]
        oneday_zscore_stds[perturbation] = sorted_zscores_stds[cond]
    elif 'Salt' in cond:
        if perturbation in salt_zscore_avgs.keys():
            perturbation = f'{perturbation}_2'
        salt_zscore_avgs[perturbation] = sorted_zscores_avgs[cond]
        salt_zscore_stds[perturbation] = sorted_zscores_stds[cond]


external_z_scores_df = pd.DataFrame(external_z_scores_dict)
external_z_scores_avgs = external_z_scores_df.mean()
external_z_scores_stds = external_z_scores_df.std()
for this_base in environment_dict.keys():
    these_envs = [cond for cond in external_z_scores_avgs.index if f'from_{this_base}' in cond]
    for external_base in environment_dict.keys():
        if this_base != external_base:
            conds_to_avg = [cond for cond in these_envs if f'{external_base}_' in cond]
            # only take perturbations that are in base_perturbations list
            conds_to_avg = [cond for cond in conds_to_avg if cond.split('_')[2] in base_perturbations]
            avg_other_base = external_z_scores_avgs[conds_to_avg].mean()
            print(f'{this_base} vs {external_base} z-score: {avg_other_base}')
# create a colormap corresponding to perturbations in the same order as the one day z scores 
colors = sns.color_palette("mako", len(oneday_zscore_avgs)+1)
perturbation_colors = {perturbation: color for perturbation, color in zip(oneday_zscore_avgs.keys(), colors)}
perturbation_colors['32Baffle']= sns.color_palette("mako", len(oneday_zscore_avgs)+1)[-1]


all_perturbations = (list(set(all_perturbations)))

# add in any perturbations that are in all_perturbations but not in one day z scores
all_perturbations = [perturbation for perturbation in oneday_zscore_avgs.keys() if perturbation in all_perturbations] + [perturbation for perturbation in all_perturbations if perturbation not in oneday_zscore_avgs.keys()]
# put the 4 base perturbations at the beginning of the list
all_perturbations = base_perturbations + [perturbation for perturbation in all_perturbations if perturbation not in base_perturbations]
# print(all_perturbations)
# change some of the pertubation labels to make them more readable
#pert_label_mapping = {'32Baffle': '32°C, Baffle', 'M3b4': 'Batch 4 Base', 'M3': 'Batch 3 Base','1.5': 'Batch 2 Base', '30': 'Batch 1 Base', '50uMParomomycin': '50uM Paro', '10uMH89': '10uM H89',
 #                     'Raf':'Raffinose','4uMH89': '4uM H89' ,'RafBaffle':'Raffinose, Baffle', '1.4':'1.4% Glucose', '1.4Baffle':'1.4% Glucose, Baffle','10uMParomomycin': '10uM Paro' ,'1.8':'1.8% Glucose',
 #                     '0.5%EtOH': '0.5% Ethanol', 'SucBaffle':'Sucrose, Baffle', '32': '32°C', 'Suc':'Sucrose', '28': '28°C', 'NS':'No Shake', '30Baffle':'30°C, Baffle', '1.8Baffle':'1.8% Glucose, Baffle'}
pert_label_mapping = {'32Baffle': '32°C, Baff', 'M3b4': 'Batch 4 Base', 'M3': 'Batch 3 Base','1.5': 'Batch 2 Base', '30': 'Batch 1 Base', '50uMParomomycin': '50uM Paro', '10uMH89': '10uM H89',
                      'Raf':'Raffinose','4uMH89': '4uM H89' ,'RafBaffle':'Raffinose, Baff', '1.4':'1.4% Glucose', '1.4Baffle':'1.4% Glucose, Baff','10uMParomomycin': '10uM Paro' ,'1.8':'1.8% Glucose',
                      '0.5%EtOH': '0.5% Ethanol', 'SucBaffle':'Sucrose, Baff', '32': '32°C', 'Suc':'Sucrose', '28': '28°C', 'NS':'No Shake', '30Baffle':'30°C, Baff', '1.8Baffle':'1.8% Glucose, Baff'}

#
all_perturbations_labels = []
for pert in all_perturbations:
    if pert in pert_label_mapping.keys():
        all_perturbations_labels.append(pert_label_mapping[pert])
    else:
        all_perturbations_labels.append(pert)

fig, axs = plt.subplots(3, 2, figsize=(8, 6), gridspec_kw={'width_ratios': [3.5,1]}, dpi=300)

for i, perturbation in enumerate(all_perturbations):
    if perturbation in oneday_zscore_avgs:
        axs[0, 0].scatter(i, oneday_zscore_avgs[perturbation], color=perturbation_colors[perturbation])
        axs[0, 0].errorbar(i, oneday_zscore_avgs[perturbation], yerr=oneday_zscore_stds.get(perturbation, 0), color=perturbation_colors[perturbation])
        axs[0, 0].set_xticks(np.arange(len(all_perturbations)))
        axs[0, 0].set_xticklabels([])
        # axs[0,0].set_ylabel('1 Day', fontsize = 12)
    if perturbation in twoday_zscore_avgs:
        axs[1, 0].scatter(i, twoday_zscore_avgs[perturbation], color=perturbation_colors[perturbation])
        axs[1, 0].errorbar(i, twoday_zscore_avgs[perturbation], yerr=twoday_zscore_stds.get(perturbation, 0), color=perturbation_colors[perturbation])
        axs[1, 0].set_xticks(np.arange(len(all_perturbations)))
        axs[1, 0].set_xticklabels([])
        # axs[1,0].set_ylabel('2 Day Average z-score', fontsize = 12)
    if perturbation in salt_zscore_avgs:
        axs[2, 0].scatter(i, salt_zscore_avgs[perturbation], color=perturbation_colors[perturbation])
        axs[2, 0].errorbar(i, salt_zscore_avgs[perturbation], yerr=salt_zscore_stds.get(perturbation, 0), color=perturbation_colors[perturbation])
        axs[2, 0].set_xticks(np.arange(len(all_perturbations)))
        
        axs[2, 0].set_xticklabels(all_perturbations_labels, rotation=90)
        # axs[2,0].set_ylabel('Salt Average z-score', fontsize = 12)
for ax in axs[:, 0]:
    ax.set_xticks(np.arange(len(all_perturbations)))
    # ax.set_xticklabels(all_perturbations, rotation=45, fontsize=8)
    ax.set_xlim(-0.5, len(all_perturbations) - 0.5)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim(-1, 14)
    ax.set_yticks(np.arange(0, 15, 5))

    # add grey box behind the first 4 perturbations 
    ax.axvspan(-0.5, 3.5, color='grey', alpha=0.1)

np.random.seed(1)

# Collect data points for jittering for external base z scores
for e, this_base in enumerate(['1Day', '2Day', 'Salt']):
    these_envs = [cond for cond in external_z_scores_avgs.index if f'from_{this_base}' in cond]
    for i,external_base in enumerate(['1Day', '2Day', 'Salt']):
        if  this_base != external_base:
            conds_to_plot = [cond for cond in these_envs if f'{external_base}_' in cond]
                # only take perturbations that are in base_perturbations list
            conds_to_plot = [cond for cond in conds_to_plot if cond.split('_')[2] in base_perturbations]
            these_scores = external_z_scores_avgs[conds_to_plot]
            # do a jitter plot with these scores
            x_vals = np.random.normal(i, 0.1, len(these_scores))
            axs[e,1].scatter(x_vals, these_scores, color=env_color_dict[external_base], alpha = 0.75)
            j=0
            for x_val, con in zip(x_vals,conds_to_plot):
                axs[e,1].errorbar(x_val, these_scores[j], yerr=external_z_scores_stds[con], color=env_color_dict[external_base], alpha = 0.5)
                j+=1


    # Formatting adjustments (similar to original plot)
    axs[e, 1].set_xlim(-0.5, (2) + 0.5)  # Creates extra space around first and last labels
    axs[e, 1].set_xticks(np.arange(3))
    axs[e, 1].set_yticklabels([])
    axs[e, 1].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    axs[e, 1].set_ylim(-1, 14)
    axs[e, 1].set_yticks(np.arange(0, 15, 5))

    if e ==2:
        axs[e, 1].set_xticklabels([f'{base_name} base' for base_name in ['1Day', '2Day', 'Salt']], rotation=90)
    else:
        axs[e, 1].set_xticklabels([])   
        axs[e, 1].set_xlabel('')

# fig.supxlabel('Perturbations', fontsize=12)
# fig.supylabel('Average Z-score', fontsize=12)
plt.tight_layout()
# plt.show()
plt.savefig('plots/fig2d.png')


