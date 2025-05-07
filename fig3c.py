
import numpy as np 
import pandas as pd
from functions import *
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy import stats
import sys
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 14

bc_counts, fitness_df, grants_df_with_barcode_df, full_df = create_full_fitness_dataframe()
organized_perturbation_fitness_df= create_delta_fitness_matrix(batches, fitness_df, environment_dict)

# remove all Li_WT
organized_perturbation_fitness_df.loc[mutant_dict['anc: Li_WT']]
organized_perturbation_fitness_df.drop(mutant_dict['anc: Li_WT'], inplace=True)

all_perts  = list(set([col.split('_')[2] for col in organized_perturbation_fitness_df.columns]))


oneday_twoday = []
oneday_salt = []
twoday_salt = []
rep_rep = []

# get rep-rep delta deviation
oneday_base_avg = fitness_df[['Batch1_1Day_30_fitness', 'Batch2_1Day_1.5_fitness', 'Batch3_1Day_M3_fitness', 'Batch4_1Day_M3b4_fitness']].mean(axis = 1)
salt_base_avg = fitness_df[['Batch1_Salt_30_fitness', 'Batch2_Salt_1.5_fitness', 'Batch3_Salt_M3_fitness']].mean(axis = 1)
twoday_base_avg = fitness_df[[ 'Batch2_2Day_1.5_fitness', 'Batch3_2Day_M3_fitness', 'Batch4_2Day_M3b4_fitness']].mean(axis = 1)


for pert in all_perts:
    oneday = [col for col in all_conds if f'{pert}_' in col and '1Day' in col]
    twoday =[col for col in all_conds if f'{pert}_' in col and '2Day' in col]
    salt = [col for col in all_conds if f'{pert}_' in col and 'Salt' in col]
    if len(oneday) != 0 and len(twoday) != 0:
        oneday_twoday.extend((organized_perturbation_fitness_df[oneday[0]]-organized_perturbation_fitness_df[twoday[0]]).values)
    if len(oneday) != 0 and len(salt) != 0:
        oneday_salt.extend((organized_perturbation_fitness_df[oneday[0]]-organized_perturbation_fitness_df[salt[0]]).values)
    if len(twoday) != 0 and len(salt) != 0:
        twoday_salt.extend((organized_perturbation_fitness_df[twoday[0]]-organized_perturbation_fitness_df[salt[0]]).values)

# # choose random environment 
# r1_delta_fitness = fitness_df[f'Batch4_2Day_0.5%EtOH-R1_fitness']-twoday_base_avg
# r2_delta_fitness = fitness_df[f'Batch4_2Day_0.5%EtOH-R2_fitness']-twoday_base_avg
# rep_rep.extend((r1_delta_fitness-r2_delta_fitness).values)
for env in all_conds:
    cond_raw = env.removesuffix("_fitness")
    r1_label = f'{cond_raw}-R1_fitness'
    r2_label =f'{cond_raw}-R2_fitness'
    base = env.split('_')[1]
    if base == '2Day':
        base_avg = twoday_base_avg
    elif base == '1Day':
        base_avg = oneday_base_avg
    elif base == 'Salt':
        base_avg = salt_base_avg
    if r1_label in fitness_df.columns and r2_label in fitness_df.columns:
        r1_delta_fitness = fitness_df[r1_label]-base_avg
        r2_delta_fitness = fitness_df[r2_label]-base_avg
        rep_rep.extend((r1_delta_fitness-r2_delta_fitness).values)


colors = sns.color_palette("rocket", n_colors=10)
# plot 4 different histograms 
fig, axs = plt.subplots(3,1,figsize=(6, 6))
for ax in axs:
    # Plotting CDF for "Rep vs Rep"
    sns.ecdfplot(np.array(rep_rep), color='gray', label='Rep vs Rep', ax=ax, alpha = 0.75)

# Plot CDF for other datasets
sns.ecdfplot(np.array(oneday_twoday), color=colors[3], label='1Day vs 2Day', ax=axs[0])
sns.ecdfplot(np.array(oneday_salt), color=colors[6], label='1Day vs Salt', ax=axs[1])
sns.ecdfplot(np.array(twoday_salt), color=colors[9], label='2Day vs Salt', ax=axs[2])


# for ax in axs:
#     # if ax == axs[0]:
#     #     sns.kdeplot(np.array(rep_rep), color='gray', label='Rep vs Rep',fill=True,  linewidth = 0, common_norm=True, ax=ax)
#     # else:
#         sns.kdeplot(np.array(rep_rep), color='gray', label='Rep vs Rep',fill=True,  linewidth = 0, common_norm=True, ax=ax)
# sns.kdeplot(np.array(oneday_twoday), color='teal', fill = True, label='1Day vs 2Day', alpha = 0.75,  linewidth = 0, common_norm=True, ax=axs[0])
# sns.kdeplot(np.array(oneday_salt), color=env_color_dict['1Day'],fill=True,  label='1Day vs Salt', alpha = 0.75,  linewidth = 0, common_norm=True, ax=axs[1])
# sns.kdeplot(np.array(twoday_salt), color=env_color_dict['2Day'], fill=True, alpha = 0.75,  label='2Day vs Salt',  linewidth = 0, common_norm=True, ax=axs[2])

print('Using KS test between replicates and deviations')
print('Salt-2Day vs replicates')
# Perform the KS test
statistic, p_value = stats.ks_2samp(rep_rep, twoday_salt)
print(f"KS Statistic: {statistic:.4f}")
print(f"P-value: {p_value:.4f}")
print('1Day-2Day vs replicates')
# Perform the KS test
statistic, p_value = stats.ks_2samp(rep_rep, oneday_twoday)
print(f"KS Statistic: {statistic:.4f}")
print(f"P-value: {p_value:.4f}")

print('Salt-1Day vs replicates')
# Perform the KS test
statistic, p_value = stats.ks_2samp(rep_rep, oneday_salt)
print(f"KS Statistic: {statistic:.4f}")
print(f"P-value: {p_value:.4f}")





# make x axes the same
# get x axis limits
x_min = min(np.min(oneday_twoday), np.min(oneday_salt), np.min(twoday_salt), np.min(rep_rep))
x_max = max(np.max(oneday_twoday), np.max(oneday_salt), np.max(twoday_salt), np.max(rep_rep))
y_min = min(np.min(oneday_twoday), np.min(oneday_salt), np.min(twoday_salt), np.min(rep_rep))
y_max = max(np.max(oneday_twoday), np.max(oneday_salt), np.max(twoday_salt), np.max(rep_rep))
for ax in axs:
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(0, 1)
    # ax.set_xticks(np.arange(x_min, x_max, 0.5))
    if ax == axs[2]:
        ax.set_xlabel('Deviation from equal delta fitness')
    else:
        ax.set_xlabel('')
        ax.set_xticklabels([])

# quantiles of rep vs rep
rep_rep_quantiles = np.quantile(rep_rep, [0.05, 0.5, 0.95])
print(f'5th percentile: {rep_rep_quantiles[0]}')
print(f'50th percentile: {rep_rep_quantiles[1]}')
print(f'95th percentile: {rep_rep_quantiles[2]}')

oneday_twoday = np.array(oneday_twoday) 
oneday_salt = np.array(oneday_salt)
twoday_salt = np.array(twoday_salt)


# how much data falls above the 95th percentile and below the 5th percentile in each of the other distributions
print('Above 95th percentile:')
print(f'1Day vs 2Day: {len(oneday_twoday[oneday_twoday > rep_rep_quantiles[2]])/len(oneday_twoday)}')
print(f'1Day vs Salt: {len(oneday_salt[oneday_salt > rep_rep_quantiles[2]])/len(oneday_salt)}')
print(f'2Day vs Salt: {len(twoday_salt[twoday_salt > rep_rep_quantiles[2]])/len(twoday_salt)}')
print('Below 5th percentile:')
print(f'1Day vs 2Day: {len(oneday_twoday[oneday_twoday < rep_rep_quantiles[0]])/len(oneday_twoday)}')
print(f'1Day vs Salt: {len(oneday_salt[oneday_salt < rep_rep_quantiles[0]])/len(oneday_salt)}')
print(f'2Day vs Salt: {len(twoday_salt[twoday_salt < rep_rep_quantiles[0]])/len(twoday_salt)}')

print((len(oneday_twoday[oneday_twoday > rep_rep_quantiles[2]])+ len(oneday_twoday[oneday_twoday < rep_rep_quantiles[0]]))/len(oneday_twoday))
print((len(oneday_salt[oneday_salt > rep_rep_quantiles[2]])+ len(oneday_salt[oneday_salt < rep_rep_quantiles[0]]))/len(oneday_salt))
print((len(twoday_salt[twoday_salt > rep_rep_quantiles[2]])+ len(twoday_salt[twoday_salt < rep_rep_quantiles[0]]))/len(twoday_salt))



# plot vertical lines
for ax in axs:
    ax.axvline(x=rep_rep_quantiles[0], color='gray', linestyle='--', label='5th percentile')
    ax.axvline(x=rep_rep_quantiles[1], color='gray', linestyle='--', label='50th percentile')
    ax.axvline(x=rep_rep_quantiles[2], color='gray', linestyle='--', label='95th percentile')


plt.tight_layout()
plt.savefig('plots/fig3c.png', dpi=300)
# plt.show()


