import numpy as np 
import pandas as pd
from functions import *
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy import stats
import sys
from matplotlib.lines import Line2D
import matplotlib.lines as mlines

import matplotlib.patches as mpatches


plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 14


# Prepare data

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
weights_1day = 1/oneday_bases_error**2
oneday_base_avg = pd.Series((oneday_bases_fitness.values*weights_1day.values).sum(axis = 1)/weights_1day.values.sum(axis = 1), index = oneday_bases_fitness.index)
oneday_base_avg_error = 1/weights_1day.sum(axis = 1)**0.5

twoday_bases_fitness = fitness_df[[ 'Batch2_2Day_1.5_fitness', 'Batch3_2Day_M3_fitness', 'Batch4_2Day_M3b4_fitness']]
twoday_bases_error = fitness_df[[col.replace('fitness', 'stderror') for col in twoday_bases_fitness]]

# take a weighted averge of the base fitnesses and calculate new standard error
weights_2day = 1/twoday_bases_error**2
twoday_base_avg = pd.Series((twoday_bases_fitness.values*weights_2day.values).sum(axis = 1)/weights_2day.values.sum(axis = 1), index = twoday_bases_fitness.index)
twoday_base_avg_error = 1/weights_2day.sum(axis = 1)**0.5

salt_bases_fitness = fitness_df[[ 'Batch1_Salt_30_fitness','Batch2_Salt_1.5_fitness', 'Batch3_Salt_M3_fitness']]
salt_bases_error = fitness_df[[col.replace('fitness', 'stderror') for col in salt_bases_fitness]]

# take a weighted averge of the base fitnesses and calculate new standard error
weights_salt = 1/salt_bases_error**2
salt_base_avg = pd.Series((salt_bases_fitness.values*weights_salt.values).sum(axis = 1)/weights_salt.values.sum(axis = 1), index = salt_bases_fitness.index)
salt_base_avg_error = 1/weights_salt.sum(axis = 1)**0.5


pert_colors = {'M3b4': (0.09009161, 0.04664706, 0.08105568), '1.5': (0.13317849, 0.07976988, 0.14465095),
                'M3': (0.17243586, 0.11091443, 0.21221911), '30': (0.20469116, 0.14319196, 0.28359861), 
                '50uMParomomycin': (0.22953569, 0.17738041, 0.35757941), '10uMH89': (0.2467815, 0.21348266, 0.43393244),
                  'Raf': (0.25412435, 0.25262843, 0.51007843), 'RafBaffle': (0.24436202, 0.30357852, 0.57519929), 
                  '4uMH89': (0.22698081, 0.35544612, 0.60642528), '1.4': (0.21435467, 0.40649717, 0.62167507), 
                  '1.4Baffle': (0.20882976, 0.45564335, 0.63219599), '10uMParomomycin': (0.20552612, 0.50390766, 0.64325811), 
                  '30Baffle': (0.20353258, 0.55199854, 0.65452542), '1.8': (0.20500731, 0.60017798, 0.66472159), 
                  'NS': (0.21430361, 0.64842404, 0.67268861), '28': (0.23946645, 0.70087194, 0.67788018), 
                  '1.8Baffle': (0.27822463, 0.74860229, 0.67914468), '0.5%EtOH': (0.34182112, 0.79437948, 0.67709394),
                    '32': (0.45189421, 0.83240398, 0.68021733), 'SucBaffle': (0.57888639, 0.86256715, 0.70802931), 
                    'Suc': (0.68991419, 0.89240612, 0.75687187), '32Baffle': (0.78462009, 0.92466151, 0.82514119)}


barcode = 'CGCTAAAGACATAATGTGGTTTGTTG_TCCATAATTGGGAATTGGATTTTGGC'


# colors = sns.color_palette("mako", len(all_perts)+1)
# pert_colors = {perturbation: color for perturbation, color in zip(all_perts, colors)}


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
pert_label_mapping = {'32Baffle': '32°C, Baff', 'M3b4': 'Batch 4 Base', 'M3': 'Batch 3 Base','1.5': 'Batch 2 Base', '30': 'Batch 1 Base', '50uMParomomycin': '50uM Paro', '10uMH89': '10uM H89',
                      'Raf':'Raffinose','4uMH89': '4uM H89' ,'RafBaffle':'Raffinose, Baff', '1.4':'1.4% Glucose', '1.4Baffle':'1.4% Glucose, Baff','10uMParomomycin': '10uM Paro' ,'1.8':'1.8% Glucose',
                      '0.5%EtOH': '0.5% Ethanol', 'SucBaffle':'Sucrose, Baff', '32': '32°C', 'Suc':'Sucrose', '28': '28°C', 'NS':'No Shake', '30Baffle':'30°C, Baff', '1.8Baffle':'1.8% Glucose, Baff'}

#

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




def build_delta_fitness_pairplot_df(
    barcode,
    all_perts,
    organized_perturbation_fitness_df,
    all_conds,
    fitness_df,
    oneday_base_avg,
    oneday_base_avg_error,
    twoday_base_avg,
    twoday_base_avg_error,
    salt_base_avg,
    salt_base_avg_error
):
    records = []

    for pert in all_perts:
        entry = {'perturbation': pert, 'color':pert_colors[pert]}
        for env, base_avg, base_error in zip(
            ['1Day', '2Day', 'Salt'],
            [oneday_base_avg, twoday_base_avg, salt_base_avg],
            [oneday_base_avg_error, twoday_base_avg_error, salt_base_avg_error]
        ):
            cols = [col for col in all_conds if f'{pert}_' in col and env in col]
            error_cols = [col for col in fitness_df.columns if f'{pert}_stderror' in col and env in col]
            if not cols:
                continue

            delta_val= organized_perturbation_fitness_df.loc[barcode, cols].values[0]
            delta_std = organized_perturbation_fitness_df.loc[barcode, error_cols].values[0]

            entry[env] = delta_val
            entry[f"{env}_error"] = delta_std

        records.append(entry)

    return pd.DataFrame(records)



def get_rep1_vs_rep2_points(barcode, all_perts, fitness_df, oneday_base_avg):
    r1_vals, r2_vals = [], []
    r1_errs, r2_errs  = [],[]
    for pert in all_perts:
        r1_cols = [col for col in fitness_df.columns if f'{pert}-R1_fitness' in col and '1Day' in col]
        r2_cols = [col for col in fitness_df.columns if f'{pert}-R2_fitness' in col and '1Day' in col]
        r1_cols_err = [col for col in fitness_df.columns if f'{pert}-R1_stderror' in col and '1Day' in col]
        r2_cols_err = [col for col in fitness_df.columns if f'{pert}-R2_stderror' in col and '1Day' in col]

        if r1_cols and r2_cols:
            r1 = (fitness_df.loc[barcode, r1_cols] - oneday_base_avg.loc[barcode] ) #/np.abs(oneday_base_avg.loc[barcode])).values[0]
            r2 = fitness_df.loc[barcode, r2_cols] - oneday_base_avg.loc[barcode]#/np.abs(oneday_base_avg.loc[barcode])).values[0]
            r1_raw_error = fitness_df.loc[barcode, r1_cols_err].values[0]
            r2_raw_error = fitness_df.loc[barcode, r2_cols_err].values[0]

            r1_vals.append(r1)
            r2_vals.append(r2)
            r1_errs.append(r1_raw_error)
            r2_errs.append(r2_raw_error)
    return np.array(r1_vals).flatten(), np.array(r2_vals).flatten(), np.array(r1_errs), np.array(r2_errs)




# Create the DataFrame
pairplot_df = build_delta_fitness_pairplot_df(
    barcode,
    all_perts,
    organized_perturbation_fitness_df,
    all_conds,
    fitness_df,
    oneday_base_avg=oneday_base_avg,
    oneday_base_avg_error=oneday_base_avg_error, 
    twoday_base_avg=twoday_base_avg,
    twoday_base_avg_error=twoday_base_avg_error,
    salt_base_avg=salt_base_avg,
    salt_base_avg_error=salt_base_avg_error
)
# Ensure all expected columns are present
for col in ['Salt', '1Day', '2Day']:
    if col not in pairplot_df.columns:
        pairplot_df[col] = np.nan
    if f"{col}_error" not in pairplot_df.columns:
        pairplot_df[f"{col}_error"] = np.nan

rep1_vals, rep2_vals, rep1_errs, rep2_errs = get_rep1_vs_rep2_points(barcode, all_perts, fitness_df, oneday_base_avg)




## PLOt the figure 

# Create the main figure with a larger size
fig = plt.figure(figsize=(18, 9))

# Create two subfigures with adjusted widths
subfigs = fig.subfigures(1, 2, wspace=0.0, width_ratios=[2.9, 1])

# Left subfigure (scatter plots)
axsLeft = subfigs[0].subplots(2, 2, sharey='row', sharex='col')
subfigs[0].subplots_adjust(hspace=0.1, wspace=0.1)

# Right subfigure (histograms)
axsRight = subfigs[1].subplots(3, 1, sharex=True)
subfigs[1].subplots_adjust(hspace=0.1)

for i,j in [(0,0), (1,0), (1,1)]:
    axsLeft[i,j].axhline(0, color ='black')
    axsLeft[i,j].axvline(0, color ='black')
    axsLeft[i,j].errorbar(rep1_vals, rep2_vals, xerr = rep1_errs, yerr= rep2_errs, fmt = 'o', color = 'gray', alpha = 0.5)

for idx, row in pairplot_df.iterrows():
    axsLeft[0, 0].errorbar(
        row['1Day'], row['2Day'],
        xerr=row['1Day_error'], yerr=row['2Day_error'],
        fmt='o', color=row['color']
    )
    axsLeft[1, 0].errorbar(
        row['1Day'], row['Salt'],
        xerr=row['1Day_error'], yerr=row['Salt_error'],
        fmt='o', color=row['color']
    )
    axsLeft[1, 1].errorbar(
        row['2Day'], row['Salt'],
        xerr=row['2Day_error'], yerr=row['Salt_error'],
        fmt='o', color=row['color']
    )

axsLeft[0,1].axis('off')
for i,j in [(0,0), (1,0), (1,1)]:
    xlims = axsLeft[i,j].get_xlim()
    ylims=axsLeft[i,j].get_xlim()
    xrange = np.linspace(xlims[0]-1, xlims[1]+1, 100)
    yrange = np.linspace(ylims[0]-1, ylims[1]+1, 100)
    axsLeft[i,j].plot(xrange, yrange, linestyle = '--',linewidth = 1,  color  = 'black')




axsLeft[0,0].set_xlim(-0.5,0.4)
axsLeft[0,0].set_ylim(-0.5,0.4)


# unique_perturbations = pairplot_df[['perturbation', 'color']].drop_duplicates()
unique_perturbations = pairplot_df[['perturbation', 'color']].drop_duplicates().sort_values('perturbation') 
# Start with "Replicate" gray dot

# Then add one for each perturbation
legend_handles = [
    Line2D(
        [0], [0],
        marker='o',
        color='none',
        markerfacecolor=pert_colors[pert],
        markeredgewidth=0,
        markeredgecolor='none',
        markersize=10,
        label=pert_label_mapping[pert]
    )
    for pert in pert_colors
]
legend_handles += [
    Line2D(
        [0], [0],
        marker='o',
        color='none',
        markerfacecolor='gray',
        markeredgewidth=0,
        markeredgecolor='none',
        markersize=10,
        alpha = 0.5,
        label='Replicates'
    )
]
axsLeft[0, 1].axis('off')
axsLeft[0, 1].legend(handles=legend_handles, loc='center', ncol=3, frameon=True)

axsLeft[1,0].set_xlim(-0.6,0.4)
axsLeft[1,0].set_ylim(-1,2)

axsLeft[1,1].set_xlim(-0.6,0.4)
axsLeft[1,1].set_ylim(-1,2)




colors = sns.color_palette("rocket", n_colors=10)

print('Using KS test between replicates and deviations')
print('Salt-2Day vs replicates')
# Perform the KS test
statistic, p_value = stats.ks_2samp(rep_rep, twoday_salt)
print(f"KS Statistic: {statistic:.4f}")
print(f"P-value: {p_value}")
print('1Day-2Day vs replicates')
# Perform the KS test
statistic, p_value = stats.ks_2samp(rep_rep, oneday_twoday)
print(f"KS Statistic: {statistic:.4f}")
print(f"P-value: {p_value}")

print('Salt-1Day vs replicates')
# Perform the KS test
statistic, p_value = stats.ks_2samp(rep_rep, oneday_salt)
print(f"KS Statistic: {statistic:.4f}")
print(f"P-value: {p_value}")

print('should be same')
statistic, p_value = stats.ks_2samp(rep_rep, rep_rep)
print(f"KS Statistic: {statistic:.4f}")
print(f"P-value: {p_value}")


# Plotting for the right subfigure with KS test results
datasets = [
    (oneday_twoday, r'$\delta X_{p}^{1Day}\mathbf{-}\delta X_{p}^{2Day}$', colors[3]),
    (oneday_salt, r'$\delta X_{p}^{1Day}\mathbf{-}\delta X_{p}^{Salt}$', colors[5]),
    (twoday_salt, r'$\delta X_{p}^{2Day}\mathbf{-}\delta X_{p}^{Salt}$', colors[8])
]

for i, (data, label, color) in enumerate(datasets):
    ax = axsRight[i]

    sns.kdeplot(np.array(rep_rep), color='gray', label=r'$\delta X_{Rep1}\mathbf{-}\delta X_{Rep2}$', ax=ax, alpha=0.25, fill=True)

    sns.kdeplot(np.array(data), color=color, label=label, ax=ax, fill=True)

    ax.set_ylabel('Density', fontsize=14)


        # Set labels and legend
    legend_right = ax.legend(fontsize=10, loc='upper left')
        # Adjust legend line length
    ax.set_xlim(-1.5, 1.5)
    # ax.set_ylim(0, 1)
    ax.set_ylim(0,2.6)

    # Hide x-axis for all but the bottom subplot
    if i < 2:
        ax.set_xticklabels([])
        ax.set_xlabel('')
    else:
        ax.set_xlabel('Differences in perturbation fitness effect,\n' + r'$\delta X_{p}^{Base 1}\mathbf{-}\delta X_{p}^{Base 2}$', 
                      fontsize=14, labelpad=10)
        # ax.set_xticklabels(np.linspace(-1.5, 1.5, 7), fontsize = 12)
        # Define tick positions and labels
        tick_positions = np.linspace(-1.5, 1.5, 7)
        tick_labels = [f"{x:.1f}" for x in tick_positions]

        # Set positions first, then labels
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=14)



# quantiles of rep vs rep
rep_rep_quantiles = np.quantile(rep_rep, [0.05, 0.5, 0.95])
# plot vertical lines
for ax in axsRight:
    ax.axvline(x=rep_rep_quantiles[0], color='gray', linestyle='--', alpha = 0.5, label='5th percentile')
    ax.axvline(x=rep_rep_quantiles[1], color='gray', linestyle='--', alpha = 0.5, label='50th percentile')
    ax.axvline(x=rep_rep_quantiles[2], color='gray', linestyle='--', alpha = 0.5, label='95th percentile')

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




# Adjust labels and titles
for i, j in [(0, 0), (1, 0), (1, 1)]:
    axsLeft[i, j].set_xlabel('')
    axsLeft[i, j].set_ylabel('')


# Move legend to a better position
legend = axsLeft[0, 1].legend(handles=legend_handles, loc='center', ncol=2, frameon=False, fontsize=12)
legend.set_in_layout(False)  # Allows legend to be placed outside the axis

axsLeft[0, 0].set_ylabel(r'Pert. fitness effect in 2 Day base, $\delta X_{p}^{2Day}$', fontsize=14)
axsLeft[1, 0].set_xlabel(r'Pert. fitness effect in 1 Day base, $\delta X_{p}^{1Day}$', fontsize=14)
axsLeft[1, 0].set_ylabel(r'Pert. fitness effect in Salt base, $\delta X_{p}^{Salt}$', fontsize=14)
axsLeft[1, 1].set_xlabel(r'Pert. fitness effect in 2 Day base, $\delta X_{p}^{2Day}$', fontsize=14)



# Add titles to subfigures
subfigs[0].suptitle('Pairwise Comparisons of Perturbation Fitness Effects', fontsize=20)
subfigs[1].suptitle('Distribution of differences\n in perturbation effects', fontsize=20)

plt.savefig('plots/fig3.png', dpi=300)
# plt.show()

plt.close()




## add plot with raw fitness values 

one_day_bases = fitness_df.loc[barcode,['Batch1_1Day_30_fitness', 'Batch2_1Day_1.5_fitness', 'Batch3_1Day_M3_fitness', 'Batch4_1Day_M3b4_fitness']]
two_day_bases = fitness_df.loc[barcode,['Batch2_2Day_1.5_fitness', 'Batch3_2Day_M3_fitness', 'Batch4_2Day_M3b4_fitness']]
salt_bases = fitness_df.loc[barcode,['Batch1_Salt_30_fitness', 'Batch2_Salt_1.5_fitness', 'Batch3_Salt_M3_fitness']]
one_day_errors = fitness_df.loc[barcode,['Batch1_1Day_30_stderror', 'Batch2_1Day_1.5_stderror', 'Batch3_1Day_M3_stderror', 'Batch4_1Day_M3b4_stderror']]
two_day_errors = fitness_df.loc[barcode,['Batch2_2Day_1.5_stderror', 'Batch3_2Day_M3_stderror', 'Batch4_2Day_M3b4_stderror']]
salt_errors= fitness_df.loc[barcode,['Batch1_Salt_30_stderror', 'Batch2_Salt_1.5_stderror', 'Batch3_Salt_M3_stderror']]

np.random.seed(0)

x_ticks = [1,3,5]

plt.figure(figsize=(3,2))
colors_bases = [env_color_dict['1Day'], env_color_dict['2Day'], env_color_dict['Salt']]
for i, (base, err) in enumerate([(one_day_bases, one_day_errors), (two_day_bases, two_day_errors), (salt_bases, salt_errors)]):
    for (cond,err) in zip(base.values, err.values):
        plt.errorbar(x_ticks[i]+np.random.normal(0,0.1,1)[0], cond, yerr = err, marker = 'o',color = colors_bases[i],  alpha = 0.75)
plt.axhline(0, color = 'k', linestyle='--')
plt.xticks([])
plt.axvspan(2,4,color='lightgray', alpha = 0.5)
# plt.xticks(x_ticks, (['1 Day \n Base', '2 Day \n Base', 'Salt \n Base']))
# plt.title('Raw fitness of mutant 1 in base environments')
plt.ylabel(r'Rel. fitness, $X$')
plt.tight_layout()
plt.savefig('plots/fig3a.png', dpi = 200)


print(organized_perturbation_fitness_df.loc[barcode, [col for col in organized_perturbation_fitness_df.columns if 'NS_fitness' in col]])
print(organized_perturbation_fitness_df.loc[barcode, [col for col in organized_perturbation_fitness_df.columns if '0.5%EtOH_fitness' in col]])