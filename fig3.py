import numpy as np 
import pandas as pd
from functions import *
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy import stats
import sys
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches


plt.rcParams['font.family'] = 'Helvetica'
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


barcode = 'CGCTAAAGACATAATGTGGTTTGTTG_TCCATAATTGGGAATTGGATTTTGGC'


colors = sns.color_palette("mako", len(all_perts)+1)
pert_colors = {perturbation: color for perturbation, color in zip(all_perts, colors)}





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

            abs_base = np.abs(base_avg.loc[barcode])
            base_err = base_error.loc[barcode]

            # Scaled delta fitness
            delta_fitness = delta_val / abs_base

            # # Propagate error
            # delta_fitness_error = np.sqrt(
            #     (delta_std / abs_base) ** 2 + 
            #     (delta_val / abs_base**2) ** 2 * base_err**2
            # )
            delta_fitness_error = np.sqrt(delta_std**2 + base_err**2)

            entry[env] = delta_fitness
            entry[f"{env}_error"] = delta_fitness_error

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
            r1_pert =(fitness_df.loc[barcode, r1_cols]).values[0]
            r2_pert = (fitness_df.loc[barcode, r2_cols]).values[0]
            abs_base_avg = np.abs(oneday_base_avg.loc[barcode])
            avg_base_error = oneday_base_avg_error.loc[barcode]

            r1_error = np.sqrt(r1_raw_error**2 + avg_base_error**2)
            r2_error = np.sqrt(r2_raw_error**2 + avg_base_error**2)

            ## use propagation of error to get error on scaled delta fitness estimates 
            #r1_error = ((r1_raw_error/ abs_base_avg))**2  + (r1_pert/ abs_base_avg**2)**2 * avg_base_error**2
            #r2_error = ((r2_raw_error/ abs_base_avg))**2  + (r2_pert/ abs_base_avg**2)**2 * avg_base_error**2
            r1_vals.append(r1)
            r2_vals.append(r2)
            r1_errs.append(r1_error)
            r2_errs.append(r2_error)
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
print(pairplot_df)
# Ensure all expected columns are present
for col in ['Salt', '1Day', '2Day']:
    if col not in pairplot_df.columns:
        pairplot_df[col] = np.nan
    if f"{col}_error" not in pairplot_df.columns:
        pairplot_df[f"{col}_error"] = np.nan

rep1_vals, rep2_vals, rep1_errs, rep2_errs = get_rep1_vs_rep2_points(barcode, all_perts, fitness_df, oneday_base_avg)


fig, axs = plt.subplots(2,2, figsize =(12,10), sharex='col', sharey='row')
fig.subplots_adjust(wspace=0.07, hspace=0.14)  # adjust as needed

for i,j in [(0,0), (1,0), (1,1)]:
    axs[i,j].axhline(0, color ='gray')
    axs[i,j].axvline(0, color ='gray')
    axs[i,j].errorbar(rep1_vals, rep2_vals, xerr = rep1_errs, yerr= rep2_errs, fmt = 'o', color = 'gray', alpha = 0.5)

for idx, row in pairplot_df.iterrows():
    axs[0, 0].errorbar(
        row['1Day'], row['2Day'],
        xerr=row['1Day_error'], yerr=row['2Day_error'],
        fmt='o', color=row['color']
    )
    axs[1, 0].errorbar(
        row['1Day'], row['Salt'],
        xerr=row['1Day_error'], yerr=row['Salt_error'],
        fmt='o', color=row['color']
    )
    axs[1, 1].errorbar(
        row['2Day'], row['Salt'],
        xerr=row['2Day_error'], yerr=row['Salt_error'],
        fmt='o', color=row['color']
    )

axs[0,0].axvline(oneday_base_avg.loc[barcode], linestyle= '--', color = env_color_dict['1Day'])
axs[0,0].axhline(twoday_base_avg.loc[barcode], linestyle= '--',color = env_color_dict['2Day'])

axs[1,0].axvline(oneday_base_avg.loc[barcode], linestyle= '--',color = env_color_dict['1Day'])
axs[1,0].axhline(salt_base_avg.loc[barcode], linestyle= '--',color = env_color_dict['Salt'])

axs[1,1].axvline(twoday_base_avg.loc[barcode], linestyle= '--',color = env_color_dict['2Day'])
axs[1,1].axhline(salt_base_avg.loc[barcode], linestyle= '--',color = env_color_dict['Salt'])

axs[0,1].axis('off')
for i,j in [(0,0), (1,0), (1,1)]:
    xlims = axs[i,j].get_xlim()
    ylims=axs[i,j].get_xlim()
    xrange = np.linspace(xlims[0]-1, xlims[1]+1, 100)
    yrange = np.linspace(ylims[0]-1, ylims[1]+1, 100)
    axs[i,j].plot(xrange, yrange, color  = 'gray')

axs[0,0].set_xlim(-0.5,1.1)
axs[0,0].set_ylim(-0.5,1.1)
axs[0,0].set_ylabel(r'Effect of perturbation in 2 Day base, $\delta X_{p}^{2Day}$', fontsize = 15)
axs[1,0].set_xlabel(r'Effect of perturbation in 1 Day base, $\delta X_{p}^{1Day}$', fontsize = 15)
axs[1,0].set_ylabel(r'Effect of perturbation in Salt base, $\delta X_{p}^{Salt}$', fontsize = 15)
axs[1,1].set_xlabel(r'Effect of perturbation in 2Day base, $\delta X_{p}^{2Day}$', fontsize = 15)

axs[1,0].set_xlim(-0.6,1.1)
axs[1,0].set_ylim(-1,3)


axs[1,1].set_xlim(-0.6,1.1)
axs[1,1].set_ylim(-1,3)




unique_perturbations = pairplot_df[['perturbation', 'color']].drop_duplicates()
# Start with "Replicate" gray dot


# Then add one for each perturbation
legend_handles = [
    Line2D(
        [0], [0],
        marker='o',
        color='none',
        markerfacecolor=row['color'],
        markeredgewidth=0,
        markeredgecolor='none',
        markersize=10,
        label=row['perturbation']
    )
    for _, row in unique_perturbations.iterrows()
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
axs[0, 1].axis('off')
axs[0, 1].legend(handles=legend_handles, loc='center', ncol=2, frameon=False)


plt.tight_layout()
plt.savefig('plots/fig3a.png', dpi = 300)