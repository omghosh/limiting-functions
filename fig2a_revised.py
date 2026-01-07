import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from functions import *
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.ticker import FormatStrFormatter

sns.set_style("white")  # Clean style without grid lines

plt.rcParams['font.family'] = 'Helvetica Neue'
plt.rcParams['font.size'] = 9
plt.rcParams['mathtext.cal'] = 'stix:italic'
plt.rcParams['lines.markersize'] = 4.5
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['lines.markersize'] = 4.5
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.minor.size'] = 2
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['xtick.labelsize'] = 9

plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.minor.size'] = 2
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['ytick.labelsize'] = 9

plt.rcParams['figure.subplot.wspace'] = 0.9
plt.rcParams['figure.subplot.hspace'] = 0.9





bc_counts, fitness_df, grants_df_with_barcode_df, full_df = create_full_fitness_dataframe()
organized_perturbation_fitness_df= create_delta_fitness_matrix(batches, fitness_df, environment_dict)

# remove all Li_WT
organized_perturbation_fitness_df.loc[mutant_dict['anc: Li_WT']]
organized_perturbation_fitness_df.drop(mutant_dict['anc: Li_WT'], inplace=True)


fitness_df = fitness_df.map(replace_extinct)

# Sort rows by fitness
this_fitness_df = fitness_df.sort_values('Batch4_Salt_0.5%EtOH_fitness')

# Get the fitness values
fitness_values = this_fitness_df['Batch4_Salt_0.5%EtOH_fitness'].values

# Define colormap and normalization
cmap = sns.color_palette("coolwarm", as_cmap=True)
absmax=2
norm = mcolors.TwoSlopeNorm(vmin=-absmax, vcenter=0, vmax=absmax)

# Create color mapping for barcodes
fitness_colors = cmap(norm(fitness_values))  # Normalize fitness values

# Create a dictionary mapping barcodes to colors
fitness_color_dict = dict(zip(this_fitness_df['barcode'], fitness_colors))


this_cond = 'Batch4_Salt_0.5%EtOH-R1'

these_conds = ['Batch4_Salt-T0']+[f'Batch4_Salt_0.5%EtOH-R1-T{t}_combined' for t in range(1, 5)]
these_bc_counts = bc_counts[these_conds]
these_freqs = these_bc_counts.div(these_bc_counts.sum(axis=0), axis=1)
these_freqs.index = bc_counts['barcode']


# do any trajectories go to 0 and come back up? 
undetectable = 6e-8
extinct = 4e-8


focal_base = '2Day'
this_fitness = fitness_df[[f'Batch4_{focal_base}_0.5%EtOH-R1_fitness', 
                         f'Batch4_{focal_base}_0.5%EtOH-R2_fitness', 
                         f'Batch4_{focal_base}_0.5%EtOH_fitness',
                         f'Batch4_{focal_base}_0.5%EtOH-R1_stderror', 
                         f'Batch4_{focal_base}_0.5%EtOH-R2_stderror']]
this_fitness = this_fitness.sort_values(f'Batch4_{focal_base}_0.5%EtOH_fitness')

fig, axes = plt.subplots(1, 2, figsize=(3.75, 1.75),  constrained_layout= True, dpi=600)

# ===== LEFT: Barcode Trajectories =====
ax = axes[0]

lines = []
colors = []
for barcode in these_freqs.index:
    if barcode in fitness_color_dict:
        traj = these_freqs.loc[barcode].copy()
        below_threshold = traj == 0
        if (below_threshold).any():
            first_undetectable = below_threshold.idxmax()
            if (traj[first_undetectable:] > 0).any():
                traj[traj == 0] = undetectable
            else:
                traj[traj == 0] = extinct
        x = np.arange(len(traj))
        y = traj.values
        points = np.array([x, y]).T
        lines.append(points)
        colors.append(fitness_color_dict[barcode])

lc = LineCollection(lines, colors=colors, linewidths=0.25, alpha=0.25)
ax.add_collection(lc)
ax.set_yscale('log')
ax.set_xlim(0, 4)
ax.set_ylim(4e-8, 1)
ax.set_xticks(range(0, 5))
ax.set_xticklabels(['T0', 'T1', 'T2', 'T3', 'T4'])
ax.axhline(undetectable, color='gray', linestyle='--', linewidth=0.8)
ax.set_yscale('log')
ax.set_xlabel('Timepoint') 
ax.set_ylabel('Barcode frequency')

# ===== RIGHT: Replicate Scatter Plot =====
ax = axes[1]

x = this_fitness[f'Batch4_{focal_base}_0.5%EtOH-R1_fitness'].values
y = this_fitness[f'Batch4_{focal_base}_0.5%EtOH-R2_fitness'].values
fitness_vals = this_fitness[f'Batch4_{focal_base}_0.5%EtOH_fitness'].values
xerr = this_fitness[f'Batch4_{focal_base}_0.5%EtOH-R1_stderror'].values
yerr = this_fitness[f'Batch4_{focal_base}_0.5%EtOH-R2_stderror'].values

for i in range(len(x)):
    color = cmap(norm(fitness_vals[i]))
    ax.errorbar(x[i], y[i],
                xerr=xerr[i], yerr=yerr[i],
                fmt='o', color=color, alpha=0.8,
                ecolor=color, capsize=1, elinewidth=0.45, markersize=1.5)

ax.plot([-2, 2], [-2, 2], 'k--', lw=1)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
tick_positions = [-2, -1, 0, 1, 2]
ax.set_xticks(tick_positions)
ax.set_yticks(tick_positions)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))  # No decimals
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))  # No decimals
ax.set_xlabel("Replicate 1 Fitness") 
ax.set_ylabel("Replicate 2 Fitness") 


sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, location='right', shrink=0.85, pad=0.02, aspect=10)
cbar.ax.set_ylabel('Fitness', rotation=270, labelpad=5, va='center', ha='center')

fig.suptitle("Base: 2 Day, Perturbation: 0.5% EtOH")
fig.canvas.draw()   # ensure constrained_layout computes the final positions
fig.savefig('plots/fig2a_revised.pdf')
plt.close()