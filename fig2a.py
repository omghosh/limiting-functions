import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from functions import *
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable


plt.rcParams['font.family'] = 'Geneva'
plt.rcParams['font.size'] = 20

bc_counts, fitness_df, grants_df_with_barcode_df, full_df = create_full_fitness_dataframe()
organized_perturbation_fitness_df= create_delta_fitness_matrix(batches, fitness_df, environment_dict)

# remove all Li_WT
organized_perturbation_fitness_df.loc[mutant_dict['anc: Li_WT']]
organized_perturbation_fitness_df.drop(mutant_dict['anc: Li_WT'], inplace=True)



fitness_df = fitness_df.applymap(replace_extinct)

# Sort rows by fitness
this_fitness_df = fitness_df.sort_values('Batch4_Salt_0.5%EtOH_fitness')

# Get the fitness values
fitness_values = this_fitness_df['Batch4_Salt_0.5%EtOH_fitness'].values

# Define colormap and normalization
cmap = sns.color_palette("mako", as_cmap=True)  # Use a diverging colormap
# norm = mcolors.TwoSlopeNorm(vmin=np.min(fitness_values), vcenter=0, vmax=np.max(fitness_values))
# absmax = np.max(np.abs(fitness_values))
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
from matplotlib.collections import LineCollection

# Prepare data
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

# Create LineCollection
lc = LineCollection(lines, colors=colors, linewidths=0.5, alpha=0.3)
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
ax.add_collection(lc)

ax.set_yscale('log')
ax.set_xlim(0, 4)
ax.set_ylim(4e-8, 1)

ax.set_xticks(range(0, 5))
ax.set_xticklabels(['T0', 'T1', 'T2', 'T3', 'T4'])
ax.axhline(undetectable, color='gray', linestyle='--')

yticks = ax.get_yticks()
yticklabels = [f'$10^{{{int(np.log10(y))}}}$' for y in yticks]
ax.set_yticklabels(yticklabels)
ax.set_ylabel('Barcode frequency')

ax.set_title(f"Base: 2 Day, Perturbation: 0.5% EtOH, Rep 1")

plt.savefig('plots/fig2a_trajectories.png')
plt.close()


focal_base = '2Day'
this_fitness = fitness_df[[f'Batch4_{focal_base}_0.5%EtOH-R1_fitness', 
                         f'Batch4_{focal_base}_0.5%EtOH-R2_fitness', 
                         f'Batch4_{focal_base}_0.5%EtOH_fitness',
                         f'Batch4_{focal_base}_0.5%EtOH-R1_stderror', 
                         f'Batch4_{focal_base}_0.5%EtOH-R2_stderror']]
this_fitness = this_fitness.sort_values(f'Batch4_{focal_base}_0.5%EtOH_fitness')



fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

# x and y values
x = this_fitness[f'Batch4_{focal_base}_0.5%EtOH-R1_fitness'].values
y = this_fitness[f'Batch4_{focal_base}_0.5%EtOH-R2_fitness'].values

# fitness values
fitness_vals = this_fitness[f'Batch4_{focal_base}_0.5%EtOH_fitness'].values

# error bars
xerr = this_fitness[f'Batch4_{focal_base}_0.5%EtOH-R1_stderror'].values
yerr = this_fitness[f'Batch4_{focal_base}_0.5%EtOH-R2_stderror'].values

# Scatter points individually
for i in range(len(x)):
    color = cmap(norm(fitness_vals[i]))  # map fitness to color
    ax.errorbar(x[i], y[i],
                xerr=xerr[i], yerr=yerr[i],
                fmt='o', color=color, alpha=0.6,
                ecolor=color, capsize=2, elinewidth=0.8, markersize=4)

# Identity line
ax.plot([-2, 2], [-2, 2], 'k--', lw=1)

# Axis settings
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xlabel("Replicate 1 Fitness")
ax.set_ylabel("Replicate 2 Fitness")
ax.set_title(f"Base: 2 Day, Perturbation: 0.5% EtOH")

# Fancy colorbar setup
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

# Scatter mappable for colorbar
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])  # (required to make ScalarMappable work correctly)

cbar = fig.colorbar(sm, cax=cax)
# cbar.set_label("Relative Fitness", rotation = 270)
# cbar.ax.tick_params(labelsize=12)


plt.tight_layout()
plt.savefig('plots/fig2a_scatter.png')
plt.close()
