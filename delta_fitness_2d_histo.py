import numpy as np 
import pandas as pd
from functions import *
import seaborn as sns
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors


import sys
from scipy import stats

# env_color_dict = {'2Day': (0.77, 0.84, 0.75), '1Day': (0.55, 0.6, 0.98), 'Salt': (1, 0.59, 0.55)}

bc_counts, fitness_df, grants_df_with_barcode_df, full_df = create_full_fitness_dataframe()
organized_perturbation_fitness_df= create_delta_fitness_matrix(batches, fitness_df, environment_dict)
# Load the mako colormap
mako_cmap = sns.color_palette("mako_r", as_cmap=True)

# Modify the colormap to set the first color to white
mako_with_white = mcolors.ListedColormap(['white'] + list(mako_cmap(np.linspace(0, 1, 256))[1:]))

# remove all Li_WT
organized_perturbation_fitness_df.loc[mutant_dict['anc: Li_WT']]
organized_perturbation_fitness_df.drop(mutant_dict['anc: Li_WT'], inplace=True)
all_perts  = list(set([col.split('_')[2] for col in organized_perturbation_fitness_df.columns]))

for focal_perturbation in ['0.5%EtOH']:#['4uMH89']: #all_perts:
    for i,base_1 in enumerate(environment_dict.keys()):
        for e,base_2 in enumerate(environment_dict.keys()):
            if base_1 == base_2:
                continue
            if e<i:
                continue

            # get batch 
            # find perturbation in column names
            if len([col for col in organized_perturbation_fitness_df.columns if base_1 in col and focal_perturbation in col]) == 0:
                print(f' {focal_perturbation} not in {base_1}')
                continue
            if len([col for col in organized_perturbation_fitness_df.columns if base_2 in col and focal_perturbation in col]) == 0:
                print(f' {focal_perturbation} not in {base_2}')
                continue
            perturbation = [col for col in organized_perturbation_fitness_df.columns if base_1 in col and focal_perturbation in col][0]
            batch = perturbation.split('_')[0]

    #         plt.errorbar(x = organized_perturbation_fitness_df[f'{batch}_{base_1}_{focal_perturbation}_fitness'], y=organized_perturbation_fitness_df[f'{batch}_{base_2}_{focal_perturbation}_fitness'],
    #                 xerr = organized_perturbation_fitness_df[f'{batch}_{base_1}_{focal_perturbation}_stderror'], yerr=organized_perturbation_fitness_df[f'{batch}_{base_2}_{focal_perturbation}_stderror']
    #                 , fmt='o', color=env_color_dict[base_1], alpha = 0.1)
    #         # plot contour lines
    #         sns.kdeplot(data=organized_perturbation_fitness_df, x=f'{batch}_{base_1}_{focal_perturbation}_fitness', y=f'{batch}_{base_2}_{focal_perturbation}_fitness', color=env_color_dict[base_1], fill=False,
    # )
    

            plt.hexbin(x = organized_perturbation_fitness_df[f'{batch}_{base_1}_{focal_perturbation}_fitness'], y=organized_perturbation_fitness_df[f'{batch}_{base_2}_{focal_perturbation}_fitness'], gridsize=50, cmap=mako_with_white)
            plt.colorbar()


            # plot vertical and horizontal lines at 0
            plt.axhline(y=0, color='gray', linestyle='--', alpha = 0.5)
            plt.axvline(x=0, color='gray', linestyle='--', alpha = 0.5)

            # what fraction of data is in each quadrant? 
            quadrant1 = len(organized_perturbation_fitness_df[(organized_perturbation_fitness_df[f'{batch}_{base_1}_{focal_perturbation}_fitness'] > 0) & (organized_perturbation_fitness_df[f'{batch}_{base_2}_{focal_perturbation}_fitness'] > 0)])/len(organized_perturbation_fitness_df)
            quadrant2 = len(organized_perturbation_fitness_df[(organized_perturbation_fitness_df[f'{batch}_{base_1}_{focal_perturbation}_fitness'] < 0) & (organized_perturbation_fitness_df[f'{batch}_{base_2}_{focal_perturbation}_fitness'] > 0)])/len(organized_perturbation_fitness_df)
            quadrant3 = len(organized_perturbation_fitness_df[(organized_perturbation_fitness_df[f'{batch}_{base_1}_{focal_perturbation}_fitness'] < 0) & (organized_perturbation_fitness_df[f'{batch}_{base_2}_{focal_perturbation}_fitness'] < 0)])/len(organized_perturbation_fitness_df)
            quadrant4 = len(organized_perturbation_fitness_df[(organized_perturbation_fitness_df[f'{batch}_{base_1}_{focal_perturbation}_fitness'] > 0) & (organized_perturbation_fitness_df[f'{batch}_{base_2}_{focal_perturbation}_fitness'] < 0)])/len(organized_perturbation_fitness_df)

            # get the axes limits
            xmin, xmax = plt.xlim()
            ymin, ymax = plt.ylim()


            # plt.text(xmax-0.5, ymax-0.55, f'{(quadrant1*100):.2f}%', fontsize=10, color = 'gray')
            # plt.text(xmin + 0.25, ymax-0.55, f'{(quadrant2*100):.2f}%', fontsize=10, color = 'gray')
            # plt.text(xmin+0.25, ymin+0.25, f'{(quadrant3*100):.2f}%', fontsize=10, color = 'gray')
            # plt.text(xmax-0.5, ymin+0.25, f'{(quadrant4*100):.2f}%', fontsize=10, color = 'gray')

            plt.xlabel(f'{base_1} delta X')
            plt.ylabel(f'{base_2} delta X')
            # add regression line 
            sns.regplot(data=organized_perturbation_fitness_df, x=f'{batch}_{base_1}_{focal_perturbation}_fitness', y=f'{batch}_{base_2}_{focal_perturbation}_fitness', scatter=False, color='gray')
            # add p value on slope of regression line
            slope, intercept, r_value, p_value, std_err = stats.linregress(organized_perturbation_fitness_df[f'{batch}_{base_1}_{focal_perturbation}_fitness'], organized_perturbation_fitness_df[f'{batch}_{base_2}_{focal_perturbation}_fitness'])

            # add a 1-1 line 
            plt.plot([xmin, xmax], [xmin, xmax], color='gray', linestyle='--', alpha = 0.5)
            plt.text(xmin + 0.25, ymax-0.75, f'p = {p_value:.2e}', fontsize=10, color = 'gray')
            plt.title(f'Delta fitness due to {focal_perturbation} on {base_1} vs {base_2}')
            # plt.show()
            plt.savefig(f'../plots/{focal_perturbation}_{base_1}_{base_2}_delta_fitness_pval.png')
            plt.close()


#             import numpy as np
# from scipy import stats
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.set(style="white")

# mean = np.zeros(3)
# cov = np.random.uniform(.2, .4, (3, 3))
# cov += cov.T
# cov[np.diag_indices(3)] = 1
# data = np.random.multivariate_normal(mean, cov, 100)
# df = pd.DataFrame(data, columns=["X", "Y", "Z"])

# def corrfunc(x, y, **kws):
#     r, _ = stats.pearsonr(x, y)
#     ax = plt.gca()
#     ax.annotate("r = {:.2f}".format(r),
#                 xy=(.1, .9), xycoords=ax.transAxes)

# g = sns.PairGrid(df, palette=["red"])
# g.map_upper(plt.scatter, s=10)
# g.map_diag(sns.distplot, kde=False)
# g.map_lower(sns.kdeplot, cmap="Blues_d")
# g.map_lower(corrfunc)