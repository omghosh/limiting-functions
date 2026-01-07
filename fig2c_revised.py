import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functions import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull

sns.set_style("white")  # Clean style without grid lines

plt.rcParams['font.family'] = 'Helvetica Neue'
plt.rcParams['font.size'] = 8
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

# plt.rcParams['figure.subplot.wspace'] = 0.9
# plt.rcParams['figure.subplot.hspace'] = 0.9



bc_counts, fitness_df, grants_df_with_barcode_df, full_df = create_full_fitness_dataframe()

# # first, mask all values of -10 with extinct_fitness
fitness_df = fitness_df.applymap(replace_extinct)
base_conditions = ['M3', '1.5' ,'30', 'M3b4']

# do pca on fitness_df
X = fitness_df[all_conds].values.T
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
pca = PCA(n_components=2)
pca.fit(X)

# color by base condition 
base_cond_pert = []
colors =[]
base_colors = []
for i,cond in enumerate(all_conds):
    pert = cond.split('_')[2]
    if '2Day' in cond:
        colors.append(env_color_dict['2Day'])
        if pert in base_conditions:
            base_cond_pert.append(i)
            base_colors.append(env_color_dict['2Day'])
            print(f'found {pert} in base conditions')
    elif '1Day' in cond:
        colors.append(env_color_dict['1Day'])
        if pert in base_conditions:
            base_cond_pert.append(i)
            base_colors.append(env_color_dict['1Day'])
    elif 'Salt' in cond:
        colors.append(env_color_dict['Salt'])
        if pert in base_conditions:
            base_cond_pert.append(i)
            base_colors.append(env_color_dict['Salt'])
    else:
        colors.append('black')

print(f'Base condition perturbations: {base_cond_pert}')

# plot pca
fig, ax = plt.subplots(figsize=(3,2), dpi=600)
# add convex hulls around points corresponding to twoday_conds, oneday_conds, and salt_conds 
twoday_points = pca.transform(X)[[all_conds.index(cond) for cond in twoday_conds]]
oneday_points = pca.transform(X)[[all_conds.index(cond) for cond in oneday_conds]]
salt_points = pca.transform(X)[[all_conds.index(cond) for cond in salt_conds]]

twoday_hull = ConvexHull(twoday_points)
oneday_hull = ConvexHull(oneday_points)
salt_hull = ConvexHull(salt_points)

for simplex in twoday_hull.simplices:
    plt.fill(twoday_points[twoday_hull.vertices,0], twoday_points[twoday_hull.vertices,1], color = env_color_dict['2Day'], alpha=0.1)
for simplex in oneday_hull.simplices:
    plt.fill(oneday_points[oneday_hull.vertices,0], oneday_points[oneday_hull.vertices,1], color = env_color_dict['1Day'], alpha=0.05)
for simplex in salt_hull.simplices:
    plt.fill(salt_points[salt_hull.vertices,0], salt_points[salt_hull.vertices,1], color =env_color_dict['Salt'], alpha=0.05)



ax.scatter(pca.transform(X)[:,0], pca.transform(X)[:,1], color=colors, alpha=0.75, marker='o', s= 5)
ax.scatter(pca.transform(X)[np.array(base_cond_pert),0], pca.transform(X)[np.array(base_cond_pert),1], marker='D',s = 5, color=base_colors, alpha = 1)
ax.set_xlabel(f'PC1, {round(pca.explained_variance_ratio_[0]*100,1)}% VE')
ax.set_ylabel(f'PC2, {round(pca.explained_variance_ratio_[1]*100,1)}% VE')
# ax.set_title('PCA Environment Space')


# legend 
ax.scatter([],[], color='gray', marker = 'D', s=5, alpha =0.8,label='Base')
ax.scatter([],[], color='gray', alpha = 0.25,label='Pert.')
#no box around legend
ax.legend(frameon=False, loc='upper left')
plt.tight_layout()
plt.savefig(f'plots/fig2c_revised.pdf')
# plt.show()