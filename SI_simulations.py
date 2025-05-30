import numpy as np 
from matplotlib import pyplot as plt 
import seaborn as sns

total_num_mutants = 5000
num_adaptive_mutants = 1000
num_bases = 2 

k = 10

## Create underlying matrices
np.random.seed(0)
# m = 500
# k = 10
# n= 10000

m_to_k = np.random.randn(total_num_mutants, k)*0.1
E = 40 # number of environments
num_envs_per_hub = E//num_bases
base1_envs = np.arange(E//num_bases)
base2_envs = np.arange(E//num_bases, 2*E//num_bases)
num_phenotypes_to_scan = num_envs_per_hub-1
colors = sns.color_palette("mako_r", num_envs_per_hub)
colors_base2 = sns.color_palette("rocket_r", num_envs_per_hub)


##### Build phi-to-e matrix
    
k_to_e_full = np.random.rand(k,E)

k_to_e_none = np.random.rand(k,E)
k_to_e_none[:k//2, E//2:] = 0
k_to_e_none[k//2:,:E//2 ] = 0
k_to_e_partial = np.random.rand(k,E)
k_to_e_partial[1:3, E//2:] = 0
k_to_e_partial[3:5,:E//2 ] = 0

# Full overlap 
X_al = m_to_k @ k_to_e_full  + np.random.randn(total_num_mutants, E)*0.05
# select the top m mutants in the first environment (evolution)
top_m = np.argsort(X_al[:, 0])[-num_adaptive_mutants:]
X_full =X_al[top_m, :]
# center the data (delta fitness)
# X_full = X_full - X_full.mean(axis=1, keepdims=True)
X1_full = X_full[:, :E//num_bases]
X2_full = X_full[:, E//num_bases:2*E//num_bases]

# Partial overlap 
X_al = m_to_k @ k_to_e_partial  + np.random.randn(total_num_mutants, E)*0.05
# select the top m mutants in the first environment (evolution)
top_m = np.argsort(X_al[:, 0])[-num_adaptive_mutants:]
X_partial =X_al[top_m, :]
# center the data (delta fitness)
# X_partial = X_partial - X_partial.mean(axis=1, keepdims=True)

X1_partial = X_partial[:, :E//num_bases]
X2_partial = X_partial[:, E//num_bases:2*E//num_bases]


# None overlap 
X_al = m_to_k @ k_to_e_none  + np.random.randn(total_num_mutants, E)*0.05
# select the top m mutants in the first environment (evolution)
top_m = np.argsort(X_al[:, 0])[-num_adaptive_mutants:]
X_none =X_al[top_m, :]
# center the data (delta fitness)
# X_none = X_none - X_none.mean(axis = 1, keepdims=True)
X1_none = X_none[:, :E//num_bases]
X2_none = X_none[:, E//num_bases:2*E//num_bases]


def lin_regress_func(Base1, Base2):
    results_dict_full_base = {}
    for j in range(num_phenotypes_to_scan):
        predicted_X1_full_base = np.zeros_like(X1)
        for held_out_index in range(num_envs_per_hub):
            held_out_env = X1[:, held_out_index]
            X1_reduced = np.delete(X1, held_out_index, axis=1)
            U1_reduced, s1_reduced, V1_reduced = np.linalg.svd(X1_reduced, full_matrices=False)
            design_matrix_1 = U1_reduced[:, j].reshape(-1, 1)
            coefficients_f_r, residuals_f_r, rank_f_r, s_vals_f_r = np.linalg.lstsq(design_matrix_1, held_out_env, rcond=None)
            predicted_held_out_env = design_matrix_1 @ coefficients_f_r
            predicted_X1_full_base[:, held_out_index] = predicted_held_out_env
        sum_squared_error = np.sum((X1 - predicted_X1_full_base)**2)
        total_sum_squares = np.sum(X1**2)
        r_squared = 1 - sum_squared_error/total_sum_squares
        results_dict_full_base[j] = r_squared

    var_exp_by_component_full_base = []
    for item in results_dict_full_base:
        var_exp_by_component_full_base.append(results_dict_full_base[item])

    base2_var_explained_by_component = []
    # now let's predict base 2 from base 1
    base2_total_sum_of_squares  = np.sum((X2)**2 )
    for r in range(num_envs_per_hub):
        design_matrix_one = u1[:, r].reshape(-1, 1)
        coefficients1_one_target, residuals1_one_target, rank1_one_target, s_vals1_one_target = np.linalg.lstsq(design_matrix_one, X2, rcond=None)
        X2_pred_one = design_matrix_one @ coefficients1_one_target
        base2_var_explained_by_component.append(1 - np.sum(residuals1_one_target)/base2_total_sum_of_squares)
    return var_exp_by_component_full_base, base2_var_explained_by_component






fig, axs = plt.subplots(4,3, figsize=(14,14))
sns.heatmap(k_to_e_full, ax = axs[0,0], cbar=None, cmap = 'BrBG',  cbar_kws={'label': 'Weight'})
axs[0,0].set_xticks([num_envs_per_hub//2, num_envs_per_hub*3//2], ['Base 1', 'Base 2'],rotation = 0)
axs[0,0].set_ylabel('Fitnotypes')
axs[0,0].set_yticks(np.arange(0.5,k+0.5), np.arange(1,k+1))
sns.heatmap(k_to_e_partial, ax = axs[0,1], cbar=None, cmap = 'BrBG',  cbar_kws={'label': 'Weight'})
axs[0,1].set_xticks([num_envs_per_hub//2, num_envs_per_hub*3//2], ['Base 1', 'Base 2'],rotation = 0)
axs[0,1].set_yticks(np.arange(0.5,k+0.5), np.arange(1,k+1))

sns.heatmap(k_to_e_none, ax = axs[0,2], cmap = 'BrBG',  cbar_kws={'label': 'Weight'})
axs[0,2].set_xticks([num_envs_per_hub//2, num_envs_per_hub*3//2], ['Base 1', 'Base 2'], rotation = 0)
axs[0,2].set_yticks(np.arange(0.5,k+0.5), np.arange(1,k+1))


sns.heatmap(X_full, ax = axs[1,0], cbar=None, cmap = 'mako', cbar_kws={'label': 'Fitness'})
axs[1,0].set_ylabel('Mutants')
axs[1,0].set_xticks([num_envs_per_hub//2, num_envs_per_hub*3//2], ['Base 1', 'Base 2'],rotation = 0)
axs[1,0].set_yticks([])

sns.heatmap(X_partial, ax = axs[1,1], cbar=None, cmap = 'mako', cbar_kws={'label': 'Fitness'})
axs[1,1].set_xticks([num_envs_per_hub//2, num_envs_per_hub*3//2], ['Base 1', 'Base 2'],rotation = 0)
axs[1,1].set_yticks([])
axs[1,1].set_ylabel('Mutants')

sns.heatmap(X_none, ax = axs[1,2], cmap = 'mako', cbar_kws={'label': 'Fitness'})
axs[1,2].set_xticks([num_envs_per_hub//2, num_envs_per_hub*3//2], ['Base 1', 'Base 2'],rotation = 0)
axs[1,2].set_yticks([])
axs[1,2].set_ylabel('Mutants')



for i,(X1, X2) in enumerate([(X1_full, X2_full), (X1_partial, X2_partial), (X1_none, X2_none)]):
    u1,s1,v1 = np.linalg.svd(X1, full_matrices=False)
    u2,s2,v2 = np.linalg.svd(X2, full_matrices=False)

    axs[2,i].semilogy(s1**2/np.sum(s1**2),'-o',alpha = 0.75, label='Base 1', color = colors[5])
    axs[2,i].semilogy(s2**2/np.sum(s2**2), '-o', alpha = 0.75, label='Base 2', color = colors_base2[5])
    axs[2,i].legend()
    axs[2,i].set_ylabel('Fraction Variance Explained')
    axs[2,i].set_xlabel('Fitnotype')
    axs[2,i].set_xticks(np.arange(X1.shape[1]), np.arange(1,X1.shape[1]+1))


    var_exp_by_component_full_base, base2_var_explained_by_component = lin_regress_func(X1, X2)
    # Define x-positions for the two bar charts
    x_positions = [0.5, 1.5]  # Spaced out to leave room for connections
    bar_width = 0.5  # Width of the bars

    # Calculate the x-coordinates of the left and right edges of each bar
    x_left_edges = [x - bar_width/2 for x in x_positions]
    x_right_edges = [x + bar_width/2 for x in x_positions]

    # Store bottoms and tops for each bar chart
    bottoms = [[], []]
    tops = [[], []]

    # Plot the stacked bars
    for idx, (diffs, x_pos) in enumerate(zip([var_exp_by_component_full_base, base2_var_explained_by_component], x_positions)):
        bottom = 0
        for q in range(num_phenotypes_to_scan):
            axs[3,i].bar(x_pos, diffs[q], bottom=bottom,  color= colors[q], width=bar_width)
            # Store bottom and top for connecting curves
            bottoms[idx].append(bottom)
            bottom += diffs[q]
            tops[idx].append(bottom)

        # Connect the corresponding segments between bar charts with filled areas
    for w in range(num_phenotypes_to_scan):
        x_vals = np.linspace(x_right_edges[0], x_left_edges[1], 100)
        bottom_curve = np.linspace(bottoms[0][w], bottoms[1][w], 100)
        top_curve = np.linspace(tops[0][w], tops[1][w], 100)
        axs[1,i].fill_between(x_vals, bottom_curve, top_curve, color = colors[w], alpha=0.3)



    var_exp_by_component_full_base, base2_var_explained_by_component = lin_regress_func(X2, X1)
    # Define x-positions for the two bar charts
    x_positions = [2.5, 3.5]  # Spaced out to leave room for connections
    bar_width = 0.5  # Width of the bars

    # Calculate the x-coordinates of the left and right edges of each bar
    x_left_edges = [x - bar_width/2 for x in x_positions]
    x_right_edges = [x + bar_width/2 for x in x_positions]

    # Store bottoms and tops for each bar chart
    bottoms = [[], []]
    tops = [[], []]

    # Plot the stacked bars
    for idx, (diffs, x_pos) in enumerate(zip([var_exp_by_component_full_base, base2_var_explained_by_component], x_positions)):
        bottom = 0
        for q in range(num_phenotypes_to_scan):
            axs[3,i].bar(x_pos, diffs[q], bottom=bottom,  color= colors_base2[q], width=bar_width)
            # Store bottom and top for connecting curves
            bottoms[idx].append(bottom)
            bottom += diffs[q]
            tops[idx].append(bottom)

        # Connect the corresponding segments between bar charts with filled areas
    for w in range(num_phenotypes_to_scan):
        x_vals = np.linspace(x_right_edges[0], x_left_edges[1], 100)
        bottom_curve = np.linspace(bottoms[0][w], bottoms[1][w], 100)
        top_curve = np.linspace(tops[0][w], tops[1][w], 100)
        axs[1,i].fill_between(x_vals, bottom_curve, top_curve, color = colors_base2[w], alpha=0.3)



    axs[3,i].set_ylabel('Variance Explained')
    axs[3,i].set_xticks([0.5, 1.5, 2.5, 3.5], ['Training \n Base 1', 'Testing \nBase 2', 'Training\n Base 2', 'Testing \nBase 1'])
    # axs[3,i].set_title('Predicting left out envs from Base 1')
    axs[3,i].axhline(1,linestyle =':', color='black', linewidth=0.5)
    axs[3,i].set_ylim([-0.01, 1.1])

plt.tight_layout()
plt.savefig('plots/SI/sims_analyses.png', dpi=300)
