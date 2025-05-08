# let's make the bar plots 
import numpy as np
import matplotlib.pyplot as plt
from functions import *
from scipy.linalg import lstsq
from sklearn.metrics import r2_score
import seaborn as sns
from scipy.stats import spearmanr
from matplotlib.lines import Line2D


bc_counts, fitness_df, grants_df_with_barcode_df, full_df = create_full_fitness_dataframe()

organized_perturbation_fitness_df = create_delta_fitness_matrix(batches, fitness_df, environment_dict)

environment_dict['Salt'] = [env for env in environment_dict['Salt'] if env != 'Batch3_Salt_NS_fitness']


plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 12


def variance_explained_in_and_out(focal_base, target_base, k):
    results = {}
    focal_base_var_explained_by_component =[]
    target_base_var_explained_by_component = []

    # Build up predicted fitness matrix env by env:
    predicted_focal_base = np.zeros_like(focal_base)

    for j in range(focal_base.shape[1]):
        held_out_env = focal_base[:, j]
        reduced_focal_base = np.delete(focal_base, j, axis=1)
        U_f_r, s_f_r, Vt_f_r = np.linalg.svd(reduced_focal_base, full_matrices=False)
        U_f_r_k = U_f_r[:, :k]
        design_matrix_f_r = U_f_r_k  # Shape: (m, k)

        # Perform least squares regression to find coefficients
        coefficients_f_r, residuals_f_r, rank_f_r, s_vals_f_r = np.linalg.lstsq(design_matrix_f_r, held_out_env, rcond=None)
        predicted_held_out_env = design_matrix_f_r @ coefficients_f_r
        predicted_focal_base[:, j] = predicted_held_out_env

    # Calculate error
    square_sum_errors = np.sum((focal_base - predicted_focal_base)**2)
    total_sum_squares = np.sum((focal_base)**2)
    for i in range(k):
        # Build up predicted fitness matrix env by env:
        predicted_focal_base = np.zeros_like(focal_base)

        for j in range(focal_base.shape[1]):
            held_out_env = focal_base[:, j]
            reduced_focal_base = np.delete(focal_base, j, axis=1)
            U_f_r, s_f_r, Vt_f_r = np.linalg.svd(reduced_focal_base, full_matrices=False)
            U_f_r_k = U_f_r[:, :k]
            design_matrix_one = U_f_r_k[:, i].reshape(-1, 1)
            # Perform least squares regression to find coefficients
            coefficients_f_r, residuals_f_r, rank_f_r, s_vals_f_r = np.linalg.lstsq(design_matrix_one, held_out_env, rcond=None)
            # Make predictions
            predicted_held_out_env = design_matrix_one @ coefficients_f_r
            predicted_focal_base[:, j] = predicted_held_out_env

        # Calculate error
        square_sum_errors = np.sum((focal_base - predicted_focal_base)**2)

        total_sum_squares = np.sum((focal_base)**2)

        focal_base_var_explained_by_component.append(1-square_sum_errors/total_sum_squares)

        
    U1, s1, Vt1 = np.linalg.svd(focal_base, full_matrices=False)

    U_k1 = U1[:, :k]  # First k left singular vectors
    s_k1 = s1[:k]
    U2, s2, Vt2 = np.linalg.svd(target_base, full_matrices=False)   
    s_k2 = s2[:k]


    # Calculate noise in target base at k components
    s2_noise = np.sum(s2[k:]**2) / np.sum(s2**2)
    s1_noise = np.sum(s1[k:]**2) / np.sum(s1**2)

    # compare to within focal base (not LOO version)
    for i in range(k):
        design_matrix_one = U_k1[:, i].reshape(-1, 1)
        coefficients1_one, residuals1_one, rank1_one, s_vals1_one = np.linalg.lstsq(design_matrix_one, focal_base, rcond=None)
        X_pred1_one = design_matrix_one @ coefficients1_one
    
    target_tss = np.sum((target_base)**2 )
    for i in range(k):
        design_matrix_one = U_k1[:, i].reshape(-1, 1)
        coefficients1_one_target, residuals1_one_target, rank1_one_target, s_vals1_one_target = np.linalg.lstsq(design_matrix_one, target_base, rcond=None)
        X_pred2_one = design_matrix_one @ coefficients1_one_target
        target_base_var_explained_by_component.append(1 - np.sum(residuals1_one_target)/target_tss)
    results['focal_base_var_explained_by_component'] = np.array(focal_base_var_explained_by_component)
    results['target_base_var_explained_by_component'] = np.array(target_base_var_explained_by_component)
    results['target_noise_at_k'] = s2_noise
    results['focal_noise_at_k'] = s1_noise
    return results


k = 7

mutant_names = ['all', 'original', 'anc: IRA1_NON', 'anc: IRA1_MIS', 'anc: GPB2', 'anc: TOR1', 'anc: CYR1']

for mut in mutant_names[:1]:
    if 'anc' in mut:
        which_mutants = mutant_dict[mut]
    elif mut == 'original':
        which_mutants =  mutant_dict['Original Training'] + mutant_dict['Original Testing']
    elif mut == 'all':
        which_mutants = fitness_df.index

    for focal_base, target_base1, target_base2 in [('Salt', '1Day', '2Day'), ('1Day', '2Day', 'Salt'), ('2Day', 'Salt', '1Day')]:

        X1 = organized_perturbation_fitness_df.loc[which_mutants, environment_dict[focal_base]].values
        X2 = organized_perturbation_fitness_df.loc[which_mutants, environment_dict[target_base1]].values
        X3 = organized_perturbation_fitness_df.loc[which_mutants, environment_dict[target_base2]].values

        variance_explained_in_and_out(X1, X2, k)
        results_f_t1 = variance_explained_in_and_out(X1, X2, k)

        results_t1_f = variance_explained_in_and_out(X2, X1, k)
        t1_explainable_as_focal = 1-np.sum(results_t1_f['focal_base_var_explained_by_component'])

        results_f_t2 = variance_explained_in_and_out(X1, X3, k)

        results_t2_f = variance_explained_in_and_out(X3, X1, k)
        t2_explainable_as_focal = 1-np.sum(results_t2_f['focal_base_var_explained_by_component'])

        f_var_explained = results_f_t1["focal_base_var_explained_by_component"]
        t1_var_explained = results_f_t1["target_base_var_explained_by_component"]
        t2_var_explained = results_f_t2["target_base_var_explained_by_component"]


        t1_noise = t1_explainable_as_focal
        t2_noise=t2_explainable_as_focal
        # f_noise = results_f_t1["focal_noise_at_k"]
        f_noise = 1 - np.sum(f_var_explained)
        unexplained_f = 0#1 - np.sum(f_var_explained) - f_noise

        f_var_explained = np.append(f_var_explained, unexplained_f)
        f_var_explained = np.append(f_var_explained, f_noise)

        unexplained_t1 = 1 - np.sum(t1_var_explained) - t1_noise
        t1_var_explained = np.append(t1_var_explained, unexplained_t1)
        t1_var_explained = np.append(t1_var_explained, t1_noise)


        unexplained_t2 = 1 - np.sum(t2_var_explained) - t2_noise
        t2_var_explained = np.append(t2_var_explained, unexplained_t2)
        t2_var_explained = np.append(t2_var_explained, t2_noise)


        #### plot things #####
     

        colors = sns.color_palette('pastel', k) + ['crimson']+['lightgray']

        if focal_base == '2Day':
            # Create figure with proper size
            plt.figure(figsize=(8,6) ,dpi=300)
                    # Define x-positions for the three bar charts
            x_positions = [1, 3, 5]  # Spaced out to leave room for connections
            bar_width = 1  # Width of the bars
        else: 
            plt.figure(figsize=(8, 6) ,dpi=150)
                    # Define x-positions for the three bar charts
            x_positions = [1, 3, 5]  # Spaced out to leave room for connections
            bar_width = 0.8  # Width of the bars

        # Calculate the x-coordinates of the left and right edges of each bar
        x_left_edges = [x - bar_width/2 for x in x_positions]
        x_right_edges = [x + bar_width/2 for x in x_positions]

        # Store bottoms and tops for each bar chart
        bottoms = [[], [], []]
        tops = [[], [], []]

        # Plot the stacked bars
        for idx, (diffs, x_pos) in enumerate(zip([t1_var_explained, f_var_explained, t2_var_explained], x_positions)):
            bottom = 0
            for i in range(k+2):
                plt.bar(x_pos, diffs[i], bottom=bottom, color=colors[i], width=bar_width)
                # Store bottom and top for connecting curves
                bottoms[idx].append(bottom)
                bottom += diffs[i]
                tops[idx].append(bottom)

            # Connect the corresponding segments between bar charts with filled areas
        for i in range(k+2):
            x_vals = np.linspace(x_right_edges[0], x_left_edges[1], 100)
            bottom_curve = np.linspace(bottoms[0][i], bottoms[1][i], 100)
            top_curve = np.linspace(tops[0][i], tops[1][i], 100)
            plt.fill_between(x_vals, bottom_curve, top_curve, color=colors[i], alpha=0.3)
            
            # Connect second and third bar chart - using the edges of the bars
            x_vals = np.linspace(x_right_edges[1], x_left_edges[2], 100)
            bottom_curve = np.linspace(bottoms[1][i], bottoms[2][i], 100)
            top_curve = np.linspace(tops[1][i], tops[2][i], 100)
            
            plt.fill_between(x_vals, bottom_curve, top_curve, color=colors[i], alpha=0.3)
  
        # Add labels and customize
        plt.xticks([]) #, [f'Target 1: {target_base1}', f'Focal: {focal_base}',f'Target 2: {target_base2}'], fontsize=12)
        plt.ylim(0, 1)

        plt.yticks([0,.2,.4,.6, .8, 1],[0,.2,.4,.6, .8, 1], fontsize = 14)
        plt.ylabel('Fraction of Variance Explained', fontsize=16)

        # Remove upper and right spines
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)


        # Add titles and adjust layout
        plt.tight_layout()
        if focal_base == '2Day':
            plt.savefig(f'plots/fig5_b.png')
        elif focal_base == 'Salt':
            plt.savefig(f'plots/fig5_c_salt.png')
        elif focal_base == '1Day':
            plt.savefig(f'plots/fig5_c_1day.png')
        # plt.show()
        plt.close()


# Assuming you have 3 components
colors = sns.color_palette('pastel', k) + ['crimson']+['lightgray']

labels = [f'k = {i+1}' for i in range(k)] + ['Missing Variance', 'Noise and non-linearities']

# Create a new figure for the standalone legend
legend_fig = plt.figure(figsize=(3, 2))

# Create line objects for the legend
legend_lines = [Line2D([0], [0], color=colors[i], lw=4) for i in range(len(colors))]

# Add a legend to the legend figure
legend = plt.legend(legend_lines, labels, frameon=False ,ncol=3, loc='center', fontsize=12)

# Remove axes for clarity
plt.axis('off')

# Save the legend as a separate file
plt.savefig('plots/fig5_legend.png', bbox_inches='tight', dpi=300)


