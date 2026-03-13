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


plt.rcParams['font.family'] = 'Helvetica Neue'
plt.rcParams['font.size'] = 28



two_day_color_scheme=[ (0.15, 0.25, 0.21) ,
(0.24, 0.39, 0.34) ,
(0.55, 0.66, 0.56),
(0.65, 0.77, 0.66),
(0.75, 0.87, 0.76) ,
(0.85, 0.93, 0.86),'gray', 'lightgray' ]

one_day_color_scheme=[(0.12, 0.09, 0.52),
(0.24, 0.19, 0.71),
 (0.49, 0.38, 1),
(0.60, 0.46, 1) ,
(0.70, 0.54, 1) ,
 (0.80, 0.62, 1) ,'gray', 'lightgray']


salt_color_scheme =[

 (1, 0.39, 0.36),
(1, 0.64, 0.60),
(1, 0.76, 0.71),
(1, 0.88, 0.82),
(1, 0.94, 0.91), (0.63, 0.31, 0.28), 'gray', 'lightgray']


def variance_explained_in_and_out(focal_base, target_base, k):
    results = {}
    m, n = focal_base.shape
    # m mutants, n environments 
    TSS_focal = np.sum(focal_base**2)
    expl_ss_comp_LOO = np.zeros(k)

    # for each environment in focal 
    for j in range(n):
        # yj is the held out perturbation
        yj = focal_base[:, j]
        # Y_minus_j is the reduced focal base matrix 
        Y_minus_j = np.delete(focal_base, j, axis=1)

        if Y_minus_j.size == 0:
            continue

        # Do svd on reduced matrix 
        U, s, Vt = np.linalg.svd(Y_minus_j, full_matrices=False)
        k_eff = min(k, U.shape[1])
        if k_eff == 0:
            continue
        U_k = U[:, :k_eff]
        c = U_k.T @ yj  # coefficients in LOO basis
        expl_ss_comp_LOO[:k_eff] += c**2

    if TSS_focal > 0:
        focal_LOO_fraction_per_component = expl_ss_comp_LOO / TSS_focal
        focal_LOO_R2_rank_k = np.sum(expl_ss_comp_LOO) / TSS_focal
    else:
        focal_LOO_fraction_per_component = np.zeros(k)
        focal_LOO_R2_rank_k = 0.0
        
        
    
    # Full SVD of focal for cross-base refit basis and tail energy
    if focal_base.size > 0:
        U1, s1, Vt1 = np.linalg.svd(focal_base, full_matrices=False)
        denom1 = np.sum(s1**2)
        s1_noise = (np.sum(s1[k:]**2) / denom1) if denom1 > 0 else 0.0
    else:
        U1 = np.empty((focal_base.shape[0], 0))
        s1 = np.array([])
        s1_noise = 0.0

    k_eff_full = min(k, U1.shape[1])
    U_k1 = U1[:, :k_eff_full]

    # Target: refit using focal U (vectorized)
    TSS_target = np.sum(target_base**2)
    if TSS_target > 0 and k_eff_full > 0:
        C_target = U_k1.T @ target_base  # shape (k_eff_full, n_target_cols)
        expl_ss_comp_target = np.sum(C_target**2, axis=1)  # per-component explained SS
        target_refit_fraction_per_component = np.zeros(k)
        target_refit_fraction_per_component[:k_eff_full] = expl_ss_comp_target / TSS_target
        target_refit_R2_rank_k = np.sum(expl_ss_comp_target) / TSS_target
    else:
        target_refit_fraction_per_component = np.zeros(k)
        target_refit_R2_rank_k = 0.0

    # # Target tail energy at k
    # if target_base.size > 0:
    #     U2, s2, Vt2 = np.linalg.svd(target_base, full_matrices=False)
    #     denom2 = np.sum(s2**2)
    #     s2_noise = (np.sum(s2[k:]**2) / denom2) if denom2 > 0 else 0.0
    # else:
    #     s2_noise = 0.0

    results['focal_LOO_fraction_per_component'] = focal_LOO_fraction_per_component
    results['focal_LOO_R2_rank_k'] = focal_LOO_R2_rank_k
    results['target_refit_fraction_per_component'] = target_refit_fraction_per_component
    results['target_refit_R2_rank_k'] = target_refit_R2_rank_k

    return results
    



k = 6

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
        t1_explainable_as_focal = 1-np.sum(results_t1_f['focal_LOO_fraction_per_component'])

        results_f_t2 = variance_explained_in_and_out(X1, X3, k)

        results_t2_f = variance_explained_in_and_out(X3, X1, k)
        t2_explainable_as_focal = 1-np.sum(results_t2_f['focal_LOO_fraction_per_component'])

        f_var_explained = results_f_t1["focal_LOO_fraction_per_component"]
        t1_var_explained = results_f_t1["target_refit_fraction_per_component"]
        t2_var_explained = results_f_t2["target_refit_fraction_per_component"]
        print(focal_base)
        print(np.sum(f_var_explained))
        print(target_base1, np.sum(t1_var_explained))
        print(target_base2, np.sum(t2_var_explained))

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
     

        # colors = sns.color_palette('pastel', k) + ['crimson']+['lightgray']

        if focal_base == '2Day':
            # Create figure with proper size
            plt.figure(figsize=(6,8))
                    # Define x-positions for the three bar charts
            x_positions = [1, 3, 5]  # Spaced out to leave room for connections
            bar_width = 1 # Width of the bars
            colors = two_day_color_scheme
        elif focal_base == '1Day': 
            colors = one_day_color_scheme
            plt.figure(figsize=(6,8) )
                    # Define x-positions for the three bar charts
            x_positions = [1, 3, 5]  # Spaced out to leave room for connections
            bar_width = 1 # Width of the bars
        elif focal_base == 'Salt':
            colors = salt_color_scheme
            plt.figure(figsize=(6,8))
                    # Define x-positions for the three bar charts
            x_positions = [1, 3, 5]  # Spaced out to leave room for connections
            bar_width = 1  # Width of the bars

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
                if i == k:
                    if idx == 0:
                        plt.bar(x_pos, diffs[i], bottom=bottom, color=env_color_dict[target_base1], hatch = '//', edgecolor = 'white', width=bar_width)
                    else :
                        plt.bar(x_pos, diffs[i], bottom=bottom, color=env_color_dict[target_base2], hatch = '//', edgecolor = 'white', width=bar_width)

                    # Store bottom and top for connecting curves
                    bottoms[idx].append(bottom)
                    bottom += diffs[i]
                    tops[idx].append(bottom)
                else:
                    plt.bar(x_pos, diffs[i], bottom=bottom, color=colors[i], width=bar_width)
                    # Store bottom and top for connecting curves
                    bottoms[idx].append(bottom)
                    bottom += diffs[i]
                    tops[idx].append(bottom)

            # Connect the corresponding segments between bar charts with filled areas
        for i in range(k+2):

            if i == k:
                x_vals = np.linspace(x_right_edges[0], x_left_edges[1], 100)
                bottom_curve = np.linspace(bottoms[0][i], bottoms[1][i], 100)
                top_curve = np.linspace(tops[0][i], tops[1][i], 100)
                plt.fill_between(x_vals, bottom_curve, top_curve, color=env_color_dict[target_base1], hatch = '//', edgecolor = 'white',alpha=0.3)

                x_vals = np.linspace(x_right_edges[1], x_left_edges[2], 100)
                bottom_curve = np.linspace(bottoms[1][i], bottoms[2][i], 100)
                top_curve = np.linspace(tops[1][i], tops[2][i], 100)
                plt.fill_between(x_vals, bottom_curve, top_curve, color=env_color_dict[target_base2], hatch = '//', edgecolor = 'white', alpha=0.3)
            
            else:
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

        plt.yticks([0,.2,.4,.6, .8, 1],[0,.2,.4,.6, .8, 1], fontsize = 28)
        plt.ylabel('Fraction of Variance Explained', fontsize=28)

        # Remove upper and right spines
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)


        # Add titles and adjust layout
        plt.tight_layout()
        if focal_base == '2Day':
            plt.savefig(f'plots/fig5_c_2Day_rev.png', dpi=300)
        elif focal_base == 'Salt':
            plt.savefig(f'plots/fig5_c_salt_rev.png', dpi=300)
        elif focal_base == '1Day':
            plt.savefig(f'plots/fig5_c_1day_rev.png', dpi= 300)
        # plt.show()
        plt.close()


