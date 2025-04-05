# let's make the bar plots but separate out each predicted condition

import numpy as np
import matplotlib.pyplot as plt
from functions import *
from scipy.linalg import lstsq
from sklearn.metrics import r2_score
import seaborn as sns
from scipy.stats import spearmanr


bc_counts, fitness_df, grants_df_with_barcode_df, full_df = create_full_fitness_dataframe()

organized_perturbation_fitness_df = create_delta_fitness_matrix(batches, fitness_df, environment_dict)

env_color_dict = {'2Day': (0.77, 0.84, 0.75), '1Day': (0.55, 0.6, 0.98), 'Salt': (1, 0.59, 0.55)}

# remove 'Batch3_Salt_NS_fitness' from the environment_dict
#  'Batch3_Salt_NS_fitness' remove from environment_dict['Salt]
environment_dict['Salt'] = [env for env in environment_dict['Salt'] if env != 'Batch3_Salt_NS_fitness']

print(environment_dict['Salt'])
mut = 'original'
which_mutants =  mutant_dict['Original Training'] + mutant_dict['Original Testing']
focal_base = '2Day'
target_base1 = 'Salt'
# target_base2 = '1Dayfor focal_base, target_base1, target_base2 in [('Salt', '1Day', '2Day'), ('1Day', '2Day', 'Salt'), ('2Day', 'Salt', '1Day')]:

for focal_base in ['2Day', '1Day', 'Salt']:
    for target_base1 in ['2Day', '1Day', 'Salt']:
        if focal_base == target_base1:
            continue
        X1 = organized_perturbation_fitness_df.loc[which_mutants, environment_dict[focal_base]].values
        X2 = organized_perturbation_fitness_df.loc[which_mutants, environment_dict[target_base1]].values
        # X3 = organized_perturbation_fitness_df.loc[which_mutants, environment_dict[target_base2]].values

        num_envs_per_hub = X1.shape[1]
        num_phenotypes_to_scan = 10

        U1, s1, V1 = np.linalg.svd(X1, full_matrices=False)
        U2, s2, V2 = np.linalg.svd(X2, full_matrices=False)
        U,s,V = np.linalg.svd(np.hstack((X1,X2)), full_matrices=False)
        ## Linear regression 
        results_dict = {}
        predicted_X1 = np.zeros_like(X1)
        held_out_env_labels = []
        for held_out_index in range(num_envs_per_hub):
            label = (environment_dict[focal_base][held_out_index]).split('_')[2]
            held_out_env_labels.append(label)
            r_squared_list  = []
            for j in range(num_phenotypes_to_scan):
                held_out_env = X1[:, held_out_index]
                X1_reduced = np.delete(X1, held_out_index, axis=1)
                U1_reduced, s1_reduced, V1_reduced = np.linalg.svd(X1_reduced, full_matrices=False)
                design_matrix_1 = U1_reduced[:, j].reshape(-1, 1)
                coefficients_f_r, residuals_f_r, rank_f_r, s_vals_f_r = np.linalg.lstsq(design_matrix_1, held_out_env, rcond=None)
                predicted_held_out_env = design_matrix_1 @ coefficients_f_r
                predicted_X1[:, held_out_index] = predicted_held_out_env
                sum_squared_error = np.sum((held_out_env - predicted_held_out_env)**2)
                total_sum_squares = np.sum(held_out_env**2)
                r_squared = 1 - sum_squared_error/total_sum_squares
                r_squared_list.append(r_squared)
            results_dict[held_out_index] = r_squared_list


        # Define x-positions for the bar charts
        x_positions = np.linspace(1,43, num_envs_per_hub)  # Spaced out to leave room for connections
        bar_width = 0.95  # Width of the bars

        # # Calculate the x-coordinates of the left and right edges of each bar
        x_left_edges = [x - bar_width/2 for x in x_positions]
        x_right_edges = [x + bar_width/2 for x in x_positions]
        bottoms = []
        tops = []

        for p in range(num_envs_per_hub):
            bottoms.append([])
            tops.append([])

        var_exp_list = []
        for i in range(num_envs_per_hub):
            var_exp_list.append(results_dict[i])


        # now let's do this "averaged" over the whole base - don't separate out the different perturbations 
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
        for i in range(num_envs_per_hub):
            design_matrix_one = U1[:, i].reshape(-1, 1)
            coefficients1_one_target, residuals1_one_target, rank1_one_target, s_vals1_one_target = np.linalg.lstsq(design_matrix_one, X2, rcond=None)
            X2_pred_one = design_matrix_one @ coefficients1_one_target
            base2_var_explained_by_component.append(1 - np.sum(residuals1_one_target)/base2_total_sum_of_squares)
 

        ######################
        ####### PLOTTING #####
        ######################
        colors = sns.color_palette("pastel", num_envs_per_hub)

        fig, axs = plt.subplots(2, 3, figsize=(15,9))


        # Panel 1: Full fitness matrix focal base
        sns.heatmap(X1, ax=axs[0,0], cmap='Blues')
        axs[0,0].set_xlabel('Environment')
        axs[0,0].set_ylabel('Mutant')
        axs[0,0].set_title(f'{focal_base} delta Fitness Matrix')


        # panel 2: Full fitness matrix target base
        sns.heatmap(X2, ax=axs[0,1], cmap='Greens')
        axs[0,1].set_xlabel('Environment')
        axs[0,1].set_ylabel('Mutant')
        axs[0,1].set_title(f'{target_base1} delta fitness Matrix')

        # panel 3: SVD 
        axs[0,2].semilogy(s1**2/np.sum(s1**2), 'o-', label=f'{focal_base}')
        axs[0,2].semilogy(s2**2/np.sum(s2**2), 'o-', label=f'{target_base1}')
        axs[0,2].semilogy(s**2/np.sum(s**2), 'o-', label='All')
        # axs[0,2].set_xlim([-.5, 15])
        axs[0,2].legend()
        axs[0,2].set_xlabel('Component')
        axs[0,2].set_ylabel('Variance Explained')
        axs[0,2].set_title('Scree Plot from SVD')
        axs[0,2].set_xlim([-0.5, 25])


        #Panel 4: variance explained by each component for each environment (within base 1) 
        for idx, (diffs, x_pos) in enumerate(zip(var_exp_list, x_positions)):
            bottom = 0
            for i in range(num_phenotypes_to_scan):
                axs[1,0].bar(x_pos, diffs[i], bottom=bottom, color = colors[i] ,width=bar_width)
                # Store bottom and top for connecting curves
                bottoms[idx].append(bottom)
                bottom += diffs[i]
                tops[idx].append(bottom)
        axs[1,0].set_xticks(x_positions, held_out_env_labels, rotation = 90)
        axs[1,0].set_ylabel('Variance Explained')
        axs[1,0].set_xlabel('Environment')
        axs[1,0].set_title(f'Variance Explained by Each Component in {focal_base}, by Environment')
        axs[1,0].set_ylim([-0.01, 1.1])

        # Panel 5: variance explained by each component in base 1 (all at once) 

        # Define x-positions for the two bar charts
        x_positions = [1, 3]  # Spaced out to leave room for connections
        bar_width = 0.8  # Width of the bars

        # Calculate the x-coordinates of the left and right edges of each bar
        x_left_edges = [x - bar_width/2 for x in x_positions]
        x_right_edges = [x + bar_width/2 for x in x_positions]

        # Store bottoms and tops for each bar chart
        bottoms = [[], []]
        tops = [[], []]

        # Plot the stacked bars
        for idx, (diffs, x_pos) in enumerate(zip([var_exp_by_component_full_base, base2_var_explained_by_component], x_positions)):
            bottom = 0
            for i in range(num_phenotypes_to_scan):
                axs[1,1].bar(x_pos, diffs[i], bottom=bottom, color=colors[i], width=bar_width)
                # Store bottom and top for connecting curves
                bottoms[idx].append(bottom)
                bottom += diffs[i]
                tops[idx].append(bottom)

            # Connect the corresponding segments between bar charts with filled areas
        for i in range(num_phenotypes_to_scan):
            x_vals = np.linspace(x_right_edges[0], x_left_edges[1], 100)
            bottom_curve = np.linspace(bottoms[0][i], bottoms[1][i], 100)
            top_curve = np.linspace(tops[0][i], tops[1][i], 100)
            axs[1,1].fill_between(x_vals, bottom_curve, top_curve, color=colors[i], alpha=0.3)
        axs[1,1].set_ylabel('Variance Explained')
        axs[1,1].set_xticks(x_positions, [f'{focal_base}', f'{target_base1}'])
        axs[1,1].set_title(f'Predicting left out envs from {focal_base}')
        axs[1,1].axhline(1,linestyle =':', color='black', linewidth=0.5)
        axs[1,1].set_ylim([-0.01, 1.1])


        # Panel 5 : variance explained by original SVD components
        bottom = 0
        for i in range(num_envs_per_hub):
            axs[1,2].bar(i+1, s1[i]**2/np.sum(s1**2), bottom=bottom, color = colors[i], width=bar_width*2, alpha = 0.75)
            bottom += s1[i]**2/np.sum(s1**2)
        axs[1,2].scatter(range(1, num_envs_per_hub+1), np.cumsum(s1**2)/np.sum(s1**2), color=colors)

        # axs[1,2].set_xticks(range(1, 20), [str(i) for i in range(1, 20)])
        axs[1,2].set_ylabel('Variance Explained')
        axs[1,2].set_xlabel('Number of Components')
        axs[1,2].set_title(f'Variance Explained by Each Component in {focal_base} (SVD)')
        axs[1,2].set_ylim([-0.01, 1.1])


        for ax1 in (axs): 
            for ax2 in ax1:
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(f'../plots/{focal_base}_to_{target_base1}_{mut}.png')
        # plt.show()
        plt.close()

# def variance_explained_in_and_out(focal_base, target_base, k):
#     results = {}
#     focal_base_var_explained_by_component =[]
#     target_base_var_explained_by_component = []
#     predicted_focal_base_by_component = []

#     # Build up predicted fitness matrix env by env:
#     predicted_focal_base = np.zeros_like(focal_base)

#     for j in range(focal_base.shape[1]):
#         held_out_env = focal_base[:, j]
#         reduced_focal_base = np.delete(focal_base, j, axis=1)
#         U_f_r, s_f_r, Vt_f_r = np.linalg.svd(reduced_focal_base, full_matrices=False)
#         U_f_r_k = U_f_r[:, :k]

#         design_matrix_f_r = U_f_r_k  # Shape: (m, k)

#         # Perform least squares regression to find coefficients
#         coefficients_f_r, residuals_f_r, rank_f_r, s_vals_f_r = np.linalg.lstsq(design_matrix_f_r, held_out_env, rcond=None)
#         predicted_held_out_env = design_matrix_f_r @ coefficients_f_r
#         predicted_focal_base[:, j] = predicted_held_out_env

#     # Calculate error
#     square_sum_errors = np.sum((focal_base - predicted_focal_base)**2)

#     total_sum_squares = np.sum((focal_base)**2)
#     # print(f'Building up prediction for all {k} components, one left out env at a time')
#     # print(f"Total sum of squares in focal base: {total_sum_squares}")
#     # print(f"Sum of squared errors from residuals: {square_sum_errors}")
#     # print(f'Variance explained in focal base: {1-square_sum_errors/total_sum_squares}')

#     # print(f'Now building up prediction for focal base one component at a time')
#     for i in range(k):
#         # Build up predicted fitness matrix env by env:
#         predicted_focal_base = np.zeros_like(focal_base)

#         for j in range(focal_base.shape[1]):
#             held_out_env = focal_base[:, j]
#             reduced_focal_base = np.delete(focal_base, j, axis=1)
#             U_f_r, s_f_r, Vt_f_r = np.linalg.svd(reduced_focal_base, full_matrices=False)
#             U_f_r_k = U_f_r[:, :k]
#             design_matrix_one = U_f_r_k[:, i].reshape(-1, 1)


#             # Perform least squares regression to find coefficients
#             coefficients_f_r, residuals_f_r, rank_f_r, s_vals_f_r = np.linalg.lstsq(design_matrix_one, held_out_env, rcond=None)
#             # Make predictions
#             predicted_held_out_env = design_matrix_one @ coefficients_f_r
#             predicted_focal_base[:, j] = predicted_held_out_env

#         # Calculate error
#         square_sum_errors = np.sum((focal_base - predicted_focal_base)**2)
#         total_sum_squares = np.sum((focal_base)**2)

#         # print(f'LOO Fraction of variance explained by residuals with component {(i+1)}: {1 - square_sum_errors/total_sum_squares}')
#         focal_base_var_explained_by_component.append(1-square_sum_errors/total_sum_squares)

#     # print(f'Now building up prediction for target base one component at a time')
        
#     U1, s1, Vt1 = np.linalg.svd(focal_base, full_matrices=False)

#     U_k1 = U1[:, :k]  # First k left singular vectors

#     s_k1 = s1[:k]

#     U2, s2, Vt2 = np.linalg.svd(target_base, full_matrices=False)   
#     s_k2 = s2[:k]


#     # Calculate noise in target base at k components
#     s2_noise = np.sum(s2[k:]**2) / np.sum(s2**2)

#     s1_noise = np.sum(s1[k:]**2) / np.sum(s1**2)
#     # compare to within focal base (not LOO version)
#     # print("compare to within focal base (not LOO version")
#     for i in range(k):
#         design_matrix_one = U_k1[:, i].reshape(-1, 1)
#         coefficients1_one, residuals1_one, rank1_one, s_vals1_one = np.linalg.lstsq(design_matrix_one, focal_base, rcond=None)
#         # print(f"Coefficient shape: {coefficients1_one.shape}")  # Should be (1, 10)
#         # Make predictions
#         X_pred1_one = design_matrix_one @ coefficients1_one
#         predicted_focal_base_by_component.append(X_pred1_one)
#         # print(f'Sum of squared errors from residuals with component {(i+1)}: {np.sum(residuals1_one)}')
#         # print(f'Full model fraction of variance explained with component {(i+1)}: {1 - np.sum(residuals1_one)/total_sum_squares}')
    
    
#     target_tss = np.sum((target_base)**2 )# - focal_base.mean())**2)
#     # print(f"Total sum of squares in target: {target_tss}")
#     for i in range(k):
#         design_matrix_one = U_k1[:, i].reshape(-1, 1)
#         coefficients1_one_target, residuals1_one_target, rank1_one_target, s_vals1_one_target = np.linalg.lstsq(design_matrix_one, target_base, rcond=None)
#         X_pred2_one = design_matrix_one @ coefficients1_one_target
#         # print(f'Sum of squared errors from residuals with component ({i+1}) in target: {np.sum(residuals1_one_target)}')
#         # print(f'Fraction of variance explained by residuals with component {i+1} in target: {1 - np.sum(residuals1_one_target)/target_tss}')
#         target_base_var_explained_by_component.append(1 - np.sum(residuals1_one_target)/target_tss)
        

#     # for i in range(k):
#     #     design_matrix_buildup = U_k1[:, :i+1]  # Shape: (m, i)
#     #     coefficients1_buildup, residuals1_buildup, rank1_buildup, s_vals1_buildup = np.linalg.lstsq(design_matrix_buildup, target_base, rcond=None)    
#     #     # print(f'Sum of squared errors from residuals with up to {i+1} components in target: {np.sum(residuals1_buildup)}')
#     #     # print(f'Fraction of variance explained by residuals with up to {i+1} components in target: {1 - np.sum(residuals1_buildup)/target_tss}')

#     results['focal_base_var_explained_by_component'] = np.array(focal_base_var_explained_by_component)
#     results['target_base_var_explained_by_component'] = np.array(target_base_var_explained_by_component)
#     results['target_noise_at_k'] = s2_noise
#     results['focal_noise_at_k'] = s1_noise
#     results['predicted_focal_base'] = predicted_focal_base

#     return results




# k = 10

# # mutant_names = ['original', 'anc: IRA1_NON', 'anc: IRA1_MIS', 'anc: GPB2', 'anc: TOR1', 'anc: CYR1', 'all']

# # for mut in mutant_names:
# #     if 'anc' in mut:
# #         which_mutants = mutant_dict[mut]
# #     elif mut == 'original':
# #         which_mutants =  mutant_dict['Original Training'] + mutant_dict['Original Testing']
# #     elif mut == 'all':
# #         which_mutants = fitness_df.index
# #         print(f'Working on {mut}')
# mut = 'original'
# which_mutants =  mutant_dict['Original Training'] + mutant_dict['Original Testing']
# # focal_base = '2Day'
# # target_base1 = 'Salt'
# # target_base2 = '1Day'
# for focal_base, target_base1, target_base2 in [('Salt', '1Day', '2Day'), ('1Day', '2Day', 'Salt'), ('2Day', 'Salt', '1Day')]:

#     X1 = organized_perturbation_fitness_df.loc[which_mutants, environment_dict[focal_base]].values
#     X2 = organized_perturbation_fitness_df.loc[which_mutants, environment_dict[target_base1]].values
#     X3 = organized_perturbation_fitness_df.loc[which_mutants, environment_dict[target_base2]].values

#     # X1 = fitness_df.loc[which_mutants, environment_dict[focal_base]].values
#     # X2 = fitness_df.loc[which_mutants, environment_dict[target_base1]].values
#     # X3 = fitness_df.loc[which_mutants, environment_dict[target_base2]].values

#     # print(f'Working on target base {target_base1} from focal base {focal_base}')
#     variance_explained_in_and_out(X1, X2, k)
#     results_f_t1 = variance_explained_in_and_out(X1, X2, k)
#     # print(f'Fraction of variance explained by {k} components in {focal_base}: {results_f_t1["focal_base_var_explained_by_component"]}')
#     # print(f'Fraction of variance explained by {k} components in {target_base1} from {focal_base}: {results_f_t1["target_base_var_explained_by_component"]}')
#     # print(f'Noise in {target_base1} at {k} components: {results_f_t1["target_noise_at_k"]}')


#     # print(f'Working on target base {target_base2} from focal base {focal_base}')
#     results_f_t2 = variance_explained_in_and_out(X1, X3, k)
#     # print(f'Fraction of variance explained by {k} components in {focal_base}: {results_f_t2["focal_base_var_explained_by_component"]}')
#     # print(f'Fraction of variance explained by {k} components in {target_base2} from {focal_base}: {results_f_t2["target_base_var_explained_by_component"]}')
#     # print(f'Noise in {target_base2} at {k} components: {results_f_t2["target_noise_at_k"]}')

#     print(results_f_t1["predicted_focal_base"].shape)
#     print(X1.shape)

#     for col in range(X1.shape[1]):
#         environment = environment_dict[focal_base][col]
#         print(environment)

#     #     error = X1[:, col] - results_f_t1["predicted_focal_base"][:, col]
#     #     sse = np.sum(error**2)
#     #     tss = np.sum(X1[:, col]**2)
#     #     print(f'Fraction of variance explained in {environment} by residuals: {1 - sse/tss}')



#     f_var_explained = results_f_t1["focal_base_var_explained_by_component"]

#     print(f_var_explained)
    # t1_var_explained = results_f_t1["target_base_var_explained_by_component"]
    # t2_var_explained = results_f_t2["target_base_var_explained_by_component"]

    # t1_noise = results_f_t1["target_noise_at_k"]
    # t2_noise = results_f_t2["target_noise_at_k"]
    # f_noise = results_f_t1["focal_noise_at_k"]
    # # print(t1_noise, t2_noise)
    # # f_noise = 1- np.sum(f_var_explained)
    # unexplained_f = 1 - np.sum(f_var_explained) - f_noise

    # f_var_explained = np.append(f_var_explained, unexplained_f)
    # f_var_explained = np.append(f_var_explained, f_noise)

    # unexplained_t1 = 1 - np.sum(t1_var_explained) - t1_noise
    # t1_var_explained = np.append(t1_var_explained, unexplained_t1)
    # t1_var_explained = np.append(t1_var_explained, t1_noise)

    # # print(t1_var_explained)

    # unexplained_t2 = 1 - np.sum(t2_var_explained) - t2_noise
    # t2_var_explained = np.append(t2_var_explained, unexplained_t2)
    # t2_var_explained = np.append(t2_var_explained, t2_noise)
    # # print(t2_var_explained)

    # # print(results_f_t1["focal_base_var_explained_by_component"])

    # # # spearman correlation between the two sets of variance explained
    # # print(f'Spearman correlation between variance explained in {focal_base} and {target_base1}: {np.round(spearmanr(f_var_explained, t1_var_explained)[0], 2)}')
    # # print(f'Spearman correlation between variance explained in {focal_base} and {target_base2}: {np.round(spearmanr(f_var_explained, t2_var_explained)[0], 2)}')
    # # print(f'Spearman correlation between variance explained in {focal_base} and {focal_base}: {np.round(spearmanr(f_var_explained, f_var_explained)[0], 2)}')

    # spearman_focal_t1 = np.round(spearmanr(f_var_explained[:k], t1_var_explained[:k])[0], 2)
    # spearman_focal_t2 = np.round(spearmanr(f_var_explained[:k], t2_var_explained[:k])[0], 2)


    # ### plot things #####

    # colors = sns.color_palette('mako_r', k) + ['salmon']+['lightgray']

    # # Create figure with proper size
    # plt.figure(figsize=(12, 8))

    # # Define x-positions for the three bar charts
    # x_positions = [1, 3, 5]  # Spaced out to leave room for connections
    # bar_width = 0.8  # Width of the bars

    # # Calculate the x-coordinates of the left and right edges of each bar
    # x_left_edges = [x - bar_width/2 for x in x_positions]
    # x_right_edges = [x + bar_width/2 for x in x_positions]

    # # Store bottoms and tops for each bar chart
    # bottoms = [[], [], []]
    # tops = [[], [], []]

    # # Plot the stacked bars
    # for idx, (diffs, x_pos) in enumerate(zip([f_var_explained, f_var_explained, f_var_explained], x_positions)):
    #     bottom = 0
    #     for i in range(k+2):
    #         plt.bar(x_pos, diffs[i], bottom=bottom, color=colors[i], width=bar_width)
    #         # Store bottom and top for connecting curves
    #         bottoms[idx].append(bottom)
    #         bottom += diffs[i]
    #         tops[idx].append(bottom)

    #     # Connect the corresponding segments between bar charts with filled areas
    # for i in range(k+2):
    #     x_vals = np.linspace(x_right_edges[0], x_left_edges[1], 100)
    #     bottom_curve = np.linspace(bottoms[0][i], bottoms[1][i], 100)
    #     top_curve = np.linspace(tops[0][i], tops[1][i], 100)
    #     plt.fill_between(x_vals, bottom_curve, top_curve, color=colors[i], alpha=0.3)
        
    #     # Connect second and third bar chart - using the edges of the bars
    #     x_vals = np.linspace(x_right_edges[1], x_left_edges[2], 100)
    #     bottom_curve = np.linspace(bottoms[1][i], bottoms[2][i], 100)
    #     top_curve = np.linspace(tops[1][i], tops[2][i], 100)
        
    #     plt.fill_between(x_vals, bottom_curve, top_curve, color=colors[i], alpha=0.3)
    # # add spearman correlation between focal and target on the plot 
    # plt.text(2, 0.1, f'Spearman correlation = {spearman_focal_t1}', fontsize=12, ha='center')
    # plt.text(4, 0.1, f'Spearman correlation = {spearman_focal_t2}', fontsize=12, ha='center')



    # # Add labels and customize
    # plt.ylabel('Fraction of Variance Explained', fontsize=14)
    # # plt.xticks(x_positions, [f'Target 1: {target_base1}', f'Focal: {focal_base}',f'Target 2: {target_base2}'], fontsize=12)
    # plt.ylim(0, 1)

    # # Add titles and adjust layout
    # plt.title(f'LOO within focal, Variance Explained, {mut}, k={k}', fontsize=16)
    # plt.tight_layout()
    # # plt.savefig(f'../plots/LOO_variance_explained_from_{focal_base}_k={k}_{mut}_deltafitness.png')
    # plt.show()
    # plt.close()




