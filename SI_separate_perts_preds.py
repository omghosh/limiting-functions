# let's make the bar plots 
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

environment_dict['Salt'] = [env for env in environment_dict['Salt'] if env != 'Batch3_Salt_NS_fitness']

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

env_color_dict_list =  {'2Day': two_day_color_scheme, '1Day': one_day_color_scheme, 'Salt': salt_color_scheme}

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
    print(f'Building up prediction for all {k} components, one left out env at a time')
    print(f"Total sum of squares in focal base: {total_sum_squares}")
    print(f"Sum of squared errors from residuals: {square_sum_errors}")
    print(f'Variance explained in focal base: {1-square_sum_errors/total_sum_squares}')

    print(f'Now building up prediction for focal base one component at a time')
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
        # print(f'Prediction from component {i}, one left out env at a time')
        # print(f"Total sum of squares in focal base: {total_sum_squares}")
        # print(f"Sum of squared errors from residuals: {square_sum_errors}")
        print(f'LOO Fraction of variance explained by residuals with component {(i+1)}: {1 - square_sum_errors/total_sum_squares}')
        focal_base_var_explained_by_component.append(1-square_sum_errors/total_sum_squares)

    print(f'Now building up prediction for target base one component at a time')
        
    U1, s1, Vt1 = np.linalg.svd(focal_base, full_matrices=False)

    U_k1 = U1[:, :k]  # First k left singular vectors

    s_k1 = s1[:k]

    U2, s2, Vt2 = np.linalg.svd(target_base, full_matrices=False)   
    s_k2 = s2[:k]


    # Calculate noise in target base at k components
    s2_noise = np.sum(s2[k:]**2) / np.sum(s2**2)

    s1_noise = np.sum(s1[k:]**2) / np.sum(s1**2)
    # compare to within focal base (not LOO version)
    print("compare to within focal base (not LOO version")
    for i in range(k):
        design_matrix_one = U_k1[:, i].reshape(-1, 1)
        coefficients1_one, residuals1_one, rank1_one, s_vals1_one = np.linalg.lstsq(design_matrix_one, focal_base, rcond=None)
        # print(f"Coefficient shape: {coefficients1_one.shape}")  # Should be (1, 10)
        # Make predictions
        X_pred1_one = design_matrix_one @ coefficients1_one
        # print(f'Sum of squared errors from residuals with component {(i+1)}: {np.sum(residuals1_one)}')

        print(f'Full model fraction of variance explained with component {(i+1)}: {1 - np.sum(residuals1_one)/total_sum_squares}')
    
    
    target_tss = np.sum((target_base)**2 )# - focal_base.mean())**2)
    print(f"Total sum of squares in target: {target_tss}")
    for i in range(k):
        design_matrix_one = U_k1[:, i].reshape(-1, 1)
        coefficients1_one_target, residuals1_one_target, rank1_one_target, s_vals1_one_target = np.linalg.lstsq(design_matrix_one, target_base, rcond=None)
        X_pred2_one = design_matrix_one @ coefficients1_one_target
        # print(f'Sum of squared errors from residuals with component ({i+1}) in target: {np.sum(residuals1_one_target)}')
        print(f'Fraction of variance explained by residuals with component {i+1} in target: {1 - np.sum(residuals1_one_target)/target_tss}')
        target_base_var_explained_by_component.append(1 - np.sum(residuals1_one_target)/target_tss)


    for i in range(k):
        design_matrix_buildup = U_k1[:, :i+1]  # Shape: (m, i)
        coefficients1_buildup, residuals1_buildup, rank1_buildup, s_vals1_buildup = np.linalg.lstsq(design_matrix_buildup, target_base, rcond=None)    
        # print(f'Sum of squared errors from residuals with up to {i+1} components in target: {np.sum(residuals1_buildup)}')
        # print(f'Fraction of variance explained by residuals with up to {i+1} components in target: {1 - np.sum(residuals1_buildup)/target_tss}')

    results['focal_base_var_explained_by_component'] = np.array(focal_base_var_explained_by_component)
    results['target_base_var_explained_by_component'] = np.array(target_base_var_explained_by_component)
    results['target_noise_at_k'] = s2_noise
    results['focal_noise_at_k'] = s1_noise

    return results


k = 8

mutant_names = ['all', 'original', 'anc: IRA1_NON', 'anc: IRA1_MIS', 'anc: GPB2', 'anc: TOR1', 'anc: CYR1']

for mut in mutant_names[:1]:
    if 'anc' in mut:
        which_mutants = mutant_dict[mut]
    elif mut == 'original':
        which_mutants =  mutant_dict['Original Training'] + mutant_dict['Original Testing']
    elif mut == 'all':
        which_mutants = fitness_df.index
        print(f'Working on {mut}')


# now separate out the different perturbation


    for focal_base in ['2Day', '1Day', 'Salt']:
        print(focal_base)
        for target_base1 in ['2Day', '1Day', 'Salt']:
            if focal_base == target_base1:
                continue
            X1 = organized_perturbation_fitness_df.loc[which_mutants, environment_dict[focal_base]].values
            X2 = organized_perturbation_fitness_df.loc[which_mutants, environment_dict[target_base1]].values
            # X3 = organized_perturbation_fitness_df.loc[which_mutants, environment_dict[target_base2]].values

            num_envs_per_hub = X1.shape[1]
            num_phenotypes_to_scan = 6

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
            x_positions = np.linspace(1,30, num_envs_per_hub)  # Spaced out to leave room for connections
            bar_width = 1  # Width of the bars

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

            num_envs_in_target_base = X2.shape[1]

            results_dict_base2 = {}
            predicted_X2 = np.zeros_like(X2)
            env_to_predict_labels = []
            for pert in range(num_envs_in_target_base):
                label = (environment_dict[target_base1][pert]).split('_')[2]
                env_to_predict_labels.append(label)
                r_squared_list_base2  = []
                for j in range(num_phenotypes_to_scan):
                    U,S,V = np.linalg.svd(X1, full_matrices=False)
                    design_matrix_one = U[:, j].reshape(-1, 1)
                    env_to_predict = X2[:, pert]
                    coefficients_f_r, residuals_f_r, rank_f_r, s_vals_f_r = np.linalg.lstsq(design_matrix_one, env_to_predict, rcond=None)
                    predicted_env = design_matrix_one @ coefficients_f_r
                    predicted_X2[:, pert] = predicted_env
                    sum_squared_error = np.sum((env_to_predict - predicted_env)**2)
                    total_sum_squares = np.sum(env_to_predict**2)
                    r_squared = 1 - sum_squared_error/total_sum_squares
                    r_squared_list_base2.append(r_squared)
                results_dict_base2[pert] = r_squared_list_base2

            plt.figure(figsize=(8, 8), dpi=100)

            #Panel 4: variance explained by each component for each environment (within base 1) 
            for idx, (diffs, x_pos) in enumerate(zip(var_exp_list, x_positions)):
                bottom = 0
                for i in range(num_phenotypes_to_scan):
                    plt.bar(x_pos, diffs[i], bottom=bottom, color = env_color_dict_list[focal_base][i] ,width=bar_width)
                    # Store bottom and top for connecting curves
                    bottoms[idx].append(bottom)
                    bottom += diffs[i]
                    tops[idx].append(bottom)



                # Connect the corresponding segments between bar charts with filled areas

            for c in range(num_envs_per_hub-1):
                for i in range(num_phenotypes_to_scan):
                    x_vals = np.linspace(x_right_edges[c], x_left_edges[c+1], 100)
                    bottom_curve = np.linspace(bottoms[c][i], bottoms[c+1][i], 100)
                    top_curve = np.linspace(tops[c][i], tops[c+1][i], 100)
                    plt.fill_between(x_vals, bottom_curve, top_curve, color=env_color_dict_list[focal_base][i], alpha=0.3)




            plt.xticks(x_positions, held_out_env_labels, rotation = 90)
            plt.ylabel('Variance Explained')
            plt.xlabel('Environment')
            plt.title(f'Focal base {focal_base}, by Environment')
            plt.ylim([-0.01, 1.1])
            plt.axhline(y=1, color='gray', linestyle='--', linewidth=1)
            plt.tight_layout()

            plt.savefig(f'plots/SI/{focal_base}_within_{mut}.png')
            # plt.show()
            plt.close()

            x_positions = np.linspace(1,30, num_envs_in_target_base)  # Spaced out to leave room for connections
            bar_width = 1 # Width of the bars

            # # Calculate the x-coordinates of the left and right edges of each bar
            x_left_edges = [x - bar_width/2 for x in x_positions]
            x_right_edges = [x + bar_width/2 for x in x_positions]
            bottoms = []
            tops = []

            for p in range(num_envs_in_target_base):
                bottoms.append([])
                tops.append([])
            var_exp_list_target = []
            for i in range(num_envs_in_target_base):
                var_exp_list_target.append(results_dict_base2[i])


            plt.figure(figsize=(8, 8), dpi=100)


            for idx, (diffs, x_pos) in enumerate(zip(var_exp_list_target, x_positions)):
                bottom = 0
                for i in range(num_phenotypes_to_scan):
                    plt.bar(x_pos, diffs[i], bottom=bottom, color = env_color_dict_list[focal_base][i] ,width=bar_width)
                    # Store bottom and top for connecting curves
                    bottoms[idx].append(bottom)
                    bottom += diffs[i]
                    tops[idx].append(bottom)


            for c in range(num_envs_in_target_base-1):
                for i in range(num_phenotypes_to_scan):
                    x_vals = np.linspace(x_right_edges[c], x_left_edges[c+1], 100)
                    bottom_curve = np.linspace(bottoms[c][i], bottoms[c+1][i], 100)
                    top_curve = np.linspace(tops[c][i], tops[c+1][i], 100)
                    plt.fill_between(x_vals, bottom_curve, top_curve, color=env_color_dict_list[focal_base][i], alpha=0.3)


            plt.xticks(x_positions, env_to_predict_labels, rotation = 90)
            plt.ylabel('Variance Explained')
            plt.xlabel('Environment')
            plt.title(f'Target base {target_base1} from focal base {focal_base}, by Environment')
            plt.ylim([-0.01, 1.1])
            plt.axhline(y=1, color='gray', linestyle='--', linewidth=1)

            plt.tight_layout()
            plt.savefig(f'plots/SI/{focal_base}_to_{target_base1}_{mut}_by_pert.png')
            # plt.show()
            plt.close()