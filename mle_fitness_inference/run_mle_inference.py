#################################################################################################################
# Script to run maximum likelihood fitness inference:                                                            #
#                                                                                                                 #
# Uses functions from mle_inference_functions.py to infer fitness from a set of sequenced barcode read counts   #
# Assumes read count is a negative binomial RV, infers overdispersion parameter and mean fitness, and then      #
# uses grid method to find the maximum likelihood fitness coefficient s and initial frequency                     #
#                                                                                                                #
# Olivia M. Ghosh                                                                                                 #
# adapted from Ascensao et al. 2022                                                                                #
#################################################################################################################

import re
from mle_inference_functions import * 
import sys
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import scipy.optimize
from math import gamma, exp, log
import math
from scipy.stats import nbinom, norm, invgamma
import csv
import seaborn as sns



inference_results={}

###########################
# input sample information#
###########################
batch = sys.argv[1] #'Batch4'
home_condition = sys.argv[2] #'2Day'
perturbation = sys.argv[3] #'M3'
rep = sys.argv[4] #1
data = sys.argv[5] #'bc_counts.csv'

#############################
# generate sample bc_counts #
#############################

# Read in full dataset 
all_bc_counts = pd.read_csv(data, index_col = 'barcode')


# Pick out only timepoints corresponding to the sample we want to calculate
rep_times = [f'{batch}_{home_condition}-T0_combined'] + [f'{batch}_{home_condition}_{perturbation}-R{rep}-T{t}_combined' for t in range(1,5)]

# which samples in reptimes are actually columns in all_bc_counts
rep_times = [col for col in rep_times if col in all_bc_counts.columns]

# Generate sample-specific dataframe of barcode counts 
sample_bc_counts = all_bc_counts[rep_times]

# Filter sample-specific data for low-coverage timepoints and return the dataframe that we will use to infer fitness
# This is an m x t dimensional dataframe with m= number of barcodes/mutants and t = number of timepoints above coverage threshold
bc_counts = filter_timepoints(sample_bc_counts) 


# Specify which barcodes are known neutrals - if it's a simulated dataset, use first 60 barcodes as neutrals
if 'simulation' in data:
    neutrals = np.arange(60)
else:
    neutrals = true_neutrals

# Generate list of numbers corresponding to timepoints included in filtered barcode counts table. 
# ie if timepoint 2 was below the coverage threshold, timepoints = [0,1,3,4]
timepoints = [int(re.search(r'T(\d)', col).group(1)) for col in bc_counts.columns]

print(timepoints)
##########################
# infer noise parameters #
##########################

# Initial guess for the effective population size (somewhat arbitrary)
Ne0 = 1e9

# Infer c values for each timepoint in our filtered sample-specific dataset
cs = infer_cs(bc_counts,timepoints, Ne0, neutrals, batch, home_condition, perturbation, rep)

#######################
# infer mean fitness  #
#######################

xbars = get_xbars(bc_counts,timepoints, neutrals,batch, home_condition, perturbation, rep)

############################################
# infer mle fitness and f0 for each barcode#
############################################
fitness_dict={}
fiterror_dict={}
freq_dict={}
likelihood_dict={}


# Loop through each barcode (~5027 total) and calculate fitness and standard error
for e,bc in enumerate(bc_counts.index):
    print(f'Barcode {e}', flush = True)

    # Generate the grid for this barcode
    s_list, f_list = get_grid(bc, bc_counts, xbars, timepoints)

    # Get likelihood grid, max likelihood s, and max likelihood f0
    log_likelihood_df, s, f0, output_s_list, output_f_list = infer_max_likelihood_params(s_list, f_list, bc,bc_counts,cs,xbars,timepoints)
    print(f'max likelihood s: {s}, max likelihood f0: {f0}', flush = True)
    if np.isnan(s) or np.isnan(f0):
        print(f'Barcode {bc} failed to converge', flush = True)
        fitness_dict[bc]=np.nan
        freq_dict[bc]=np.nan
        fiterror_dict[bc]= np.nan
        likelihood_dict[bc] = np.nan

        continue

    # Pick out row of likelihood grid corresponding to f0 = max_likelihood_f0 to use as one dimensional likelihood function
    likelihood_1d = (log_likelihood_df.iloc[np.where(output_f_list==f0)[0]].values).reshape((100,))

    # max_ll is the maximum value of the likelihood function 
    max_ll = likelihood_1d[np.where(output_s_list == s)[0]]

    ####################
    # infer error bars #
    ####################

    # Infer standard error using one dimensional likelihood function
    standard_error = std_error(likelihood_1d, s_list)


    #approx_standard_error = std_from_approx(s, max_ll, f0,bc, sample_bc_counts,xbars,cs,timepoints)

    #################
    # Save results  #
    #################

    fitness_dict[bc]=s
    
    freq_dict[bc]=f0

    fiterror_dict[bc]= standard_error

    likelihood_dict[bc] = likelihood_1d

direc1= 'results'
if not os.path.exists(f'{direc1}/'):
    os.makedirs(f'{direc1}/')



# save results as a dataframe where rows are indexed by barcode and columns are mle fitness, mle f0, standard error, 
# and the full one-dimensional likelihood function (so we can look at quantiles down the line)
inference_results[f'{batch}_{home_condition}_{perturbation}-R{rep}_fitness'] = fitness_dict
inference_results[f'{batch}_{home_condition}_{perturbation}-R{rep}_f0'] = freq_dict
inference_results[f'{batch}_{home_condition}_{perturbation}-R{rep}_stderror'] = fiterror_dict
inference_results[f'{batch}_{home_condition}_{perturbation}-R{rep}_likelihood1d'] = likelihood_dict


# Save intermediate inferred parameters (noise parameters c and mean fitnesses x_bar) as a separate dataframe
intermediate_parameters_dict={}
intermediate_parameters_dict['timepoints'] = timepoints
intermediate_parameters_dict['cs'] = cs
intermediate_parameters_dict['xbars']=xbars

direc2= 'intermediate_parameters'
if not os.path.exists(f'{direc2}/'):
    os.makedirs(f'{direc2}/')

(pd.DataFrame(intermediate_parameters_dict)).to_csv(f'intermediate_parameters/{batch}_{home_condition}_{perturbation}-R{rep}.csv')

(pd.DataFrame(inference_results)).to_csv(f'results/{batch}_{home_condition}_{perturbation}-R{rep}_results.csv')



###############
# Visualize   #
###############

# We plot frequency trajectories (with neutrals in red) and the mean-fitness corrected frequency trajectories (neutrals should look flat)
# Second plots are total coverage (Rs) and c values inferred 


direc3= 'plots'
if not os.path.exists(f'{direc3}/'):
    os.makedirs(f'{direc3}/')

sns.set_style('darkgrid')

fig, axs = plt.subplots(2, 2, figsize = (12,8))
#plot trajectories (first plot) 
Rs = bc_counts.sum(axis = 0)
for bc in bc_counts.index:
    if bc in true_neutrals:
        axs[0,0].semilogy((bc_counts.loc[bc]/Rs), color = 'red', alpha = 0.5)
    else:
        axs[0,0].semilogy((bc_counts.loc[bc]/Rs), alpha = 0.2)
axs[0,0].set_title('Frequency trajectories')
axs[0,0].set_ylabel('frequency')
axs[0,0].set_xticklabels([f'T{t}' for t in timepoints])

#plot trajectories normalized to mean fitness (neutrals should be straight) (second plot)
for bc in bc_counts.index:
    if bc in true_neutrals:
        axs[0,1].semilogy((bc_counts.loc[bc]/Rs)*(np.exp(xbars*np.array(timepoints))), color = 'red', alpha = 0.5)
    else:
        axs[0,1].semilogy((bc_counts.loc[bc]/Rs)*(np.exp(xbars*np.array(timepoints))), alpha = 0.2)
axs[0,1].set_title('Frequency trajectories w/ mean fitness correction')
axs[0,1].set_ylabel('frequency')
axs[0,1].set_xticklabels([f'T{t}' for t in timepoints])

axs[1,0].semilogy(Rs, '-o', color = 'slateblue')
axs[1,0].set_title('Total Coverage')
axs[1,0].set_ylabel('Reads mapped to barcodes')
axs[1,0].set_xticklabels([f'T{t}' for t in timepoints])

axs[1,1].plot(np.array(timepoints), cs, '-o', color = 'salmon')
axs[1,1].set_title('Inferred noise parameter')
axs[1,1].set_ylabel('c')
axs[1,1].set_xticks(np.array(timepoints))
axs[1,1].set_xticklabels([f'T{t}' for t in timepoints])

plt.savefig(f'plots/{batch}_{home_condition}_{perturbation}-R{rep}.png')
plt.close()


#TODO: 
# in the end, plot expected trajectories vs real trajectories
# bootstrap over neutrals to get uncertainty in kappas? (lower priority) 
# PUT IN ESCAPE VALVE (if it needs to recalculate some # of times, return nan)
# calculate Z score variances and compare to cs 

