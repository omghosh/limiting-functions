#############################################################
# Functions for maximum likelihood fitness inference:        #
#                                                             #
# Olivia M. Ghosh                                             #
# adapted from Ascensao et al. 2022                            #
#############################################################

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import scipy.optimize
from math import gamma, exp, log
import math
from scipy.stats import nbinom, norm, invgamma


# get rid of timepoints with a total coverage under the coverage threshold, default is 10,000 reads
# return dataframe with only "good" timepoints
def filter_timepoints(sample_bc_counts, coverage_threshold = 1e4):
    column_sums = sample_bc_counts.sum(axis=0)
    filtered_columns = column_sums[column_sums >= coverage_threshold]
    columns_to_keep = filtered_columns.index.tolist()
    filtered_data_frame = sample_bc_counts[columns_to_keep]
    return filtered_data_frame


# List of barcodes corresponding to known neutrals
true_neutrals=['CGCTAAAGACATAATGTGGTTTGTTG_TTTTTAAAATGAAACAAGCTTGTATG',
 'CGCTAAAGACATAATGTGGTTTGTTG_ATCTTAAGATAAAAGGCATTTTATTC',
 'CGCTAAAGACATAATGTGGTTTGTTG_ACAGCAACCGTGAATGTACTTCGCAC',
 'CGCTAAAGACATAATGTGGTTTGTTG_AGTATAAGCGCTAAATAATTTTCTCC',
 'CGCTAAAGACATAATGTGGTTTGTTG_AGCATAATCTTTAACACGCTTGTCAG',
 'CGCTAAAGACATAATGTGGTTTGTTG_CCGTTAAAAAGAAATGATTTTTATAG',
 'CGCTAAAGACATAATGTGGTTTGTTG_CCAGTAATTGGAAAACTCCTTGGGAA',
 'CGCTAAAGACATAATGTGGTTTGTTG_ATAGCAACCCAGAAGATCCTTATCTG',
 'CGCTAAAGACATAATGTGGTTTGTTG_CTGAAAAGAGATAAATATTTTACATC',
 'CGCTAAAGACATAATGTGGTTTGTTG_GGAACAAATCAAAATTAATTAACAAA',
 'CGCTAAAGACATAATGTGGTTTGTTG_TAGGAAATTCCAAAGAAATTTGGTAA',
 'CGCTAAAGACATAATGTGGTTTGTTG_CATCTAATTGATAACTTCTTTCCCGC',
 'CGCTAAAGACATAATGTGGTTTGTTG_ATGGAAACAAAAAATAGTATTCGCAC',
 'CGCTAAAGACATAATGTGGTTTGTTG_TAAGAAAATCAGAACCGCTTTCAGGG',
 'CGCTAAAGACATAATGTGGTTTGTTG_GCTTTAATGAAAAATTATTTTGGCTG',
 'CGCTAAAGACATAATGTGGTTTGTTG_CTCCTAACCAGTAATAACTTTCTCAA',
 'CGCTAAAGACATAATGTGGTTTGTTG_GTCCAAAACAACAAATGCACTTGCAA',
 'CGCTAAAGACATAATGTGGTTTGTTG_TTTTTAATTTGCAAGGACCTTAGTCT',
 'CGCTAAAGACATAATGTGGTTTGTTG_TGCATAACCTGCAAACAGATTGCCGT',
 'CGCTAAAGACATAATGTGGTTTGTTG_GCATCAATGATCAACGCGGTTACCTG',
 'CGCTAAAGACATAATGTGGTTTGTTG_AGGCTAATACCCAATTCGATTGTCAT',
 'CGCTAAAGACATAATGTGGTTTGTTG_TTTTCAAATAGTAATTGATTTCCAGT',
 'CGCTAAAGACATAATGTGGTTTGTTG_AAGCTAATTGAGAATTATTTTGCATT',
 'CGCTAAAGACATAATGTGGTTTGTTG_TTGCGAATACGTAATTTTGTTGCGGG',
 'CGCTAAAGACATAATGTGGTTTGTTG_CAGTAAACGATCAAATACTTTCAATA',
 'CGCTAAAGACATAATGTGGTTTGTTG_TAACAAATGTCTAATGGAATTTTGCA',
 'CGCTAAAGACATAATGTGGTTTGTTG_GTTGTAATCTCGAAGAGATTTCTAGC',
 'CGCTAAAGACATAATGTGGTTTGTTG_ATACTAAAAAGTAAAGTGGTTATTCT',
 'CGCTAAAGACATAATGTGGTTTGTTG_CACGTAACGCAGAAGTGCTTTGAAAG',
 'CGCTAAAGACATAATGTGGTTTGTTG_CTACGAACTACGAAATGCCTTGTTTC',
 'CGCTAAAGACATAATGTGGTTTGTTG_GACTAAAACTGTAACATTTTTAATGG',
 'CGCTAAAGACATAATGTGGTTTGTTG_GAGGAAAATTATAACGAATTTTGTCG',
 'CGCTAAAGACATAATGTGGTTTGTTG_CGATTAACTAATAATTCTTTTTAAAG',
 'CGCTAAAGACATAATGTGGTTTGTTG_TTTCCAAATGTAAAGCGTATTCATAC',
 'CGCTAAAGACATAATGTGGTTTGTTG_GGTAGAATATGGAATGTTTTTACGAA',
 'CGCTAAAGACATAATGTGGTTTGTTG_TCCCCAACCATTAATTATGTTTACAC',
 'CGCTAAAGACATAATGTGGTTTGTTG_ACCCCAAATACCAAAGGAGTTGCGTG',
 'CGCTAAAGACATAATGTGGTTTGTTG_TCGACAAATTGCAAGAAGGTTTCATC',
 'CGCTAAAGACATAATGTGGTTTGTTG_TCGATAATGACCAATACCATTTTGTC',
 'CGCTAAAGACATAATGTGGTTTGTTG_CGGAAAACATTGAACTTTATTAATGG',
 'CGCTAAAGACATAATGTGGTTTGTTG_AGGGAAAACAGGAAACCCGTTTCCCT',
 'CGCTAAAGACATAATGTGGTTTGTTG_TTTTCAACCGGTAATTAAATTCTCGT',
 'CGCTAAAGACATAATGTGGTTTGTTG_ATTCAAATCGAAAATGATGTTCTTCA',
 'CGCTAAAGACATAATGTGGTTTGTTG_AATAGAACCCCAAATATTTTTTCTCG',
 'CGCTAAAGACATAATGTGGTTTGTTG_TGCTTAAGCGCGAAATGCTTTACACC',
 'CGCTAAAGACATAATGTGGTTTGTTG_CGGCCAATTTGGAAGTCGCTTATATA',
 'CGCTAAAGACATAATGTGGTTTGTTG_TTCATAAACTCTAACAACCTTTTTAG',
 'CGCTAAAGACATAATGTGGTTTGTTG_ACGAGAATAGCAAACGCAGTTGAGCG',
 'CGCTAAAGACATAATGTGGTTTGTTG_GCATCAAGAATAAAGCGGGTTCCGCT',
 'CGCTAAAGACATAATGTGGTTTGTTG_CAAATAATCAAGAACGGCCTTATGTT',
 'CGCTAAAGACATAATGTGGTTTGTTG_AAAGAAAAGCTTAAAGATATTGATGA',
 'CGCTAAAGACATAATGTGGTTTGTTG_CAATCAAGGGTCAATTAACTTTTCAA']


##########################
# infer noise parameters #
##########################

# calculates intermediate kappas (variance in shifts of sqrt frequency between timepoints) from neutral barcodes
def calculate_kappas(sample_bc_counts,timepoints, neutrals, batch, home_condition, perturbation, rep):
    kappa_dict={}

    # get dataframe with only neutral barcodes
    neutral_df = sample_bc_counts.loc[neutrals]

    # generate phis (square root of frequency of each neutral barcode at each timepoint)
    freqs = neutral_df/sample_bc_counts.sum(axis=0)
    phis = np.sqrt(freqs)

    # for each pair of timepoints, calculate the variance in the difference between phi for each barcode
    # save as a dictionary of kappas, where the integers represent the timepoints between which it was calculated
    for j in (timepoints):
        for k in (timepoints):
            if j<k:
                if j ==0:
                    kappa_dict[f'kappa_{j}{k}'] = np.var(phis.loc[:,f'{batch}_{home_condition}-T{j}_combined']-phis.loc[:,f'{batch}_{home_condition}_{perturbation}-R{rep}-T{k}_combined'])
                else:
                    kappa_dict[f'kappa_{j}{k}'] = np.var(phis.loc[:,f'{batch}_{home_condition}_{perturbation}-R{rep}-T{j}_combined']-phis.loc[:,f'{batch}_{home_condition}_{perturbation}-R{rep}-T{k}_combined'])
    return kappa_dict


# To fit technical noise parameters eta_i and genetic drift noise paramter Ne
# we need to minimize the difference in expected relationship
# We calculate kappa and the single objective function is for one pair of timepoints, kappa measured minus kappa expected
def obj_single(kappa_jk, eta_j, eta_k, n_transfers, Ne):
    kappa_est = eta_j+eta_k+float(n_transfers)/(4*Ne)
    return (float(kappa_jk)-kappa_est)**2



# Here, we add up all objective functions for each timepoint pair, to fit the proper noise parameters 
# across a full trajectory. params is a vector of noise parameters (eta_i where sum(i) is number of timepoints, Ne) 
# that we are optimizing,
def add_obj(params, sample_dict,timepoints):
    mse =0
    for kappa in sample_dict:

        #get timepoint indices from measured kappa
        j=int(kappa[-2])
        k=int(kappa[-1])

        #exponentiate input parameters (because we are searching log space to numerically minimize function) 
        eta_j = np.exp(params[timepoints.index(j)])
        eta_k = np.exp(params[timepoints.index(k)])
        n_transfers = k-j
        Ne=np.exp(params[-1])

        #get measured kappa 
        kappa_jk = sample_dict[kappa]

        #add up mean squared errors 
        mse+=obj_single(kappa_jk,eta_j,eta_k, n_transfers,Ne)

    #return log of mean squared error (easier numerically to minimize)
    return np.log(mse)



# generate guesses for parameter values at which to start minimization algorithm, and bounds on parameters
def guess_params(sample_bc_counts, Ne0):
    Rs = sample_bc_counts.sum(axis=0)
    param_guess=[]
    bounds=[]

    # technical noise should not be smaller than sampling noise from sequencing, separate guess for each timepoint
    for R in Rs:
        bounds.append((np.log(1/(4*R)),np.log(50/(4*R))))
        param_guess.append(np.log(10/(4*R)))
    param_guess.append(np.log(Ne0))
    bounds.append((np.log(10), np.log(1e20)))
    return param_guess,bounds



# To infer c, we need to minimize the difference between expected kappa and measured kappa given etas and Ne
# We then need to transform back to "read count units" - we obtain dispersion parameters for each timepoint

def infer_cs(sample_bc_counts,timepoints, Ne0, neutrals, batch, home_condition, perturbation, rep):

    # empirically calculate kappas from data
    kappa_dict = calculate_kappas(sample_bc_counts,timepoints, neutrals, batch, home_condition, perturbation, rep)

    # generate intial guesses and bounds for our parameters
    param_guess,bounds =guess_params(sample_bc_counts,Ne0)

    # minimize mean squared error of expected relationship
    res=scipy.optimize.minimize(lambda params: add_obj(params,kappa_dict, timepoints),param_guess, method='L-BFGS-B', bounds =bounds)
    etas=np.exp(res.x[:-1])
    Ne=np.exp(res.x[-1])

    # Transform into c, a read count overdispersion parameter
    Rs = sample_bc_counts.sum(axis=0)
    cs = (4*np.array(etas)+np.array(1/Ne))*np.array(Rs)
    return cs 


######################
# infer mean fitness #
######################

# Use neutral barcodes to infer mean fitness of population with respect to ancestor
# Generate changes in mean fitness between all combinations of timepoints
def infer_all_mean_fitness(sample_bc_counts, timepoints, neutrals ,batch, home_condition, perturbation, rep):
    
    # get dataframe of only neutral barcodes
    neutral_df = sample_bc_counts.loc[neutrals]

    # Take sum of all neutral barcode counts as a single "superbarcode" and get its frequency over time
    neutral_superbarcode = neutral_df.sum(axis=0)
    freqs = neutral_superbarcode/sample_bc_counts.sum(axis=0)

    # generate dictionary of all mean fitness changes over all time intervals 
    xbar_dict ={}

    # t_anchor is the initial timepoint we are comparing later timepoints to, and t is the second timepoint
    for t_anchor in timepoints[:-1]:
        for t in timepoints[(timepoints.index(t_anchor)+1):]:

            # find the log of the frequency of the neutral superbarcode a timepoint t (second timepoint)
            ref_ratio = np.log(freqs[f'{batch}_{home_condition}_{perturbation}-R{rep}-T{t}_combined'])

            # find the log of the frequency of neutral superbarcode at anchor timepoint (initial timepoint)
            # column name for T0 is slightly different, so if t_anchor is 0 we take that into account 
            if t_anchor ==0:
                anchor_ratio =np.log(freqs[f'{batch}_{home_condition}-T{t_anchor}_combined'])
            else:   
                anchor_ratio=np.log(freqs[f'{batch}_{home_condition}_{perturbation}-R{rep}-T{t_anchor}_combined'])

            # Mean fitness change is the negative of the difference between these log frequencies, divided by time interval
            x_bars =-(1/(t-t_anchor))*(ref_ratio-anchor_ratio)
            xbar_dict[f'T{t_anchor}_T{t}']=x_bars

    #return full dictionary with mean fitness changes across all possible timepoint intervals
    return xbar_dict



# to get specific mean fitnesses we want to use for our inference, we have to decide if we want to anchor the 
# comparison frequency in our model at f0, or use single time intervals. Currently, it is written as anchoring at T0
# So default for anchored = True -- TODO add option to use single time intervals
def get_xbars(sample_bc_counts,timepoints, neutrals,batch, home_condition, perturbation, rep,anchored = True):

    # first, we need to get full x_bars dictionary 
    x_bars = infer_all_mean_fitness(sample_bc_counts,timepoints,neutrals, batch, home_condition, perturbation, rep)
    xbars =[]

    # Use as an anchoring timepoint the first timepoint that passed filtering
    if anchored == True:
        anchor_timepoint =timepoints[0]
        times = [fit for fit in x_bars.keys() if f'T{anchor_timepoint}' in fit[:2]]
        for t in times:
            xbars.append(x_bars[t])

        # insert 0 in the first index because mean fitness has not changed between timepoint 0 and 0 (it's the same timepoint duh)
        xbars.insert(0,0)
    elif anchored == False:
        for i in range(len(timepoints) - 1):
            xbars.append(x_bars[f'T{timepoints[i]}_T{timepoints[i+1]}'])
        xbars.insert(0,0)

    # return a list of mean fitness changes from 0 to timepoint t
    return xbars

#####################################
# infer fitness of a single barcode #
#####################################

# To make the grid search more efficient, we can create an s grid that is centered on a rough estimate of s
# We use a simple log linear fit to the data to get our best s guess (slope of line through log(frequencies)
# rs is read counts at each timepoint for a single barcode, Rs is total number of reads at each timepoint
def get_s_guess(rs,Rs,xbars, timepoints):
    x=np.arange(len(rs))

    # add pseudocounts so that we don't get -inf when taking a log(rs)
    rs = rs + 1

    # get frequencies of barcodes at each timepoint
    freqs = rs/Rs

    # "normalize" these frequencies to a gauge where the neutrals don't change in frequency over time
    normalized_freqs=freqs*np.exp(np.array(xbars)*np.array(timepoints))

    # draw a line of best fit through the log of these normalized frequencies and get slope
    coefficients = np.polyfit(x,np.log(normalized_freqs),1)
    slope = coefficients[0]

    return slope



# Generate grid over which to evaluate likelihood function using informed guesses for center of grid, return s_list and f_list 
def get_grid(bc, sample_bc_counts, xbars, timepoints):

    # rs is read counts of barcode at each timepoint, Rs is total number of reads at each timepoint
    rs = sample_bc_counts.loc[bc]
    Rs = sample_bc_counts.sum(axis = 0)

    # get the best s guess using log-linear fit to frequencies
    s_guess = get_s_guess(rs, Rs, xbars,timepoints)
    print(f'linear s_guess = {s_guess}', flush = True )

    # use initial frequnecy in data as our first guess for true initial frequency (f0)
    f0_guess = (rs[0]+1)/Rs[0]
    print(f'f0 guess = {f0_guess}', flush = True)

    # Create list of input s parameters by generating an evenly spaced list between 2 above and 2 below s 
    s_list = np.linspace((s_guess-2),(s_guess+2), 100)

    # # Check which value of s is closest to 0 and changes that value to 0 (to allow us to infer completely neutral fitness)
    # closest_idx, closest_val = min(enumerate(s_list), key=lambda x: abs(x[1]))
    # s_list[closest_idx] = 0 

    # generate bounds on the f list by scanning two orders of magnitude above and below the intial f_0 guess
    lower_exponent = int(np.log10(f0_guess)) - 2
    upper_exponent = int(np.log10(f0_guess)) + 2

    # generate f list by including 100 f0 values evenly spaced in log_10 space around our guess
    f_list = np.logspace(lower_exponent, upper_exponent, num=100, base=10)
    return (s_list,f_list)



# get approximation for a log of a gamma function (factorial) using Stirling's approx
def log_stirling_gamma(x):
    if x <= 0:
        print("trying to take log of negative # in likelihood function", flush = True)
        return None
    gamma_approx = log(math.sqrt(2*math.pi/x))+ x*log(x/math.e)
    return gamma_approx



# Given a numpy dataframe, return the maximum value of the grid and the indices of that maximum value 
def find_max_value_and_indices(df):
    max_val = df.values.max()
    max_indices = list(zip(*np.where(df.values == max_val)))
    return max_val, max_indices



# Compute the log likelihood using the pmf for a negative binomial reparameterized as mean (mu) and variance (c*mu)
# s and f0 are input parameters for mu (mean), and we will search for max likelihood values of these parameters with grid
def log_likelihood_function(s,f0,bc, sample_bc_counts,xbars,cs,timepoints):
    lh=0
    rs = sample_bc_counts.loc[bc]
    Rs = sample_bc_counts.sum(axis = 0)

    # if all rs are 0, return np.nan
    if all(rs == 0):
        # return nan
        return np.nan
         

    # split up log likelihood into different values for each timepoint
    for r,R,x_bar,c,t in zip(rs, Rs,xbars,cs, timepoints):

        # calculate each term in the likelihood function separately and add/subtract at the end
        mu = R * f0 *exp((s-x_bar)*t)
        t1 = log_stirling_gamma(r+(mu/(c-1)))
        t2=r*log(c-1 )
        t3=log_stirling_gamma(mu/(c-1))
        t4=log_stirling_gamma(r+1)
        t5 = (r+mu/(c-1))*log(c)

        # Add single timepoint's log likelihood to all previous timepoints' log likelihoods to get total log likelihood
        lh+=(t1+t2-t3-t4-t5)

    return lh 


# Calculate log likelihood at all points in the grid
def create_likelihood_grid(s_list, f_list, bc,sample_bc_counts,cs,xbars,timepoints):

    # Generate dataframe that will become the grid on which to search for maximum 
    df = pd.DataFrame()

    # Calculate ll for all s values and all f0 values
    for s in s_list:
        l=[]
        for f0 in f_list:
            lh = log_likelihood_function(s,f0,bc,sample_bc_counts,xbars,cs,timepoints)
            l.append(lh)

        # each column of grid corresponds to a different s value, and each row corresponds to a different f0 value 
        df[f's={s}']=l

    # Return grid 
    return df

# Find the maximum value of the log likelihood grid
def find_grid_max(df,s_list,f_list,bc,sample_bc_counts,cs,xbars,timepoints):

    # if df is all nan, return nan
    if df.isnull().values.all():
        return np.nan, np.nan
    else:

        # Get maximum value and indices of likelihood grid (pandas DataFrame)
        res = find_max_value_and_indices(df)

        # Get s and f0 separately 
        s = s_list[res[1][0][1]]
        f0 = f_list[res[1][0][0]]
        print(f'Fitness is {s}, frequency is {f0}', flush = True)

        # return max likelihood estimates for s and f0 
        return (s, f0)

# Put it all together! (we want to make sure we don't infer a max likelihood at the edge of the grid) 
def infer_max_likelihood_params(s_list, f_list, bc,sample_bc_counts,cs,xbars,timepoints):
    highest_absval_s = 4
    highest_f0 = 1
    lowest_f0 = 1e-9

    rs = sample_bc_counts.loc[bc]
    # if np.sum(rs) < 100:
    #     return np.nan, np.nan, np.nan, s_list, f_list


    # Calculate the initial likelihood grid
    likelihood_grid = create_likelihood_grid(s_list, f_list, bc,sample_bc_counts,cs,xbars,timepoints)
    
    if likelihood_grid.isnull().values.all():
        return likelihood_grid, np.nan, np.nan, s_list, f_list
    num_iterations = 0
    while True:
        # Find the maximum likelihood in the current grid
        max_likelihood_s, max_likelihood_f = find_grid_max(likelihood_grid,s_list,f_list,bc,sample_bc_counts,cs,xbars,timepoints)

        if max_likelihood_s < -highest_absval_s:
            print('max likelihood s lower than threshold', flush = True)
            return likelihood_grid, -10, max_likelihood_f, s_list, f_list
        
        if max_likelihood_s > highest_absval_s:
            print('max likelihood s higher than threshold', flush = True)
            return likelihood_grid, 10, max_likelihood_f, s_list, f_list


        if max_likelihood_f > highest_f0 or max_likelihood_f < lowest_f0:
            # return likelihood_grid, max_likelihood_s, max_likelihood_f with all nan values
            print('max likelihood s or f0 is outside threshold', flush = True)
            return likelihood_grid, max_likelihood_s, np.nan, s_list, f_list
        
        s_max = max(s_list)
        s_min = min(s_list)
        f_max = max(f_list)
        f_min = min(f_list)
        # Check if the inferred s value(s) are extremums of s_list
        s_extremum = False
        if max_likelihood_s == s_max or max_likelihood_s == s_min:
            s_extremum = True
            print(f'extremum s chosen', flush = True)
        
        # Check if the inferred f value(s) are extremums of f_list
        f_extremum = False
        if max_likelihood_f == f_max or max_likelihood_f == f_min:
            f_extremum = True
            print(f'extremum f chosen', flush = True)
        # If neither s nor f is an extremum, we're done
        if not s_extremum and not f_extremum:
            break

        # Generate the new s and/or f lists, centered on the extremum value(s)

        if s_extremum:
            s_list = np.concatenate((np.linspace(max_likelihood_s-2, max_likelihood_s+2, 99),np.array([0])))
        if f_extremum:
            lower_exponent = int(np.log10(max_likelihood_f)) - 2
            upper_exponent = int(np.log10(max_likelihood_f)) + 2
            f_list =  np.logspace(lower_exponent, upper_exponent, num=100, base=10)

        # Recalculate the likelihood grid using the new s and/or f lists
        likelihood_grid = create_likelihood_grid(s_list, f_list, bc,sample_bc_counts,cs,xbars,timepoints)
        num_iterations += 1
        if num_iterations > 10:
            # return likelihood_grid, max_likelihood_s, max_likelihood_f with all nan values
            return likelihood_grid, np.nan, np.nan, s_list, f_list
    return likelihood_grid, max_likelihood_s, max_likelihood_f, s_list, f_list

##################################
# Generate uncertainty estimates #
##################################

def quantiles(lh, s_list):
    likelihood_func = lh.values
    normalized_likelihood_func = likelihood_func / np.sum(likelihood_func)
    cdf = np.cumsum(normalized_likelihood_func)
    quantiles = [0.25, 0.50, 0.75]
    quantile_values = np.interp(quantiles, cdf, s_list)
# TODO Check this function! Should grid be s_list, and is it in the right spot for the parameters of this function? 


# Calculate the standard error of our fitness estimate using Fisher information
# input likelihood_func is the log likelihood function in one dimension (s) using only the row of grid corresponding to 
# max likelihood f0
def std_error(likelihood_func, s_list):
    parameter_values = s_list  

    # Need grid spacing to compute the gradient (should be constant spacing)
    grid_spacing = s_list[1]-s_list[0]

    # Get index and s value for maximum
    maximum_likelihood_idx = np.argmax(likelihood_func) 
    maximum_likelihood_parameter = parameter_values[maximum_likelihood_idx]

    # Compute Fisher informatoin by taking the negative second derivative of the log likelihood function
    observed_information = -np.gradient(np.gradient(likelihood_func,grid_spacing),grid_spacing)

    # We only care about the information at max
    observed_information_at_maximum_likelihood = observed_information[maximum_likelihood_idx]

    # Standard error is computed as the reciprocal of the square root of the information
    standard_error = 1.0 / np.sqrt(observed_information_at_maximum_likelihood)
    print("Maximum Likelihood Parameter Value: ", maximum_likelihood_parameter, flush = True)
    print("Standard Error: ", standard_error, flush = True)
    return standard_error


# get standard error from likelihood via numerical approx of second derivative of log-likelihood
def std_from_approx(s, max_ll, f0,bc, sample_bc_counts,xbars,cs,timepoints, h=1e-5):
    deriv2 = (-log_likelihood_function((s + 2*h), f0,bc, sample_bc_counts,xbars,cs,timepoints) 
        + 16*log_likelihood_function((s + h),f0,bc, sample_bc_counts,xbars,cs,timepoints) 
        - 30*max_ll[0]+ 16*log_likelihood_function((s - h),f0,bc, sample_bc_counts,xbars,cs,timepoints) 
        - log_likelihood_function((s - 2*h),f0,bc, sample_bc_counts,xbars,cs,timepoints))/(12*(h**2))
    std = 1/np.sqrt(-deriv2)
    print(f'input max likelihood s is {s}', flush = True )
    return std


