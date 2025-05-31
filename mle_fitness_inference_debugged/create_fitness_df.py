# ## create full fitness dataframe: 
import pandas as pd
import os
import re
import numpy as np
from matplotlib import pyplot as plt

# Set the directory containing the CSV files
directory ="/scratch/groups/dpetrov/BigBatchBootcamp/mle_fitness_inference_debugged/results"

# Define the regular expression patterns to match column names
pattern_fitness = re.compile('.*_fitness')
pattern_stderr = re.compile('.*_stderror')

# Read all CSV files into a list of data frames
csv_files = [pd.read_csv(os.path.join(directory, f), index_col=0) for f in os.listdir(directory) if f.endswith('.csv')]

# Merge the data frames into one based on the columns that match the regular expression patterns
merged_fitness = pd.concat([df.filter(regex=pattern_fitness) for df in csv_files], axis=1)
merged_stderr = pd.concat([df.filter(regex=pattern_stderr) for df in csv_files], axis=1)

# Merge the two data frames into one based on the index
merged_df = pd.concat([merged_fitness, merged_stderr], axis=1)

# sort columns alphabetically
merged_df = merged_df.reindex(sorted(merged_df.columns), axis=1)

# print(merged_df)
# merged_df.to_csv('/Users/olivia/Desktop/PetrovLab/natty/mle_fitness_df.csv')

# Average together replicates and calculate the standard error of the mean
# find sets of columns that are the same except for the replicate number
# for each set of columns, average the fitness values and calculate the standard error of the mean
# add the averaged fitness values and standard errors to the data frame



# bc_counts = pd.read_csv('/Users/olivia/Desktop/PetrovLab/natty/natty_mle_fitness_inference/barcode_counts.csv', index_col=0)
# print(bc_counts.columns)

# conds will be list of column names in bc_counts that have _R in them
conds = [col for col in merged_df.columns if '_R' in col]

# split string at _R and take first element
conds = [col.split('_R')[0] for col in conds]
# remove duplicates
conds = list(set(conds))
print(conds)
replicates = np.arange(1,6)
for cond in conds:
    fitness_dict={}
    error_dict={}
    fitnesses_to_average=[]
    errors_to_average=[]
    for rep in replicates:
        if f'{cond}_R{rep}_fitness' in merged_df.columns:
            fitnesses_to_average.append(f'{cond}_R{rep}_fitness')
            errors_to_average.append(f'{cond}_R{rep}_stderror')
    for bc in merged_df.index:
        if len(fitnesses_to_average)>0:
            fitnesses = merged_df[fitnesses_to_average].loc[bc]
            errors = merged_df[errors_to_average].loc[bc]
            # choose only non-nan values
            fitnesses = fitnesses[~np.isnan(fitnesses)].values
            errors = errors[~np.isnan(errors)].values
            # if there are any non-nan values, average them and calculate the standard error of the mean
            if len(fitnesses)>0:
                weights = 1/(errors**2)
                fitness_dict[bc] = (fitnesses*weights).sum()/weights.sum()
                error_dict[bc] = 1/np.sqrt(weights.sum())
            else:
                fitness_dict[bc] = np.nan
                error_dict[bc] = np.nan
    merged_df[f'{cond}_fitness'] = pd.Series(fitness_dict)
    merged_df[f'{cond}_stderror'] = pd.Series(error_dict)



# print(merged_df['SC_fitness'])



# 			if (f'{batch}_{home_condition}_{perturbation}-R1_fitness') in merged_df.columns:
# 				if (f'{batch}_{home_condition}_{perturbation}-R2_fitness') in merged_df.columns:
# 					fitness1 = merged_df[(f'{batch}_{home_condition}_{perturbation}-R1_fitness')]
# 					fitness2 = merged_df[(f'{batch}_{home_condition}_{perturbation}-R2_fitness')]

# 					error1 = merged_df[(f'{batch}_{home_condition}_{perturbation}-R1_stderror')]
# 					error2 = merged_df[(f'{batch}_{home_condition}_{perturbation}-R2_stderror')]

# 					weights1 = 1/(error1**2)
# 					weights2 = 1/(error2**2)

# 					merged_df[f'{batch}_{home_condition}_{perturbation}_fitness'] = (fitness1*weights1+fitness2*weights2)/(weights1+weights2)

# 					merged_df[f'{batch}_{home_condition}_{perturbation}_stderror'] =1/np.sqrt(weights2+weights1)


# # bc_counts = pd.read_csv('fitness_inferences/bc_counts.csv')
# # bc_counts = bc_counts.rename(columns={'Unnamed: 0': 'barcode'})
# # bc_counts = bc_counts.set_index('barcode')

# li_neutrals = ['GTAATAAGAAGTAAGATGATTCTATT_GCTCGAATCACGAATATTTTTTTGTC',
# 'GTAATAAGAAGTAAGATGATTCTATT_TATGCAAGTGCAAATGTTTTATCTC',
# 'GTAATAAGAAGTAAGATGATTCTATT_TGGATAACTCGTAAGGACCTTGTCAA',
# 'GTAATAAGAAGTAAGATGATTCTATT_TGCATAACCAATAATTAGATTGTAGA',
# 'GTAATAAGAAGTAAGATGATTCTATT_ATTGTAACTAAGAATACCGTTCGCAG',
# 'GTAATAAGAAGTAAGATGATTCTATT_CCTTTAAACCAAAACACAGTTCGACA',
# 'GTAATAAGAAGTAAGATGATTCTATT_GAAGAAATAGTAAAGGCAATTCCAAA',
# 'GTAATAAGAAGTAAGATGATTCTATT_AACACAAATGCGAATGAGTTTAATAG',
# 'GTAATAAGAAGTAAGATGATTCTATT_AAGCCAAGTTCGAAAATGATTTAGGG',
# 'GTAATAAGAAGTAAGATGATTCTATT_AGACAAATCCAGAACATCCTTAGTTT',
# 'GTAATAAGAAGTAAGATGATTCTATT_TACAAAACTATAAACTTGATTAATCA',
# 'GTAATAAGAAGTAAGATGATTCTATT_TGACTAAGAGTTAATTCAGTTTATAT',
# 'GTAATAAGAAGTAAGATGATTCTATT_TGCTCAACTGGCAAAAAATTTCGAAC',
# 'GTAATAAGAAGTAAGATGATTCTATT_TTGAGAAATACGAATACACTTGTAAA',
# 'GTAATAAGAAGTAAGATGATTCTATT_CAAAGAAAAGTTAATACCGTTCGGCG',
# 'GTAATAAGAAGTAAGATGATTCTATT_TACTCAATGTGTAAAGAGCTTCCTTT',
# 'GTAATAAGAAGTAAGATGATTCTATT_TATTGAATCAGTAAGATGTTTGATTA',
# 'CCCCCAATCCTCAACCCGCTTCGTAC_TTGTTAAAACCGAACTCAATTTTTAT',
# 'GTAATAAGAAGTAAGATGATTCTATT_GCTGTAACACAGAAGGTAGTTTTCCT',
# 'GTAATAAGAAGTAAGATGATTCTATT_TTTTTAAACAGTAATGCCATTAGCAG',
# 'GTAATAAGAAGTAAGATGATTCTATT_CGTCCAATCGTTAATAGACTTTGCTT',
# 'GTAATAAGAAGTAAGATGATTCTATT_CCCACAATACTTAAGGGCATTGTCAT',
# 'GTAATAAGAAGTAAGATGATTCTATT_ATGTAAAGGCGAAATCTAATTTACTG',
# 'GTAATAAGAAGTAAGATGATTCTATT_AAGGTAAACCATAAATCGGTTGAATT',
# 'GTAATAAGAAGTAAGATGATTCTATT_ACCCCAAAAGTCAAAGGGCTTTACAA',
# 'GTAATAAGAAGTAAGATGATTCTATT_CATCGAATATTCAAGTATGTTTTCTA',
# 'CCCCCAATCCTCAACCCGCTTCGTAC_TTCAGAACAGACAAGAATTTTTTATG',
# 'GTAATAAGAAGTAAGATGATTCTATT_TCACTAAGACCTAAGGTGATTGTTTC',
# 'GTAATAAGAAGTAAGATGATTCTATT_GTCTAAACGTTTAAAATAATTACGTA',
# 'GTAATAAGAAGTAAGATGATTCTATT_TCATGAACGGATAAGCTGGTTGGTTG',
# 'GTAATAAGAAGTAAGATGATTCTATT_CCGGGAATCACGAATGTGTTTTTAAC',
# 'GTAATAAGAAGTAAGATGATTCTATT_ATGTCAAGTTTGAACGATATTACTAG',
# 'GTAATAAGAAGTAAGATGATTCTATT_ATGGAAAAGATTAATGTGCTTGTCCA',
# 'CGTATAAAGCGCAACACTGTTGATCG_AATCTAAACCTGAAAAGAATTTGGTA',
# 'GTAATAAGAAGTAAGATGATTCTATT_ACTAGAAAACTGTTATGAA',
# 'GTAATAAGAAGTAAGATGATTCTATT_TATAAAACAGTTAACAGCGTTTGCTT',
# 'GTAATAAGAAGTAAGATGATTCTATT_TCAGCAACCAGCAAAATTATTATCTC',
# 'GTAATAAGAAGTAAGATGATTCTATT_CCCCCAATTTTCAAGTATATTATGTC',
# 'GTAATAAGAAGTAAGATGATTCTATT_TTCGCAAAGCAGAACGCTATTAAAAA',
# 'GTAATAAGAAGTAAGATGATTCTATT_GATCCAATTATTAAATGATTTGCCGG',
# 'GTAATAAGAAGTAAGATGATTCTATT_TATACAATACTCAACTCTGTTTGTAA',
# 'GTAATAAGAAGTAAGATGATTCTATT_CAAGGAAACGTAAACTTACTTTTCGA',
# 'GTAATAAGAAGTAAGATGATTCTATT_ATCTGAATTCACAATGATCTTTACTT',
# 'GTAATAAGAAGTAAGATGATTCTATT_GTTTAAATATGGAAATTTATTCTAAC',
# 'GTAATAAGAAGTAAGATGATTCTATT_ATAACAAACATCAACTGAGTTATATG',
# 'GTAATAAGAAGTAAGATGATTCTATT_TCATCAACAGCTAACCGATTTGCGTT',
# 'GTAATAAGAAGTAAGATGATTCTATT_GGCAAAAAGACGAAATGGCTTTGATA',
# 'GTAATAAGAAGTAAGATGATTCTATT_CCCAAAAACATTAAAGAGATTCGTTG',
# 'GTAATAAGAAGTAAGATGATTCTATT_TCAAGAATGCTCAAATGCATTTTCTT',
# 'GTAATAAGAAGTAAGATGATTCTATT_ACTGTAATGTGAAAGGCATTTTTTGC',
# 'GTAATAAGAAGTAAGATGATTCTATT_TTAGGAAATTCTAATGCGATTGAATG',
# 'GTAATAAGAAGTAAGATGATTCTATT_ATTAGAATTTATAAGTAGGTTGTATT',
# 'GTAATAAGAAGTAAGATGATTCTATT_CCCCGAACGTGAAATGTTGTTATATA',
# 'GTAATAAGAAGTAAGATGATTCTATT_TGTTCAAAAGTGAAATCTGTTTTGTA',
# 'GTAATAAGAAGTAAGATGATTCTATT_GTGGGAAGATGCAAGTCCATTTACTC',
# 'GTAATAAGAAGTAAGATGATTCTATT_TGGCTAACAGGAAACGAGGTTGGGCG',
# 'GTAATAAGAAGTAAGATGATTCTATT_GAAAAAAGAGGTAACCTCCTTTAGGA',
# 'GTAATAAGAAGTAAGATGATTCTATT_TCTAGAATTAGCAAGTACATTGCCGT',
# 'GTAATAAGAAGTAAGATGATTCTATT_ACTAGAAGTGAGAAGACTCTTGGATT',
# 'GTAATAAGAAGTAAGATGATTCTATT_TATTGAAGCGTAAATCTACTTAGTAA',
# 'GTAATAAGAAGTAAGATGATTCTATT_CCTACAAGTCCAAACTCTTTTCAGTT',
# 'GTAATAAGAAGTAAGATGATTCTATT_AAACAAACATCCAAATATCTTTGTGT'	]			

# # all_li_barcodes = np.array(bc_counts[bc_counts['source_publication']=='Li2019'].index)

# # Add in a separate fitness and error column for all barcodes where we measure the fitness relative to the Li ancestor 

# for batch,conditions in batches.items():
# 	for home_condition,perturbations in conditions.items():
# 		for perturbation in perturbations:
# 			if (f'{batch}_{home_condition}_{perturbation}_fitness') in merged_df.columns:
# 				if (f'{batch}_{home_condition}_{perturbation}_stderror') in merged_df.columns:
# 					fitness_values = merged_df[(f'{batch}_{home_condition}_{perturbation}_fitness')][li_neutrals]
# 					standard_errors = merged_df[(f'{batch}_{home_condition}_{perturbation}_stderror')][li_neutrals]

# 					weights = 1 / standard_errors  # Calculate the weights by taking the reciprocal of the standard errors
# 					weighted_fitness = fitness_values * weights  # Multiply fitness values by weights
# 					weighted_average = weighted_fitness.sum() / weights.sum()  # Calculate the weighted average
# 					new_standard_error = 1 / np.sqrt(weights.sum())

# 					print(f'{batch}_{home_condition}_{perturbation}')
# 					# print("Weighted Average Fitness:", weighted_average)
# 					# print("New Standard Error:", new_standard_error)
# 					merged_df[f'{batch}_{home_condition}_{perturbation}_fitness_Li'] =  merged_df[(f'{batch}_{home_condition}_{perturbation}_fitness')] - weighted_average
# 					merged_df[f'{batch}_{home_condition}_{perturbation}_stderror_Li'] = np.sqrt(merged_df[(f'{batch}_{home_condition}_{perturbation}_stderror')]**2 + new_standard_error**2) 
# 					# print(merged_df[f'{batch}_{home_condition}_{perturbation}_fitness_Li'][all_li_barcodes])
# 					# print(merged_df[f'{batch}_{home_condition}_{perturbation}_fitness'][all_li_barcodes])


# #Print the resulting data frame
merged_df.to_csv('fitness.csv')





