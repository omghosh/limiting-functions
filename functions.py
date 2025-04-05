## functions for bootcamp analysis
import pandas as pd 
import numpy as np



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

li_file = pd.read_csv('/Users/olivia/Desktop/PetrovLab/bigbatchbootcamp_git/Handpicked_Li2019_neutrals_rearray.csv', index_col=0)
li_neutrals = li_file['barcode'].values


mutant_colorset = {

                 'Diploid':'#fb9a99', # light red
                 'Diploid + Chr11Amp':'#e31a1c', # dark red for adaptive diploids
                 'Diploid + Chr12Amp':'#e31a1c',
                 'Diploid + IRA1':'#e31a1c',
                 'Diploid + IRA2':'#e31a1c',
                 'Diploid':'#e31a1c',
                 'Diploid + Chr11Amp':'#fb9a99', # light red for adaptive diploids
                 'Diploid + Chr12Amp':'#fb9a99',
                 'Diploid + IRA1':'#fb9a99',
                 'Diploid + IRA2':'#fb9a99',
                 'Diploid + Chr11Amp':'#a50f15', # dark red for high-fitness diploids (additional mutations/clearly higher fitness)
                 'Diploid + Chr12Amp':'#a50f15',
                 'Diploid + IRA1':'#a50f15',
                 'Diploid + IRA2':'#a50f15',
                 'Diploid_adaptive':'#a50f15',
                 'High-fitness Diploid':'#a50f15',


                 'GPB1':'#b2df8a',  # light green
                 'GPB2':'#33a02c',  # dark green
                 'IRA1':'#1f78b4', # dark blue
                 'IRA2':'#a6cee3', # light blue
                 'IRA1_nonsense':'#1f78b4', # dark blue
                    'IRA1_NON':'#1f78b4', # dark blue
                 'IRA1_missense':'#a6cee3', # dark blue
                    'IRA1_MIS':'#a6cee3', # dark blue
                 'IRA2':'gray', # light blue
                 'IRA1_other':'gray', # light blue
                 'NotSequenced':'gray',
                 'NotSequenced_adaptive':'gray',
                 'PDE2':'#ff7f00',  # dark orange
                 'CYR1':'#cab2d6', # light purple
                 'RAS2':'#b15928', # brown
                 'TFS1':'#fdbf6f', # light orange
                 'RAS2':'#b15928', # brown
                 'TFS1':'#b15928', # light orange
                 'PBS2':'#6a3d9a', # dark purple for HOG mutants
                 

                 'SSK2':'gray',
                 'SCH9':'#6a3d9a', # dark purple for TOR mutants
                 'TOR1':'salmon',
                 'KOG1':'turquoise', 
                    'PBS2':'cyan',
                    'GSH1':'blue', 
                    'BMH1':'green',
                    'RTG2':'purple',
                 'TOR/Sch9 Pathway':'#6a3d9a',
                 'other':'lightgray',
                 'other_adaptive':'darkgray',
                 "no_gene":'lightgray',
                 'ExpNeutral':'k'}

def define_batches_and_replicates():
    batches = {'Batch1':{
                        '1Day':['28','30','30Baffle','32','32Baffle'],
                        '2Day':['28','30','30Baffle','32','32Baffle'],
                        'Salt':['28','30','30Baffle','32','32Baffle']
                        },
            'Batch2':{
                        '1Day':['1.4','1.4Baffle','1.5','1.8','1.8Baffle'],
                        '2Day':['1.4','1.4Baffle','1.5','1.8','1.8Baffle'],
                        'Salt':['1.4','1.4Baffle','1.5','1.8','1.8Baffle']
                        },
            'Batch3':{
                        '1Day':['M3','NS','Suc','SucBaffle','Raf', 'RafBaffle'],
                        '2Day':['M3','NS','Suc','SucBaffle','Raf', 'RafBaffle'],
                        'Salt':['M3','NS','Suc','SucBaffle','Raf', 'RafBaffle']
                        },
            'Batch4':{
                        '1Day':['M3b4','4uMH89','10uMH89','10uMParomomycin','50uMParomomycin', '0.5%EtOH'],
                        '2Day':['M3b4','4uMH89','10uMH89','10uMParomomycin','50uMParomomycin', '0.5%EtOH'],
                        'Salt':['M3b4','4uMH89','10uMH89','10uMParomomycin','50uMParomomycin', '0.5%EtOH']
                        }
            }
    replicates = [1,2]
    return batches, replicates

def create_full_fitness_dataframe():
    batches, replicates = define_batches_and_replicates()

    bc_counts = pd.read_csv('/Users/olivia/Desktop/PetrovLab/bigbatchbootcamp_git/data/bc_counts.csv')
    fitness_df = pd.read_csv('/Users/olivia/Desktop/PetrovLab/bigbatchbootcamp_git/data/fitness_cleaned.csv', index_col=0)  
    # fitness_df = fitness_df.rename(columns={'Unnamed: 0': 'barcode'})
    # copy index over to new column called barcode
    fitness_df['barcode'] = fitness_df.index
    fitness_df.rename(columns = {'gene':'gene_old'}, inplace = True)
    bc_df=pd.read_csv('/Users/olivia/Desktop/PetrovLab/bigbatchbootcamp_git/data/Kinsler_et_al_2020_BCID_to_barcode_sequence.csv')

    grants_df = pd.read_csv('/Users/olivia/Desktop/PetrovLab/bigbatchbootcamp_git/data/Kinsler_et_al_2020_fitnessdata_noReplicates.csv')
    grants_df = grants_df.rename(columns={'barcode': 'BCID'})

    grants_df_with_barcode_df =pd.merge(grants_df, bc_df, on='BCID', how='inner')

    # This "full_df" has fitness info from both experiments, but only for barcodes that were present in both experiments
    full_df = pd.merge(grants_df_with_barcode_df, fitness_df, on='barcode', how='inner')
    # full_df = full_df.dropna()

    # rename all columns so that uM_ becomes uM 
    bc_counts.columns = [col.replace('uM_', 'uM') for col in bc_counts.columns]
    fitness_df.columns = [col.replace('uM_', 'uM') for col in fitness_df.columns]
    grants_df_with_barcode_df.columns = [col.replace('uM_', 'uM') for col in grants_df_with_barcode_df.columns]
    full_df.columns = [col.replace('uM_', 'uM') for col in full_df.columns]


    bc_counts.columns = [col.replace('Batch4_1Day_M3', 'Batch4_1Day_M3b4') for col in bc_counts.columns]
    fitness_df.columns = [col.replace('Batch4_1Day_M3', 'Batch4_1Day_M3b4') for col in fitness_df.columns]
    grants_df_with_barcode_df.columns = [col.replace('Batch4_1Day_M3', 'Batch4_1Day_M3b4') for col in grants_df_with_barcode_df.columns]
    full_df.columns = [col.replace('Batch4_1Day_M3', 'Batch4_1Day_M3b4') for col in full_df.columns]

    bc_counts.columns = [col.replace('Batch4_2Day_M3', 'Batch4_2Day_M3b4') for col in bc_counts.columns]
    fitness_df.columns = [col.replace('Batch4_2Day_M3', 'Batch4_2Day_M3b4') for col in fitness_df.columns]
    grants_df_with_barcode_df.columns = [col.replace('Batch4_2Day_M3', 'Batch4_2Day_M3b4') for col in grants_df_with_barcode_df.columns]
    full_df.columns = [col.replace('Batch4_2Day_M3', 'Batch4_2Day_M3b4') for col in full_df.columns]

    bc_counts.columns = [col.replace('Batch4_Salt_M3', 'Batch4_Salt_M3b4') for col in bc_counts.columns]
    fitness_df.columns = [col.replace('Batch4_Salt_M3', 'Batch4_Salt_M3b4') for col in fitness_df.columns]
    grants_df_with_barcode_df.columns = [col.replace('Batch4_Salt_M3', 'Batch4_Salt_M3b4') for col in grants_df_with_barcode_df.columns]
    full_df.columns = [col.replace('Batch4_Salt_M3', 'Batch4_Salt_M3b4') for col in full_df.columns]

    return bc_counts, fitness_df, grants_df_with_barcode_df, full_df


def generate_lists(full_df, batches, fitness_df):
    training_bcs = list(full_df[full_df['set']=='Train']['barcode'].values)
    testing_bcs = list(full_df[full_df['set']=='Test']['barcode'].values)


    conds = []

    for batch,conditions in batches.items():
        for home_condition,perturbations in conditions.items():
            for perturbation in perturbations:
                if f'{batch}_{home_condition}_{perturbation}_fitness' in fitness_df.columns:
                    conds.append(f'{batch}_{home_condition}_{perturbation}_fitness')
    # twoday_conds = [col for col in conds if '2Day' in col]
    # oneday_conds = [col for col in conds if '1Day' in col]
    # salt_conds = [col for col in conds if 'Salt' in col]
    # put them in order of perturbations in batches
    twoday_conds = []
    oneday_conds = []
    salt_conds = []
    for batch,conditions in batches.items():
        for home_condition,perturbations in conditions.items():
            for perturbation in perturbations:
                if f'{batch}_{home_condition}_{perturbation}_fitness' in fitness_df.columns:
                    if '2Day' in f'{batch}_{home_condition}_{perturbation}_fitness':
                        twoday_conds.append(f'{batch}_{home_condition}_{perturbation}_fitness')
                    elif '1Day' in f'{batch}_{home_condition}_{perturbation}_fitness':
                        oneday_conds.append(f'{batch}_{home_condition}_{perturbation}_fitness')
                    elif 'Salt' in f'{batch}_{home_condition}_{perturbation}_fitness':
                        salt_conds.append(f'{batch}_{home_condition}_{perturbation}_fitness')
    return training_bcs, testing_bcs, oneday_conds, twoday_conds, salt_conds 

def get_second_steps(anc_list, bc_counts):
    second_steps = {}
    for anc in anc_list:
        second_steps[anc]= (bc_counts[bc_counts['ancestor']==anc]['barcode'].values)
    return second_steps


def get_training_and_testing_bcs():
        # create pathway dictionary: 
    pathway_to_genes = {'RasPKA': ['IRA1', 'IRA2', 'GPB1', 'GPB2', 'PDE2', 'CYR1', 'GPR1', 'RAS2', 'TFS1'], 'TOR/Sch9':['TOR1', 'KOG1', 'SCH9', 'KSP1'], 
                        'HOG':['HOG1', 'PBS2', 'SSK2'], 'RTG':['MKS1', 'RTG2', 'BMH1'], 'TCA_cycle': ['CIT1', 'KGD1', 'MDH1', 'MAE1', 'ALD5'],
                        'Mito_bio':['PUF3', 'PAB1', 'PAN2', 'PAN3', 'AIM17'], 'Others': ['MKT1', 'GSH1', 'ARO80']}

    db = pd.read_csv('../data/fitness_withMutations.csv', index_col=0)
    ''
    # create new column that is both ancesotr and gene
    db['ancestor_gene'] = db['ancestor'] + '_' + db['gene']

    # create new column that is pathway
    db['pathway'] = 'other'
    for pathway, genes in pathway_to_genes.items():
        for gene in genes:
            db.loc[db['ancestor_gene']==gene, 'pathway'] = pathway

    train_bc_dict = {}
    test_bc_dict = {}
    for ancestor_gene_combo in db['ancestor_gene'].unique():
        training_bcs = []
        testing_bcs = []
        if db[db['ancestor_gene']==ancestor_gene_combo].shape[0] > 1:
            num_bcs = (db[db['ancestor_gene']==ancestor_gene_combo].shape[0])
            # take half and add to training up to 20 barcodes, then add the rest to testing
            training_bcs = list(db[db['ancestor_gene']==ancestor_gene_combo]['barcode'].values[:min(20,int(num_bcs/2))])
            testing_bcs = list(db[db['ancestor_gene']==ancestor_gene_combo]['barcode'].values[min(20,int(num_bcs/2)):])

        else:
            testing_bcs = list(db[db['ancestor_gene']==ancestor_gene_combo]['barcode'].values)

        train_bc_dict[ancestor_gene_combo] = training_bcs
        test_bc_dict[ancestor_gene_combo] = testing_bcs
        # within - pathway prediction: 
    # for each pathway, take half of the barcodes and add to training up to 20 barcodes, then add the rest to testing
    pathway_to_training_bcs = {}
    pathway_to_testing_bcs = {}
    for pathway, genes in pathway_to_genes.items():
        training_bcs = []
        testing_bcs = []
        for gene in genes:
            if db[db['gene']==gene].shape[0] > 1:
                num_bcs = (db[db['gene']==gene].shape[0])
                # take half and add to training up to 20 barcodes, then add the rest to testing
                training_bcs.extend(list(db[db['gene']==gene]['barcode'].values[:min(20,int(num_bcs/2))]))
                testing_bcs.extend(list(db[db['gene']==gene]['barcode'].values[min(20,int(num_bcs/2)):]))

            else:
                testing_bcs.extend(list(db[db['gene']==gene]['barcode'].values))
        pathway_to_training_bcs[pathway] = training_bcs
        pathway_to_testing_bcs[pathway] = testing_bcs

    # get all values in train_bc_dict and test_bc_dict
    all_training_bcs = []
    all_testing_bcs = []
    for key, value in train_bc_dict.items():
        all_training_bcs.extend(value)
    for key, value in test_bc_dict.items():
        all_testing_bcs.extend(value)
    
    return all_training_bcs, all_testing_bcs, train_bc_dict, test_bc_dict, pathway_to_training_bcs, pathway_to_testing_bcs



# Function to replace numerical values less than -4 with "extinct_fitness"
def replace_extinct(cell_value, extinct_fitness = -4):
    if isinstance(cell_value, (int, float)) and cell_value < extinct_fitness:
        return extinct_fitness
    return cell_value

# Apply the function to each cell of the DataFrame


def bicross_validation(train_mutants, test_mutants, train_envs, test_envs, fitness_df):

        # if there is any overlap between train and test, print how many are overlapping and then remove them
    overlap = [bc for bc in train_mutants if bc in test_mutants]
    # if len(overlap) != 0:
    #     print(f'Overlap between train and test: {len(overlap)}. Removed from training set!')
    train_mutants = [bc for bc in train_mutants if bc not in overlap]

    

    # train mutants and environments 
    train_df = fitness_df.loc[train_mutants, train_envs].values
    # test mutants and environments
    test_df = fitness_df.loc[test_mutants, test_envs].values
    #train mutants and test environments
    train_test_df = fitness_df.loc[train_mutants, test_envs].values
    # test mutants and train environments
    test_train_df = fitness_df.loc[test_mutants, train_envs].values

    # check that shapes are correct
    assert train_df.shape[0] == len(train_mutants)
    assert train_df.shape[1] == len(train_envs)
    assert test_df.shape[0] == len(test_mutants)
    assert test_df.shape[1] == len(test_envs)
    assert train_test_df.shape[0] == len(train_mutants)
    assert train_test_df.shape[1] == len(test_envs)
    assert test_train_df.shape[0] == len(test_mutants)
    assert test_train_df.shape[1] == len(train_envs)

    # if any of them are empty, return None
    if train_df.shape[0] == 0 or train_df.shape[1] == 0 or test_df.shape[0] == 0 or test_df.shape[1] == 0:
        return None, None, None

    truth = test_df

    max_rank = min([len(train_envs),len(train_mutants)])

    # first, do SVD on the training data
    u_train, s_train, v_train = np.linalg.svd(train_df, full_matrices=False)
    all_predictions = {}
    prediction_dfs = {}
    for rank in range(1,max_rank+1):

        new_s = np.asarray(list(s_train[:rank]) + list(np.zeros(s_train[rank:].shape)))
        S2 = np.zeros((u_train.shape[0],v_train.shape[0]))
        S2[:min([u_train.shape[0],v_train.shape[0]]),:min([u_train.shape[0],v_train.shape[0]])] = np.diag(new_s)

        D_hat = np.dot(u_train[:,:rank],np.dot(S2,v_train)[:rank,:])
        A_hat = np.dot(test_train_df,np.dot(np.linalg.pinv(D_hat),train_test_df)) # Eqn 3.2 from Owen and Perry 2009

        all_predictions[rank] = A_hat
        A_hat_df = pd.DataFrame(A_hat,columns=test_envs)
        A_hat_df['barcode'] = test_mutants
        # A_hat_df['truth'] = truth
        # set the index to be the barcode
        # A_hat_df = A_hat_df.set_index('barcode')
        prediction_dfs[rank] = A_hat_df


    return all_predictions, prediction_dfs, truth

def find_best_rank(all_predictions, truth, prediction_metric = 'mse'):
    best_rank = 1
    if prediction_metric == 'mse':
        mse = np.inf
        for k in all_predictions:
            prediction = all_predictions[k]
            if np.mean((prediction-truth)**2) < mse:
                best_rank = k
                mse = np.mean((prediction-truth)**2)
        return best_rank, mse
    elif prediction_metric == 'r2':
        r2 = 0
        for k in all_predictions:
            prediction = all_predictions[k]
            ss_res  = np.sum((truth - prediction) ** 2)
            ss_tot = np.sum((truth - np.mean(truth)) ** 2)
            curr_r2 = 1 - (ss_res / ss_tot)
            if curr_r2 > r2:
                best_rank = k
                r2 = curr_r2
        return best_rank, r2
    elif prediction_metric == 'scaled_sse':
        scaled_sse = np.inf
        for k in all_predictions:
            prediction = all_predictions[k]
            ss_res = np.sum((truth - prediction) ** 2)
            ss_tot = np.sum((truth - np.mean(truth)) ** 2)
            curr_scaled_sse = ss_res / ss_tot
            if curr_scaled_sse < scaled_sse:
                best_rank = k
                scaled_sse = curr_scaled_sse
        return best_rank, scaled_sse
    

 






bc_counts, fitness_df, grants_df_with_barcode_df, full_df = create_full_fitness_dataframe()
batches, replicates = define_batches_and_replicates()
original_training_bcs, original_testing_bcs, oneday_conds, twoday_conds, salt_conds = generate_lists(full_df,batches, fitness_df)

all_conds = oneday_conds + twoday_conds + salt_conds


environment_dict = {'2Day': twoday_conds, '1Day': oneday_conds, 'Salt': salt_conds}
### Create delta fitness matrix 

# create delta_fitnes_matrix 

def create_delta_fitness_matrix(batches, fitness_df, environment_dict):
        fitness_df = fitness_df.applymap(replace_extinct)
        perts=[]

        base_conditions = ['M3', '1.5' ,'30']

        for batch,conditions in batches.items():
            for home_condition,perturbations in conditions.items():
                for perturbation in perturbations:
                    if f'{batch}_{home_condition}_{perturbation}_fitness' in fitness_df.columns:
                        perts.append(perturbation)

        # get set of perts
        # Initialize dataframes for averages and errors
        perts = list(set(perts))
        home_averages = pd.DataFrame(index=fitness_df.index, columns=environment_dict.keys())
        home_errors = pd.DataFrame(index=fitness_df.index, columns=environment_dict.keys())



        for home in environment_dict.keys():
            # Get list of base conditions for this home
            base_conds = [cond for cond in environment_dict[home] if 'M3' in cond or '1.5_' in cond or '30_' in cond or 'M3b4' in cond]
            
            # Corresponding fitness and standard error columns
            fitness_cols = base_conds
            error_cols = [cond.replace('_fitness', '_stderror') for cond in base_conds]
            
            # Fitness values and standard errors
            fitness_values = fitness_df[fitness_cols]
            errors = fitness_df[error_cols]

            
            # Calculate weights (reciprocal of the squared errors)
            weights = 1 / (errors ** 2)

            weights_normalized = weights.div(weights.sum(axis=1), axis=0)
            print("Sample normalized weights:\n", weights_normalized)
            # multiply weights_normalized and fitness_values elementwise

            weighted_avg = (fitness_values.values * weights_normalized.values).sum(axis=1)

            # Standard error of the weighted average: 1 / sqrt(sum(1 / error^2))
            weighted_se = 1 / np.sqrt(weights.sum(axis=1))
            
            # Store results in the respective dataframes
            home_averages[home] = weighted_avg
            home_errors[home] = weighted_se

        # for each condition, in a new dataframe, subtract out the home average 
        perturbation_fitness_df =  pd.DataFrame(columns = fitness_df.columns, index = fitness_df.index)#fitness_df[twoday_conds+oneday_conds+salt_conds].copy()
        for env in twoday_conds:
            perturbation_fitness_df[env] = fitness_df[env] - home_averages['2Day']
            env_error_col = env.replace('_fitness', '_stderror')
            perturbation_fitness_df[env_error_col] = np.sqrt(fitness_df[env_error_col]**2 + home_errors['2Day']**2)

        for env in oneday_conds:
            perturbation_fitness_df[env] = fitness_df[env] - home_averages['1Day']
            env_error_col = env.replace('_fitness', '_stderror')
            perturbation_fitness_df[env_error_col] = np.sqrt(fitness_df[env_error_col]**2 + home_errors['1Day']**2)
        for env in salt_conds:
            perturbation_fitness_df[env] = fitness_df[env] - home_averages['Salt']
            env_error_col = env.replace('_fitness', '_stderror')
            perturbation_fitness_df[env_error_col] = np.sqrt(fitness_df[env_error_col]**2 + home_errors['Salt']**2)
                
        organized_perturbation_fitness_df = pd.DataFrame(index = perturbation_fitness_df.index)
        for pert in perts:
            cols = [col for col in perturbation_fitness_df.columns if f'{pert}_' in col]
            organized_perturbation_fitness_df[cols] = perturbation_fitness_df[cols]
        return organized_perturbation_fitness_df

def get_home_averages(batches, fitness_df, environment_dict):
        fitness_df = fitness_df.applymap(replace_extinct)
        perts=[]

        base_conditions = ['M3', '1.5' ,'30', 'M3b4']

        for batch,conditions in batches.items():
            for home_condition,perturbations in conditions.items():
                for perturbation in perturbations:
                    if f'{batch}_{home_condition}_{perturbation}_fitness' in fitness_df.columns:
                        perts.append(perturbation)

        # get set of perts
        # Initialize dataframes for averages and errors
        perts = list(set(perts))
        home_averages = pd.DataFrame(index=fitness_df.index, columns=environment_dict.keys())
        home_errors = pd.DataFrame(index=fitness_df.index, columns=environment_dict.keys())



        for home in environment_dict.keys():
            # Get list of base conditions for this home
            base_conds = [cond for cond in environment_dict[home] if 'M3' in cond or '1.5_' in cond or '30_' in cond or 'M3b4' in cond]
            
            # Corresponding fitness and standard error columns
            fitness_cols = base_conds
            error_cols = [cond.replace('_fitness', '_stderror') for cond in base_conds]
            
            # Fitness values and standard errors
            fitness_values = fitness_df[fitness_cols]
            errors = fitness_df[error_cols]

            
            # Calculate weights (reciprocal of the squared errors)
            weights = 1 / (errors ** 2)

            weights_normalized = weights.div(weights.sum(axis=1), axis=0)
            print("Sample normalized weights:\n", weights_normalized)
            # multiply weights_normalized and fitness_values elementwise

            weighted_avg = (fitness_values.values * weights_normalized.values).sum(axis=1)

            # Standard error of the weighted average: 1 / sqrt(sum(1 / error^2))
            weighted_se = 1 / np.sqrt(weights.sum(axis=1))
            
            # Store results in the respective dataframes
            home_averages[home] = weighted_avg
            home_errors[home] = weighted_se

        return home_averages, home_errors

# print(organized_perturbation_fitness_df.head()[['Batch4_2Day_0.5%EtOH_fitness']])
# print(fitness_df.head()['Batch4_2Day_0.5%EtOH_fitness'])


original_training_bcs = [bc for bc in original_training_bcs if bc in fitness_df.index]
original_testing_bcs = [bc for bc in original_testing_bcs if bc in fitness_df.index]


all_training_bcs, all_testing_bcs, train_bc_dict, test_bc_dict, pathway_to_training_bcs, pathway_to_testing_bcs = get_training_and_testing_bcs()

all_training_bcs = [bc for bc in all_training_bcs if bc in fitness_df.index]
all_testing_bcs = [bc for bc in all_testing_bcs if bc in fitness_df.index]

first_step_bcs = bc_counts[bc_counts['ancestor'] == 'WT']['barcode'].values
second_step_bcs = bc_counts[bc_counts['ancestor'] != 'WT']['barcode'].values

gpb2_anc_bcs= [bc for bc in bc_counts[bc_counts['ancestor'] == 'GPB2']['barcode'].values if bc in fitness_df.index]
tor1_anc_bcs = [bc for bc in bc_counts[bc_counts['ancestor'] == 'TOR1']['barcode'].values if bc in fitness_df.index]
cyr1_anc_bcs = [bc for bc in bc_counts[bc_counts['ancestor'] == 'CYR1']['barcode'].values if bc in fitness_df.index]
wt_anc_bcs = [bc for bc in bc_counts[bc_counts['ancestor'] == 'WT']['barcode'].values if bc in fitness_df.index]
li_wt_anc_bcs = [
    bc for bc in bc_counts[
        (bc_counts['ancestor'] == 'WT') & (bc_counts['evolution_condition'] == 'Evo1D')
    ]['barcode'].values if bc in fitness_df.index
]
ira1_mis_anc_bcs = [bc for bc in bc_counts[bc_counts['ancestor'] == 'IRA1_MIS']['barcode'].values if bc in fitness_df.index]
ira1_non_anc_bcs = [bc for bc in bc_counts[bc_counts['ancestor'] == 'IRA1_NON']['barcode'].values if bc in fitness_df.index]

evo1d_bcs = [bc for bc in bc_counts[bc_counts['evolution_condition'] == 'Evo1D']['barcode'].values if bc in fitness_df.index]
evo2d_bcs = [bc for bc in bc_counts[bc_counts['evolution_condition'] == 'Evo2D']['barcode'].values if bc in fitness_df.index]
evo3d_bcs = [bc for bc in bc_counts[bc_counts['evolution_condition'] == 'Evo3D']['barcode'].values if bc in fitness_df.index]


first_step_bcs = [bc for bc in first_step_bcs if bc in fitness_df.index]
second_step_bcs = [bc for bc in second_step_bcs if bc in fitness_df.index]


# for each class, get half the barcodes and add to training up to 20 barcodes, then add the rest to testing
cyr1_training_bcs = []
cyr1_testing_bcs = []
ira1mis_training_bcs = []
ira1mis_testing_bcs = []
ira1non_training_bcs = []
ira1non_testing_bcs = []
wt_training_bcs = []
wt_testing_bcs = []
liwt_training_bcs = []
liwt_testing_bcs = []
gpb2_training_bcs = []
gpb2_testing_bcs = []
tor1_training_bcs = []
tor1_testing_bcs = []

for class_name in bc_counts[bc_counts['barcode'].isin(cyr1_anc_bcs)]['class'].unique():
    # take slice of dataframe that is only the class
    # get CYR1 ancesotr dataframe 
    cyr1_df = bc_counts[bc_counts['ancestor']=='CYR1']
    class_df = cyr1_df[cyr1_df['class']==class_name]
    # get the barcodes
    class_bcs = class_df['barcode'].values
    # split bcs in half 
    if len(class_bcs) > 1:
        num_bcs = len(class_bcs)
        # take half and add to training up to 20 barcodes, then add the rest to testing
        cyr1_training_bcs.extend(list(class_bcs[:min(20,int(num_bcs/2))]))
        cyr1_testing_bcs.extend(list(class_bcs[min(20,int(num_bcs/2)):]))

    else:
        cyr1_testing_bcs.extend(list(class_bcs))

cyr1_training_bcs = [bc for bc in cyr1_training_bcs if bc in fitness_df.index]
cyr1_testing_bcs = [bc for bc in cyr1_testing_bcs if bc in fitness_df.index]

for class_name in bc_counts[bc_counts['barcode'].isin(ira1_mis_anc_bcs)]['class'].unique():
    # take slice of dataframe that is only the class
    ira1_mis_df = bc_counts[bc_counts['ancestor']=='IRA1_MIS']
    class_df = ira1_mis_df[ira1_mis_df['class']==class_name]
    # get the barcodes
    class_bcs = class_df['barcode'].values
    # split bcs in half 
    if len(class_bcs) > 1:
        num_bcs = len(class_bcs)
        # take half and add to training up to 20 barcodes, then add the rest to testing
        ira1mis_training_bcs.extend(list(class_bcs[:min(20,int(num_bcs/2))]))
        ira1mis_testing_bcs.extend(list(class_bcs[min(20,int(num_bcs/2)):]))

    else:
        ira1mis_testing_bcs.extend(list(class_bcs))

ira1mis_training_bcs = [bc for bc in ira1mis_training_bcs if bc in fitness_df.index]
ira1mis_testing_bcs = [bc for bc in ira1mis_testing_bcs if bc in fitness_df.index]

for class_name in bc_counts[bc_counts['barcode'].isin(ira1_non_anc_bcs)]['class'].unique():
    # take slice of dataframe that is only the class
    ira1_non_df = bc_counts[bc_counts['ancestor']=='IRA1_NON']
    class_df = ira1_non_df[ira1_non_df['class']==class_name]
    # get the barcodes
    class_bcs = class_df['barcode'].values
    # split bcs in half 
    if len(class_bcs) > 1:
        num_bcs = len(class_bcs)
        # take half and add to training up to 20 barcodes, then add the rest to testing
        ira1non_training_bcs.extend(list(class_bcs[:min(20,int(num_bcs/2))]))
        ira1non_testing_bcs.extend(list(class_bcs[min(20,int(num_bcs/2)):]))

    else:
        ira1non_testing_bcs.extend(list(class_bcs))

ira1non_training_bcs = [bc for bc in ira1non_training_bcs if bc in fitness_df.index]
ira1non_testing_bcs = [bc for bc in ira1non_testing_bcs if bc in fitness_df.index]

for class_name in bc_counts[bc_counts['barcode'].isin(wt_anc_bcs)]['class'].unique():
    # take slice of dataframe that is only the class
    wt_df = bc_counts[bc_counts['ancestor']=='WT']
    class_df = wt_df[wt_df['class']==class_name]
    # get the barcodes
    class_bcs = class_df['barcode'].values
    # split bcs in half 
    if len(class_bcs) > 1:
        num_bcs = len(class_bcs)
        # take half and add to training up to 20 barcodes, then add the rest to testing
        wt_training_bcs.extend(list(class_bcs[:min(20,int(num_bcs/2))]))
        wt_testing_bcs.extend(list(class_bcs[min(20,int(num_bcs/2)):]))

    else:
        wt_testing_bcs.extend(list(class_bcs))

wt_training_bcs = [bc for bc in wt_training_bcs if bc in fitness_df.index]
wt_testing_bcs = [bc for bc in wt_testing_bcs if bc in fitness_df.index]

for class_name in bc_counts[bc_counts['barcode'].isin(li_wt_anc_bcs)]['class'].unique():
    # take slice of dataframe that is only the class
    li_wt_df = bc_counts[bc_counts['ancestor']=='WT']
    class_df = li_wt_df[li_wt_df['class']==class_name]
    # get the barcodes
    class_bcs = class_df['barcode'].values
    # split bcs in half 
    if len(class_bcs) > 1:
        num_bcs = len(class_bcs)
        # take half and add to training up to 20 barcodes, then add the rest to testing
        liwt_training_bcs.extend(list(class_bcs[:min(20,int(num_bcs/2))]))
        liwt_testing_bcs.extend(list(class_bcs[min(20,int(num_bcs/2)):]))

    else:
        liwt_testing_bcs.extend(list(class_bcs))

liwt_training_bcs = [bc for bc in liwt_training_bcs if bc in fitness_df.index]
liwt_testing_bcs = [bc for bc in liwt_testing_bcs if bc in fitness_df.index]

for class_name in bc_counts[bc_counts['barcode'].isin(gpb2_anc_bcs)]['class'].unique():
    # take slice of dataframe that is only the class
    gpb2_df = bc_counts[bc_counts['ancestor']=='GPB2']
    class_df = gpb2_df[gpb2_df['class']==class_name]
    # get the barcodes
    class_bcs = class_df['barcode'].values
    # split bcs in half 
    if len(class_bcs) > 1:
        num_bcs = len(class_bcs)
        # take half and add to training up to 20 barcodes, then add the rest to testing
        gpb2_training_bcs.extend(list(class_bcs[:min(20,int(num_bcs/2))]))
        gpb2_testing_bcs.extend(list(class_bcs[min(20,int(num_bcs/2)):]))

    else:
        gpb2_testing_bcs.extend(list(class_bcs))

gpb2_training_bcs = [bc for bc in gpb2_training_bcs if bc in fitness_df.index]
gpb2_testing_bcs = [bc for bc in gpb2_testing_bcs if bc in fitness_df.index]

for class_name in bc_counts[bc_counts['barcode'].isin(tor1_anc_bcs)]['class'].unique():
    # take slice of dataframe that is only the class
    tor1_df = bc_counts[bc_counts['ancestor']=='TOR1']
    class_df = tor1_df[tor1_df['class']==class_name]
    # get the barcodes
    class_bcs = class_df['barcode'].values
    # split bcs in half 
    if len(class_bcs) > 1:
        num_bcs = len(class_bcs)
        # take half and add to training up to 20 barcodes, then add the rest to testing
        tor1_training_bcs.extend(list(class_bcs[:min(20,int(num_bcs/2))]))
        tor1_testing_bcs.extend(list(class_bcs[min(20,int(num_bcs/2)):]))

    else:
        tor1_testing_bcs.extend(list(class_bcs))
tor1_training_bcs = [bc for bc in tor1_training_bcs if bc in fitness_df.index]
tor1_testing_bcs = [bc for bc in tor1_testing_bcs if bc in fitness_df.index]






# only keep extinct guys in the test set?? 



mutant_dict = {'Original Training': original_training_bcs, 'Original Testing': original_testing_bcs, 
               'All Training': all_training_bcs, 'All Testing': all_testing_bcs, 
               'First Step': first_step_bcs, 'Second Step': second_step_bcs, 'anc: GPB2': gpb2_anc_bcs,
               'anc: TOR1': tor1_anc_bcs, 'anc: CYR1': cyr1_anc_bcs, 'anc: WT': wt_anc_bcs,
               'anc: Li_WT': li_wt_anc_bcs, 'anc: IRA1_MIS': ira1_mis_anc_bcs, 'anc: IRA1_NON': ira1_non_anc_bcs,
               'Evo1D': evo1d_bcs, 'Evo2D': evo2d_bcs, 'Evo3D': evo3d_bcs, 
               'CYR1 Training': cyr1_training_bcs, 'CYR1 Testing': cyr1_testing_bcs,
               'IRA1_MIS Training': ira1mis_training_bcs, 'IRA1_MIS Testing': ira1mis_testing_bcs,
               'IRA1_NON Training': ira1non_training_bcs, 'IRA1_NON Testing': ira1non_testing_bcs,
               'WT Training': wt_training_bcs, 'WT Testing': wt_testing_bcs,
               'Li_WT Training': liwt_training_bcs, 'Li_WT Testing': liwt_testing_bcs,
               'GPB2 Training': gpb2_training_bcs, 'GPB2 Testing': gpb2_testing_bcs,
               'TOR1 Training': tor1_training_bcs, 'TOR1 Testing': tor1_testing_bcs, 'original': original_training_bcs + original_testing_bcs
               }

# print(bc_counts[bc_counts['barcode'].isin(original_training_bcs)]['gene'].value_counts())
# print(bc_counts[bc_counts['barcode'].isin(original_testing_bcs)]['gene'].value_counts())

col_list = [col for col in grants_df_with_barcode_df.columns if 'fitness' in col]
subtle=[col for col in col_list if 'EC' in col]
subtle.append('1.4% Gluc_fitness')
subtle.append('12 hr Ferm_fitness')
subtle.append('1% Gly_fitness')
subtle.append('1.8% Gluc_fitness')
subtle.append('0.5% Raf_fitness')
subtle.append('8.5 μM GdA (B1)_fitness')
subtle.append('8 hr Ferm_fitness')
subtle.append('Baffle (B8)_fitness')
subtle.append('Baffle (B9)_fitness')
subtle.append('0.5% DMSO_fitness')
subtle.append('1% Raf_fitness')
subtle.append('Baffle, 1.7% Gluc_fitness')
subtle.append('Baffle, 1.6% Gluc_fitness')
subtle.append('18 hr Ferm_fitness')
subtle.append('Baffle, 1.4% Gluc_fitness')
subtle.append('2 μg/ml Flu_fitness')

strong = [col for col in grants_df_with_barcode_df.columns if ('fitness' in col) & (col not in subtle)]
subtle_errors = [entry.replace('fitness', 'error') for entry in subtle]



# gene dictionary 
# use grants_df_with_barcode_df to get the gene names and barcodes 
gene_dict = {}
for gene in grants_df_with_barcode_df['gene'].unique():
    gene_dict[gene] = list(grants_df_with_barcode_df[grants_df_with_barcode_df['gene']==gene]['barcode'].values)
    gene_dict[gene] = [bc for bc in gene_dict[gene] if bc in fitness_df.index]

ira1_non_list = grants_df_with_barcode_df[grants_df_with_barcode_df['mutation_type'] == 'IRA1_nonsense']['barcode']
ira1_non_list = ira1_non_list[ira1_non_list.isin(fitness_df.index)].values
ira1_non_list = [bc for bc in ira1_non_list if bc in fitness_df.index]

gene_dict['IRA1_nonsense'] = ira1_non_list

ira1_mis_list = grants_df_with_barcode_df[grants_df_with_barcode_df['mutation_type'] == 'IRA1_missense']['barcode']
ira1_mis_list = ira1_mis_list[ira1_mis_list.isin(fitness_df.index)].values
ira1_mis_list = [bc for bc in ira1_mis_list if bc in fitness_df.index]

gene_dict['IRA1_missense'] = ira1_mis_list



rebarcoding_source_mutants = {'IRA1_MIS':'CGCTAAAGACATAATGTGGTTTGTTG_CTTCCAACAAAAAATCATTTTTATAC', # BCID 43361 from venkataram 2016
'IRA1_NON':'CGCTAAAGACATAATGTGGTTTGTTG_AGAGTAATCTGCAAGATTCTTTTTCT', # BCID 21967 from venkataram 2016
'CYR1':    'CGCTAAAGACATAATGTGGTTTGTTG_CTCGAAACAGGAAAAGCACTTATCGA', # BCID 43692 from venkataram 2016
'TOR1':    'CGCTAAAGACATAATGTGGTTTGTTG_TAGACAAAATGCAATTGTATTGTCAG' , # BCID 21543 from venkataram 2016
'GPB2':    'CGCTAAAGACATAATGTGGTTTGTTG_TCATGAACGGATAAGCTGGTTGGTTG' } # BCID 7774 from venkataram 2016

ancestral_mutations = {'IRA1_NON':'II:522427:A:T:IRA1:stop_gained:c.4202T>A:p.Leu1401*',
                      'IRA1_MIS':'II:522697:G:A:IRA1:missense_variant:c.3932C>T:p.Ala1311Val',
                      'CYR1':'X:427906:C:A:CYR1:missense_variant:c.2750C>A:p.Ser917Tyr',
                      'GPB2':'I:40104:T:G:GPB2:stop_gained:c.846T>G:p.Tyr282*',
                      'TOR1':'X:564551:T:G:TOR1:missense_variant:c.5136T>G:p.Phe1712Leu'}

env_color_dict = {'2Day': (0.46,0.74,1), '1Day': (0.4,0.62,0.37), 'Salt': (0.96,0.45,0.42 )}



