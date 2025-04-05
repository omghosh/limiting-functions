import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
# import numpy as np 
# import pandas as pd
from functions import *

bc_counts, fitness_df, grants_df_with_barcode_df, full_df = create_full_fitness_dataframe()


fitness_df = fitness_df.applymap(lambda x: replace_extinct(x, extinct_fitness=-2))

    # make columns in particular order
    # use this order of perturbations 
    # ['30', '1.5', 'M3', 'M3b4', '50uMParomomycin', '10uMH89', 'Raf', '4uMH89', 'RafBaffle', '1.4', '1.4Baffle', '10uMParomomycin', '1.8', '30Baffle', 'NS', '28', '1.8Baffle', '0.5%EtOH', 'SucBaffle', '32', 'Suc', '32Baffle']
sorted_list_of_perts = ['30', '1.5', 'M3', 'M3b4', '50uMParomomycin', '10uMH89', 'Raf', '4uMH89',
                        'RafBaffle', '1.4', '1.4Baffle', '10uMParomomycin', '1.8', '30Baffle', 'NS', 
                        '28', '1.8Baffle', '0.5%EtOH', 'SucBaffle', '32', 'Suc', '32Baffle']
#make empty list of length of twoday_conds
sorted_twoday = [None]*len(twoday_conds)
sorted_oneday = [None]*len(oneday_conds)
sorted_salt =[None]*len(salt_conds)
for cond in twoday_conds:
    pert = cond.split('_')[2]
    # i want to add cond to sorted all conds if pert is in sorted_list_of_perts
    if pert in sorted_list_of_perts:
        # get index of pert in sorted_list_of_perts
        idx = sorted_list_of_perts.index(pert)
        sorted_twoday.insert(idx, cond)
for cond in oneday_conds:
    pert = cond.split('_')[2]
    if pert in sorted_list_of_perts:
        idx = sorted_list_of_perts.index(pert)
        sorted_oneday.insert(idx, cond)
for cond in salt_conds:
    pert = cond.split('_')[2]
    if pert in sorted_list_of_perts:
        idx = sorted_list_of_perts.index(pert)
        sorted_salt.insert(idx, cond)
sorted_list_of_perts = sorted_twoday + sorted_oneday + sorted_salt
# get rid of None values
sorted_list_of_perts = [x for x in sorted_list_of_perts if x is not None]
sorted_twoday = [x for x in sorted_twoday if x is not None]
sorted_oneday = [x for x in sorted_oneday if x is not None]
sorted_salt = [x for x in sorted_salt if x is not None]

pert_label_mapping = {'32Baffle': '32°C, Baffle', 'M3b4': 'Batch 4 Base', 'M3': 'Batch 3 Base','1.5': 'Batch 2 Base', '30': 'Batch 1 Base', '50uMParomomycin': '50uM Paro', '10uMH89': '10uM H89',
                      'Raf':'Raffinose','4uMH89': '4uM H89' ,'RafBaffle':'Raffinose, Baffle', '1.4':'1.4% Glucose', '1.4Baffle':'1.4% Glucose, Baffle','10uMParomomycin': '10uM Paro' ,'1.8':'1.8% Glucose',
                      '0.5%EtOH': '0.5% Ethanol', 'SucBaffle':'Sucrose, Baffle', '32': '32°C', 'Suc':'Sucrose', '28': '28°C', 'NS':'No Shake', '30Baffle':'30°C, Baffle', '1.8Baffle':'1.8% Glucose, Baffle'}


fitness_df = fitness_df[sorted_list_of_perts]
print(fitness_df.max().max())

environment_dict = {'2Day': sorted_twoday, '1Day': sorted_oneday, 'Salt': sorted_salt}

np.random.seed(1)
def prepare_data(fitness_df, bc_counts, subset_size=5, subset_heatmap=True):
    """
    Prepare and process the data for visualization, returning separate dataframes for each ancestor background.
    """
    anc_list = ['WT', 'IRA1_MIS', 'IRA1_NON', 'CYR1', 'TOR1', 'GPB2'] #bc_counts['ancestor'].unique()[::-1]

    ancestor_dfs = {}  # Dictionary to store DataFrames for each ancestor
    
    for ancestor in anc_list:
        if ancestor == 'WT':
            continue
        ancestor_df = pd.DataFrame(columns=list(fitness_df.columns) + ['ancestor', 'evolution_condition'])

        for evolution_condition in ['Evo1D', 'Evo2D']:
            these_muts = bc_counts[
                (bc_counts['ancestor'] == ancestor) & 
                (bc_counts['evolution_condition'] == evolution_condition)
            ]['barcode']
            
            if len(these_muts) == 0:
                continue
                
            these_muts = [mut for mut in these_muts if mut in fitness_df.index]
            if len(these_muts) == 0:
                continue
            print(evolution_condition, ancestor)
      
            bcs_with_genes = bc_counts[bc_counts['barcode'].isin(these_muts)].dropna(subset=['gene'])
            if len(bcs_with_genes) == 0:

                these_muts = np.random.choice(these_muts, subset_size, replace=False)
                this_df = fitness_df.loc[these_muts]
                this_df['ancestor'] = ancestor
                this_df['evolution_condition'] = evolution_condition
                ancestor_df = pd.concat([ancestor_df, this_df], ignore_index=True)
            else:
                these_muts = bcs_with_genes['barcode'].values
                these_muts = np.random.choice(these_muts, subset_size, replace=False)
                this_df = fitness_df.loc[these_muts]
                this_df['ancestor'] = ancestor
                this_df['evolution_condition'] = evolution_condition
                ancestor_df = pd.concat([ancestor_df, this_df], ignore_index=True)
                print(bc_counts[bc_counts['barcode'].isin(these_muts)]['gene'])
        
        if not ancestor_df.empty:
            ancestor_dfs[ancestor] = ancestor_df

    
    return ancestor_dfs, anc_list

def prepare_wt_data(fitness_df, bc_counts, rebarcoding_source_mutants):
    """
    Prepare wild-type data for visualization
    """
    wt_barcode = np.random.choice(bc_counts[
        (bc_counts['ancestor'] == 'WT') & 
        (bc_counts['class'] == 'neutral_haploids') & 
        (bc_counts['evolution_condition'] == 'Evo2D')
    ]['barcode'])
    
    rebarcoding_source_mutants['WT'] = wt_barcode
    rebarcoding_source_mutants = dict([('WT', rebarcoding_source_mutants['WT'])] + 
                                    [(k, v) for k, v in rebarcoding_source_mutants.items() if k != 'WT'])
    
    wt_barcodes = list(rebarcoding_source_mutants.values())
    wt_df = fitness_df.loc[wt_barcodes].copy()
    wt_df['Gene'] = [gene for gene, barcode in rebarcoding_source_mutants.items() if barcode in wt_df.index]
    wt_df.reset_index(drop=True, inplace=True)

    # Identify the index of WT row
    wt_index = wt_df.index[wt_df['Gene'] == 'WT'][0]  # Assuming 'WT' is labeled in the 'Gene' column


    # Create a blank row with NaN values
    blank_row = pd.DataFrame([[np.nan] * len(wt_df.columns)], columns=wt_df.columns)
    blank_row['Gene'] = " "  # Empty label for spacing

    # Insert n  blank rows immediately after the WT row
    for n in range(2):
        wt_df = pd.concat([wt_df.iloc[:wt_index + 1+n], blank_row, wt_df.iloc[wt_index + n +1:]], ignore_index=True)
    # wt_df = pd.concat([wt_df.iloc[:wt_index + 1], blank_row, wt_df.iloc[wt_index + 1:]], ignore_index=True)

    # add a blank row at the end
    for i in range(2):
        wt_df = pd.concat([wt_df, blank_row], ignore_index=True)
    return wt_df

def create_color_schemes(anc_list, non_wt_df):
    """
    Create color schemes for different components
    """
    # ancestor_colors = {ancestor: color for ancestor, color in 
    #                   zip(non_wt_df['ancestor'].unique(), 
    #                       sns.color_palette("viridis", len(non_wt_df['ancestor'].unique())))}
    # print(ancestor_colors)

    ancestor_colors =          {'GPB2':(0.2, 0.63, 0.17), ##33a02c',  # dark green
                    'IRA1_NON': (0.12,0.47,0.71), #'#1f78b4', # dark blue
                    'IRA1_MIS': (0.65,0.81, 0.89), #'#a6cee3', # dark blue
                 'CYR1': (0.79, 0.7,0.84 ), # '#cab2d6', # light purple
                    'TOR1':(0.85,0.53,0.75 )}
    
    gene_colors = {'WT': (0.5, 0.5, 0.5), ' ': (1, 1, 1)}


    gene_colors.update(ancestor_colors)
    
    return ancestor_colors, gene_colors

def plot_heatmap(wt_df, ancestor_dfs, environment_dict, ancestor_colors, gene_colors, 
                 vmin, vmax, num_conditions=3, figsize=(16, 14)):
    """
    Create separate heatmap subplots for each ancestor background.
    """
    num_ancestors = len(ancestor_dfs)


    fig, axs = plt.subplots(
        num_ancestors + 1, num_conditions + 1, figsize=figsize,
        gridspec_kw={
            'height_ratios': [len(wt_df)] + [len(df) for df in ancestor_dfs.values()],
            'width_ratios': [0.05] + [1] * num_conditions,
            'hspace': 0.04,
            'wspace': 0.21
        }
    )

    # Plot WT heatmaps
    for i, (base_env, env_cols) in enumerate(environment_dict.items()):
        if i >= num_conditions:
            break
            
        ax = axs[0, i + 1]
        sns.heatmap(
            wt_df[env_cols], cmap='mako',
            ax=ax, center=0, vmin=vmin, vmax=vmax,
            cbar=False
        )
        ax.set_title(f'{base_env} base')
        ax.set_xticks([])
        # if i == 0:
        #     ax.set_yticks(np.arange(len(wt_df)))
        #     ax.set_yticklabels(wt_df['Gene'])
        # else:
        ax.set_yticks([])
        ax.set_yticklabels([])

    # WT ancestor color bar
    wt_ancestor_ax = axs[0, 0]
    wt_ancestor_ax.set_xticks([])
    wt_ancestor_ax.set_yticks(np.arange(len(wt_df)))
    wt_ancestor_ax.set_yticklabels(wt_df['Gene'])
    # wt_ancestor_ax.set_yticks([])
    # wt_ancestor_ax.set_ylabel("First Step Mutants")
    # wt_ancestor_ax.imshow(
    #     [[gene_colors[gene]] for gene in wt_df['Gene'].values],
    #     aspect='auto', interpolation='nearest'
    # )
    wt_ancestor_ax.imshow([[gene_colors[gene]] for gene in wt_df['Gene'].values],
    aspect='auto', interpolation='nearest', cmap=ListedColormap(list(gene_colors.values())))
    wt_ancestor_ax.spines[:].set_visible(False)  # Remove axis outline
    wt_ancestor_ax.set_frame_on(False)  # Remove any remaining frame

    

    # Plot each ancestor background separately
    for row_idx, (ancestor, df) in enumerate(ancestor_dfs.items(), start=1):
        for i, (base_env, env_cols) in enumerate(environment_dict.items()):
            if i >= num_conditions:
                break
                
            ax = axs[row_idx, i + 1]
            sns.heatmap(
                df[env_cols], cmap='mako',
                ax=ax, center=0, vmin=vmin, vmax=vmax,
                cbar=False
            )
            ax.set_yticks([])
            
            # Remove x-tick labels from all rows except the last one
            if row_idx == num_ancestors:
                ax.set_xticks(np.arange(len(env_cols)))
                perts = [cond.split('_')[2] for cond in env_cols]
                pert_labels = [pert_label_mapping[pert] for pert in perts]
                ax.set_xticklabels(pert_labels, rotation=90)
            else:
                ax.set_xticks([])
                ax.set_xticklabels([])

            if df.shape[0] > 3:
                ax.hlines(y=3, xmin=0, xmax=df.shape[1] - 1, color='white', linewidth=2)


        # Add ancestor color bar
        ancestor_ax = axs[row_idx, 0]
        ancestor_ax.set_xticks([])
        ancestor_ax.set_yticks([])
        ancestor_ax.set_ylabel(ancestor)
        # ancestor_ax.imshow(
        #     [[ancestor_colors[ancestor]] for _ in range(len(df))],
        #     aspect='auto', interpolation='nearest'
        # )
        ancestor_ax.imshow([[ancestor_colors[ancestor]] for _ in range(len(df))],
        aspect='auto', interpolation='nearest', cmap=ListedColormap(list(ancestor_colors.values())))
        ancestor_ax.spines[:].set_visible(False)  # Remove axis outline
        ancestor_ax.set_frame_on(False)  # Remove any remaining frame


    # Add color bar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cbar_ax.spines[:].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap='mako', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Fitness relative to WT')
    cbar.set_ticks([-2, -1, 0, 1, 1.5])
    cbar.set_ticklabels(['< -2', '-1', '0', '1', '1.5'])
    cbar.outline.set_visible(False)


    return fig


def main(fitness_df, bc_counts, rebarcoding_source_mutants, environment_dict, 
         subset_size=5, subset_heatmap=True, num_conditions=3):
    """
    Main function to create the visualization
    """
    # Prepare data
    ancestor_dfs, anc_list = prepare_data(fitness_df, bc_counts, subset_size, subset_heatmap)
    wt_df = prepare_wt_data(fitness_df, bc_counts, rebarcoding_source_mutants)

    # Get color schemes
    ancestor_colors, gene_colors = create_color_schemes(anc_list, pd.concat(ancestor_dfs.values()))

    # Calculate global min and max
    vmin = min(df[env].min().min() for df in ancestor_dfs.values() for env in environment_dict.values())
    vmax = max(df[env].max().max() for df in ancestor_dfs.values() for env in environment_dict.values())

    # Create plot
    fig = plot_heatmap(wt_df, ancestor_dfs, environment_dict, ancestor_colors, gene_colors, 
                      vmin, vmax, num_conditions)
    
    return fig

# Usage example:
fig = main(fitness_df, bc_counts, rebarcoding_source_mutants, environment_dict, 
          subset_size=3, subset_heatmap=True, num_conditions=3)
# plt.tight_layout()
plt.savefig(f'../plots/heatmap_feb10_pretty.png', dpi=300)
# plt.show()
# change order of perturbations !!!!!!!
# calculate t score instead of z scores???

# print([col for col in bc_counts if 'Batch' not in col])
# print(bc_counts[bc_counts['barcode'].isin(mutant_dict['anc: IRA1_NON'])]['additional_muts'].value_counts())

grant_plosbio_df = pd.read_csv('/Users/olivia/Desktop/journal.pbio.3002848.s013.csv')
for ancestor in 'IRA1_NON', 'IRA1_MIS', 'CYR1', 'TOR1', 'GPB2':
    print(ancestor)
    for evolution_condition in 'Evo1D', 'Evo2D':
        print(evolution_condition)
        print(grant_plosbio_df[(grant_plosbio_df['ancestor'] == ancestor) & (grant_plosbio_df['evolution_condition'] == evolution_condition)]['gene'].value_counts())
    # print(grant_plosbio_df[grant_plosbio_df['ancestor'] == ancestor]['gene'].value_counts())
