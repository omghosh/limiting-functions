# let's make the bar plots 
import numpy as np
import matplotlib.pyplot as plt
from functions import *
from scipy.linalg import lstsq
from sklearn.metrics import r2_score
import seaborn as sns
from scipy.stats import spearmanr

plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 12

np.random.seed(100)

colors=[ (0.15, 0.25, 0.21) ,
(0.24, 0.39, 0.34) ,
(0.55, 0.66, 0.56),
(0.65, 0.77, 0.66),
(0.75, 0.87, 0.76) ,
(0.85, 0.93, 0.86),'gray', 'lightgray' ]

def plot(t1_var_explained, f_var_explained, t2_var_explained, title):
    # colors = sns.color_palette('pastel', k) + ['crimson']+['lightgray']

    plt.figure(figsize=(6, 8) ,dpi=300)
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
            if i ==k:
                plt.bar(x_pos, diffs[i], bottom=bottom, color=env_color_dict['Salt'], hatch = '//', edgecolor = 'white',  width=bar_width)
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
            plt.fill_between(x_vals, bottom_curve, top_curve, color=env_color_dict['Salt'], hatch = '//', edgecolor = 'white', alpha=0.3)
            
            # Connect second and third bar chart - using the edges of the bars
            x_vals = np.linspace(x_right_edges[1], x_left_edges[2], 100)
            bottom_curve = np.linspace(bottoms[1][i], bottoms[2][i], 100)
            top_curve = np.linspace(tops[1][i], tops[2][i], 100)
            plt.fill_between(x_vals, bottom_curve, top_curve, color=env_color_dict['Salt'], hatch = '//', edgecolor = 'white', alpha=0.3)

        else:
            x_vals = np.linspace(x_right_edges[0], x_left_edges[1], 100)
            bottom_curve = np.linspace(bottoms[0][i], bottoms[1][i], 100)
            top_curve = np.linspace(tops[0][i], tops[1][i], 100)
            plt.fill_between(x_vals, bottom_curve, top_curve, color=colors[i],  alpha=0.3)
            
            # Connect second and third bar chart - using the edges of the bars
            x_vals = np.linspace(x_right_edges[1], x_left_edges[2], 100)
            bottom_curve = np.linspace(bottoms[1][i], bottoms[2][i], 100)
            top_curve = np.linspace(tops[1][i], tops[2][i], 100)
            plt.fill_between(x_vals, bottom_curve, top_curve, color=colors[i], alpha=0.3)

    # Add labels and customize
    plt.ylabel('Fraction of Variance Explained', fontsize=14)
    plt.xticks(x_positions, [f'Target 1', f'Focal',f'Target 2'], fontsize=12)
    plt.ylim(0, 1)

    # Add titles and adjust layout
    plt.tight_layout()

    plt.savefig(f'plots/{title}.png')
    # plt.show()
    plt.close()


k = 6


# Full overlap 

t1_var_explained = [0.4, 0.2, 0.1, 0.05, 0.03, 0.001,0]
gray_part_t1 = 1-np.sum(np.array(t1_var_explained))

noise1 = np.random.randn(k+1)*0
noise2 = np.random.randn(k+1)*0

f_var_explained = t1_var_explained+ noise1
gray_part_f =  1-np.sum(np.array(f_var_explained))

t2_var_explained = [0.1, 0.05, 0.5, 0.01,0.04,0.08,0]
gray_part_2 =  1-np.sum(np.array(t2_var_explained))

t1_var_explained= np.append(t1_var_explained,gray_part_t1)
f_var_explained=np.append(f_var_explained,gray_part_f)
t2_var_explained=np.append(t2_var_explained,gray_part_2)

plot(t1_var_explained, f_var_explained, t2_var_explained, 'fig5_strict_modularity')


## No overlap 

t1_var_explained = [0.4, 0.2, 0.1, 0.05, 0.03, 0.001,0]
gray_part_t1 = 1-np.sum(np.array(t1_var_explained))


noise1 = np.abs(np.random.randn(k)*0.01)
noise2 = np.abs(np.random.randn(k)*0.01)
print(noise1)
f_var_explained = noise1
gray_part_f =  1-np.sum(np.array(t1_var_explained))

t2_var_explained = noise2
gray_part_2 =  1-np.sum(np.array(t1_var_explained))

unexplained_variation_f = 1-np.sum(f_var_explained)- gray_part_f
unexplained_variation_t2 = 1-np.sum(t2_var_explained)-gray_part_2

t1_var_explained= np.append(t1_var_explained,gray_part_t1)
f_var_explained=np.append(np.append(f_var_explained,unexplained_variation_f),gray_part_f)
t2_var_explained=np.append(np.append(t2_var_explained,unexplained_variation_t2), gray_part_2)

plot(f_var_explained,t1_var_explained, t2_var_explained, 'fig5_no_overlap')

# partial overlap 

t1_var_explained = [0.4, 0.2, 0.1, 0.05, 0.03,  0.001]
gray_part_t1 = 1-np.sum(np.array(t1_var_explained))


noise1 = np.abs(np.random.randn(k)*0.01)
noise2 = np.array(t1_var_explained)/2+(np.random.randn(k)*0.01)
print(noise1)
f_var_explained = noise1
gray_part_f =  1-np.sum(np.array(t1_var_explained))

t2_var_explained = noise2
gray_part_2 =  1-np.sum(np.array(t1_var_explained))

unexplained_variation_f = 1-np.sum(f_var_explained)- gray_part_f
unexplained_variation_t2 = 1-np.sum(t2_var_explained)-gray_part_2

t1_var_explained= np.append(np.append(t1_var_explained,0),gray_part_t1)
f_var_explained=np.append(np.append(f_var_explained,unexplained_variation_f),gray_part_f)
t2_var_explained=np.append(np.append(t2_var_explained,unexplained_variation_t2), gray_part_2)

plot(f_var_explained,t1_var_explained, t2_var_explained, 'fig5_partial_overlap')
