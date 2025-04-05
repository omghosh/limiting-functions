import numpy as np 
import pandas as pd
from functions import *
import seaborn as sns
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors


import sys
from scipy import stats

env_color_dict = {'2Day': (0.77, 0.84, 0.75), '1Day': (0.55, 0.6, 0.98), 'Salt': (1, 0.59, 0.55)}

bc_counts, fitness_df, grants_df_with_barcode_df, full_df = create_full_fitness_dataframe()
organized_perturbation_fitness_df= create_delta_fitness_matrix(batches, fitness_df, environment_dict)


print(organized_perturbation_fitness_df)