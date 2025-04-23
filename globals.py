import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import seaborn as sns
import torch
import networkx as nx
import random
import math



# set seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# set plotting params
plt.rcParams['figure.figsize'] = (10, 6)
non_neurons_cmap = 'Oranges'
neurons_cmap = 'Blues'



### phyisical constants ###

# cell size
cell_radius = 1.0 # units: ???
cell_volume = 1.0 # units: ???
cell_initial_vertex_length = math.sqrt(3) * cell_radius
cell_initial_surface_area = 3 * math.sqrt(3) / 2 * cell_initial_vertex_length**2
cell_initial_height = cell_volume / cell_initial_surface_area

# probs
neuron_prob = 0.5 # probability of a cell being a neuron

# mechanical properties
spring_constant_marginal = 1.0 
spring_constant_boundary = 1.0
spring_constant_internal = 0.5
line_tension_constant = 1.0
